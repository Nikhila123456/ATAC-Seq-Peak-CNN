"""
model.py — CNN architecture for ATAC-seq peak prediction

Architecture Overview (scBasset-inspired):

    DNA sequence (500 bp)
         │
         ▼
    One-hot encoding  →  (batch, 4, 500)
         │
         ▼
    Stem conv block   →  (batch, 128, 500)   learns simple motifs
         │
         ▼
    Residual blocks   →  (batch, 256, 125)   learns complex patterns
    (×4, with pooling)                        with skip connections
         │
         ▼
    Global Avg Pool   →  (batch, 256)         position-invariant
         │
         ▼
    Dense layers      →  (batch, 64)
         │
         ▼
    Output (sigmoid)  →  (batch, 1)           probability of open chromatin

Key architectural choices explained below each layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    A single convolutional block:
        Conv1d → BatchNorm → ReLU → Dropout

    Why each piece?
    - Conv1d:     scans the DNA sequence with a fixed-width filter,
                  learning to detect sequence motifs (like TGAGTCA for AP-1)
    - BatchNorm:  normalizes activations so training is stable and fast
    - ReLU:       non-linearity — allows the network to learn complex patterns
    - Dropout:    randomly zeros activations during training → prevents
                  memorizing training data (overfitting)
    """

    def __init__(
        self,
        in_channels:  int,      # number of input feature maps
        out_channels: int,      # number of filters (output feature maps)
        kernel_size:  int = 8,  # motif width in bp (AP-1 = 7bp, CTCF = 19bp)
        dropout:      float = 0.1,
        dilation:     int = 1,  # dilation > 1 expands receptive field
    ):
        super().__init__()

        # padding = 'same' keeps sequence length unchanged after conv
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv  = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,         # BatchNorm has its own bias term
        )
        self.bn      = nn.BatchNorm1d(out_channels)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: (batch, channels, seq_length)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with a skip connection:

        input ──────────────────────────────┐
          │                                  │
          ▼                                  │
        ConvBlock (main path)                │ (skip path — 1×1 conv if
          │                                  │  channels change)
          ▼                                  │
        ConvBlock                            │
          │                                  │
          └──────── + (element-wise add) ────┘
                    │
                    ▼
                  output

    Why residual connections?
        Deep networks suffer from vanishing gradients — signals become tiny
        as they propagate backwards. Skip connections create a "highway"
        that lets gradients flow directly, enabling much deeper networks.
        This is the core innovation in ResNet (He et al. 2016).
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, dropout)
        self.conv2 = ConvBlock(channels, channels, kernel_size, dropout)

    def forward(self, x):
        residual = x                    # save the input (skip path)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual            # add skip connection
        return out


# ── Main model ──────────────────────────────────────────────────────────────────

class ATACPeakCNN(nn.Module):
    """
    CNN for ATAC-seq peak prediction from DNA sequence.

    Input:  (batch_size, 4, seq_length)  — one-hot encoded DNA
    Output: (batch_size, 1)              — P(open chromatin), sigmoid-activated
    """

    def __init__(
        self,
        seq_length:   int   = 500,   # input sequence length in bp
        n_filters_1:  int   = 128,   # filters in stem conv (learns simple motifs)
        n_filters_2:  int   = 256,   # filters in residual blocks (complex patterns)
        n_residual:   int   = 4,     # number of residual blocks
        kernel_size_stem: int = 15,  # wide kernel for stem (capture long motifs)
        kernel_size_res:  int = 3,   # narrow kernel for residual (local refinement)
        dropout:      float = 0.1,
        fc_size:      int   = 64,    # fully connected layer size
    ):
        super().__init__()
        self.seq_length = seq_length

        # ── 1. Stem block ────────────────────────────────────────────────────
        # First layer sees raw one-hot sequence (4 channels = A, C, G, T).
        # Wide kernel (15bp) captures full TF binding motifs in one filter.
        # This is equivalent to a position weight matrix (PWM) scan.
        self.stem = ConvBlock(
            in_channels=4,
            out_channels=n_filters_1,
            kernel_size=kernel_size_stem,
            dropout=dropout,
        )

        # ── 2. Max-pool to reduce sequence length ────────────────────────────
        # 500 → 250 bp; reduces computation and forces the network to learn
        # position-invariant features (a motif anywhere in the window counts)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ── 3. Expansion conv: 128 → 256 channels ───────────────────────────
        # More filters = more patterns the network can detect
        self.expand = ConvBlock(
            in_channels=n_filters_1,
            out_channels=n_filters_2,
            kernel_size=kernel_size_stem,
            dropout=dropout,
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Sequence is now 500 → 250 → 125 bp

        # ── 4. Residual blocks ───────────────────────────────────────────────
        # Learn complex combinatorial patterns (e.g. AP-1 near ETS motif)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(n_filters_2, kernel_size_res, dropout)
              for _ in range(n_residual)]
        )

        # ── 5. Global Average Pooling ────────────────────────────────────────
        # Collapse the entire sequence dimension to one value per filter.
        # This makes the prediction position-invariant: a peak-driving motif
        # contributes the same signal regardless of where it sits in the window.
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # ── 6. Fully connected head ──────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(n_filters_2, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(fc_size, 1),
        )
        # NOTE: No sigmoid here — we use BCEWithLogitsLoss during training,
        # which is numerically more stable than sigmoid + BCELoss.
        # Apply sigmoid manually during inference (predict.py).

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with He initialization (recommended for ReLU networks).
        Proper initialization prevents vanishing/exploding gradients at the start.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: (batch, 4, seq_length) one-hot encoded DNA

        Returns:
            logits: (batch, 1) raw scores (apply sigmoid for probabilities)
        """
        # Stem: learn motif patterns from raw sequence
        x = self.stem(x)        # → (batch, 128, 500)
        x = self.pool1(x)       # → (batch, 128, 250)

        # Expand: learn higher-level combinations
        x = self.expand(x)      # → (batch, 256, 250)
        x = self.pool2(x)       # → (batch, 256, 125)

        # Residual: refine complex patterns with skip connections
        x = self.residual_blocks(x)   # → (batch, 256, 125)

        # Aggregate: collapse sequence dimension
        x = self.global_avg_pool(x)   # → (batch, 256, 1)
        x = x.squeeze(-1)             # → (batch, 256)

        # Predict: dense layers to scalar output
        x = self.fc(x)          # → (batch, 1)
        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ATACPeakCNN(seq_length=500)
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")

    # Test forward pass with a batch of 4 random sequences
    x = torch.randn(4, 4, 500)   # (batch=4, channels=4, length=500)
    out = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")  # expected: (4, 1)
    print(f"Output (logits): {out.detach().squeeze().tolist()}")
