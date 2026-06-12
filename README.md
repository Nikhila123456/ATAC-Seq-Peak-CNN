# ATAC-seq Peak Prediction with CNN

A convolutional neural network that predicts chromatin accessibility (ATAC-seq peaks)
from DNA sequence — inspired by the scBasset architecture (Yuan & Kelley, Nature Methods 2022).

## What this does

Given a DNA sequence (500 bp), the model predicts whether that region will be
**open chromatin** (accessible, ATAC-seq peak) or **closed chromatin** (inaccessible).

This is the core prediction problem for sequence-to-function deep learning models
in regulatory genomics, including:
- [scBasset](https://www.nature.com/articles/s41592-022-01562-8): single-cell ATAC-seq modelling
- [Enformer](https://www.nature.com/articles/s41592-021-01252-x): gene expression prediction
- [Linder et al. 2025](https://www.nature.com/articles/s41588-024-02053-6): RNA-seq from sequence

## Why this architecture works

DNA sequence → one-hot encoding (A/C/G/T → 4-channel tensor)
→ Stacked convolutional layers (learn motif patterns at different scales)
→ Residual connections (allow deeper networks without vanishing gradients)
→ Global average pooling (position-invariant summary)
→ Dense layers → Binary output (open/closed)

The convolutional filters learn to recognize transcription factor binding motifs
(AP-1, ETS, CTCF etc.) that drive chromatin accessibility.

## Project Structure

```
atac_cnn/
├── src/
│   ├── model.py        # CNN architecture (the core of this project)
│   ├── dataset.py      # PyTorch Dataset — DNA one-hot encoding
│   ├── train.py        # Training loop with validation
│   ├── evaluate.py     # Metrics, ROC curves, confusion matrix
│   └── utils.py        # DNA utilities, reproducibility helpers
├── data/
│   └── generate_synthetic.py   # Generate training data (no downloads needed)
├── notebooks/
│   └── tutorial.ipynb  # Step-by-step walkthrough
├── train_model.py      # Main entry point — run this to train
├── predict.py          # Run predictions on new sequences
├── requirements.txt
└── environment.yml
```

## Quick Start

```bash
# 1. Set up environment
conda env create -f environment.yml
conda activate atac-cnn

# 2. Generate synthetic training data
python data/generate_synthetic.py

# 3. Train the model
python train_model.py

# 4. Evaluate results (saved to results/)
python predict.py --sequence "ATGCATGCATGC..."
```

## Understanding the Biology

**ATAC-seq** (Assay for Transposase-Accessible Chromatin using sequencing) measures
which regions of the genome are "open" and accessible to transcription factors.

Open regions contain **transcription factor binding motifs** — short sequence patterns
(6-12 bp) that TFs recognize and bind to. A CNN learns these patterns as convolutional
filters, exactly like how an image CNN learns edges and textures.

**Positive training examples**: sequences containing known accessibility motifs
(AP-1: TGASTCA, ETS: GGAA, CTCF: CCGCGNGGNGGCAG)

**Negative training examples**: random genomic sequences lacking these motifs

## Key Concepts for Interviews

- **One-hot encoding**: A → [1,0,0,0], C → [0,1,0,0], G → [0,0,1,0], T → [0,0,0,1]
- **Conv1d**: scans a 1D filter along the sequence — learns motifs of fixed width
- **Dilated convolutions**: expand receptive field to capture long-range dependencies
- **Residual connections**: skip connections that stabilize deep network training
- **Global average pooling**: makes prediction position-invariant (peak anywhere in window)
- **Attribution/saliency**: DeepLIFT/GradCAM shows which sequence positions drive predictions

## References

- Yuan H & Kelley DR. scBasset (2022) — this model's main inspiration
- Avsec Ž et al. Enformer (2021) — long-range sequence-to-expression
- Linder J et al. (2025) — RNA-seq coverage from DNA sequence
