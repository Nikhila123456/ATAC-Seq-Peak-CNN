"""
train.py — Training loop for ATAC-seq CNN

Training loop structure (runs every epoch):
    1. For each batch in train_loader:
        a. Forward pass: model(X) → predictions
        b. Compute loss: how wrong are the predictions?
        c. Backward pass: compute gradients via backpropagation
        d. Update weights: optimizer takes a step
    2. Evaluate on validation set (no gradient computation)
    3. Save model if validation loss improved (early stopping)

Key concepts:
    - Loss function: BCEWithLogitsLoss — binary cross-entropy for 0/1 labels
    - Optimizer: Adam — adaptive learning rate, works well out of the box
    - Learning rate scheduler: ReduceLROnPlateau — reduces LR when val_loss stalls
    - Early stopping: halts training if model stops improving (prevents overfitting)
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


def train_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    device:      torch.device,
    epoch:       int,
) -> float:
    """
    Run one training epoch.
    Returns: mean training loss for this epoch.
    """
    model.train()   # switch to training mode (enables Dropout, BatchNorm updates)
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (sequences, labels) in enumerate(loader):
        # Move data to GPU if available
        sequences = sequences.to(device)          # (batch, 4, seq_len)
        labels    = labels.to(device).unsqueeze(1) # (batch, 1)

        # ── Forward pass ──────────────────────────────────────────────────────
        optimizer.zero_grad()           # clear gradients from previous step
        logits = model(sequences)       # raw scores, shape (batch, 1)
        loss   = criterion(logits, labels)  # compare to true labels

        # ── Backward pass ─────────────────────────────────────────────────────
        loss.backward()                 # compute gradients via backprop

        # Gradient clipping: prevents exploding gradients (good practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()                # update model weights

        total_loss += loss.item()

        # Print progress every 20% of epoch
        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{n_batches}] "
                  f"loss: {loss.item():.4f}")

    return total_loss / n_batches


@torch.no_grad()   # no gradient computation during validation (saves memory + speed)
def validate_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    """
    Run one validation epoch.
    Returns: (val_loss, val_accuracy)
    """
    model.eval()    # switch to eval mode (disables Dropout, uses running BatchNorm stats)
    total_loss    = 0.0
    correct       = 0
    total         = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels    = labels.to(device).unsqueeze(1)

        logits    = model(sequences)
        loss      = criterion(logits, labels)
        total_loss += loss.item()

        # Convert logits → probabilities → binary predictions
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.numel()

    val_loss = total_loss / len(loader)
    val_acc  = correct / total
    return val_loss, val_acc


def train(
    model:        nn.Module,
    loaders:      Dict[str, DataLoader],
    device:       torch.device,
    n_epochs:     int   = 30,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int   = 5,
    save_path:    str   = "results/best_model.pt",
) -> Dict[str, List[float]]:
    """
    Full training loop with early stopping and learning rate scheduling.

    Args:
        model       : the ATACPeakCNN model
        loaders     : dict with 'train' and 'val' DataLoaders
        device      : cpu or cuda
        n_epochs    : maximum training epochs
        lr          : initial learning rate
        weight_decay: L2 regularization strength
        patience    : early stopping patience (epochs without improvement)
        save_path   : where to save the best model checkpoint

    Returns:
        history dict with train_loss, val_loss, val_acc per epoch
    """
    # ── Optimizer ────────────────────────────────────────────────────────────
    # Adam: adaptive learning rate optimizer; works well without much tuning
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy in one numerically
    # stable operation. Use this instead of sigmoid → BCELoss.
    criterion = nn.BCEWithLogitsLoss()

    # ── Learning rate scheduler ───────────────────────────────────────────────
    # Reduces LR by factor 0.5 if val_loss doesn't improve for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Training state ────────────────────────────────────────────────────────
    history       = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\nStarting training for up to {n_epochs} epochs")
    print(f"Device: {device} | LR: {lr} | Patience: {patience}\n")
    print("=" * 60)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Training
        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device, epoch)

        # Validation
        val_loss, val_acc = validate_epoch(model, loaders["val"], criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch:3d}/{n_epochs} | "
              f"train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_acc: {val_acc:.4f} | "
              f"{elapsed:.1f}s")

        # ── Save best model ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":   val_loss,
                "val_acc":    val_acc,
            }, save_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

        print("-" * 60)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {save_path}")
    return history
