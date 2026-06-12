import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix,
)


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    for sequences, labels in loader:
        sequences = sequences.to(device)
        logits = model(sequences)
        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    acc = (preds == labels).mean()
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": acc,
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }


def print_metrics(metrics):
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  AUROC       : {metrics['auroc']:.4f}")
    print(f"  AUPRC       : {metrics['auprc']:.4f}")
    print(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"               Pred 0    Pred 1")
    print(f"  True 0 (-)   {metrics['tn']:6d}    {metrics['fp']:6d}")
    print(f"  True 1 (+)   {metrics['fn']:6d}    {metrics['tp']:6d}")
    print("=" * 50)


def plot_roc_curve(probs, labels, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#0D6E72", lw=2, label=f"CNN (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — ATAC-seq Peak Prediction")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved -> {save_path}")


def plot_loss_curves(history, save_path):
    if not history.get("train_loss"):
        print("No history to plot — skipping loss curves")
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#1A3A52", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val", color="#0D6E72", lw=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["val_acc"], color="#0D6E72", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curves saved -> {save_path}")


def plot_prediction_distribution(probs, labels, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probs[labels == 0], bins=50, alpha=0.6, color="#D32F2F", label="Closed chromatin")
    ax.hist(probs[labels == 1], bins=50, alpha=0.6, color="#0D6E72", label="Open chromatin")
    ax.axvline(0.5, color="black", linestyle="--", label="Threshold = 0.5")
    ax.set_xlabel("Predicted P(open chromatin)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Score Distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Score distribution saved -> {save_path}")


def full_evaluation(model, loader, device, history, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    print("\nRunning full test-set evaluation...")
    probs, labels = get_predictions(model, loader, device)
    metrics = compute_metrics(probs, labels)
    print_metrics(metrics)
    plot_roc_curve(probs, labels, os.path.join(results_dir, "roc_curve.png"))
    plot_loss_curves(history, os.path.join(results_dir, "training_curves.png"))
    plot_prediction_distribution(probs, labels, os.path.join(results_dir, "score_distribution.png"))
    return metrics
