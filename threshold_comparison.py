"""
threshold_comparison.py
Run from inside atac_cnn/:
    python threshold_comparison.py

Prints a table comparing sensitivity, specificity, and accuracy
at different classification thresholds, and saves a plot.
"""

import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.utils    import get_device
from src.model    import ATACPeakCNN
from src.dataset  import make_dataloaders
from src.evaluate import get_predictions, compute_metrics


def main():
    # ── Load model ─────────────────────────────────────────────────────────────
    device = get_device()
    loaders = make_dataloaders("data", batch_size=64)

    model = ATACPeakCNN(seq_length=500).to(device)
    checkpoint = torch.load("results/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch']}\n")

    # ── Get predictions once ───────────────────────────────────────────────────
    probs, labels = get_predictions(model, loaders["test"], device)

    # ── Compare thresholds ────────────────────────────────────────────────────
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    print(f"{'Threshold':<12} {'Sensitivity':<14} {'Specificity':<14} "
          f"{'Accuracy':<12} {'Precision':<12} {'F1':<8}")
    print("-" * 72)

    rows = []
    for t in thresholds:
        m = compute_metrics(probs, labels, threshold=t)

        # Precision = TP / (TP + FP)
        precision = (m["tp"] / (m["tp"] + m["fp"])
                     if (m["tp"] + m["fp"]) > 0 else 0.0)

        # F1 = harmonic mean of precision and recall (sensitivity)
        f1 = (2 * precision * m["sensitivity"] /
              (precision + m["sensitivity"])
              if (precision + m["sensitivity"]) > 0 else 0.0)

        marker = " <-- default" if t == 0.5 else ""
        print(f"{t:<12.1f} {m['sensitivity']:<14.4f} {m['specificity']:<14.4f} "
              f"{m['accuracy']:<12.4f} {precision:<12.4f} {f1:<8.4f}{marker}")

        rows.append({
            "threshold":   t,
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "accuracy":    round(m["accuracy"],    4),
            "precision":   round(precision,        4),
            "f1":          round(f1,               4),
            "auroc":       round(m["auroc"],       4),
        })

    # ── Save table as JSON ────────────────────────────────────────────────────
    with open("results/threshold_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved -> results/threshold_comparison.json")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ts   = [r["threshold"]   for r in rows]
    sens = [r["sensitivity"] for r in rows]
    spec = [r["specificity"] for r in rows]
    acc  = [r["accuracy"]    for r in rows]
    f1s  = [r["f1"]          for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, sens, "o-", color="#0D6E72", lw=2, label="Sensitivity (recall)")
    ax.plot(ts, spec, "s-", color="#1A3A52", lw=2, label="Specificity")
    ax.plot(ts, acc,  "^-", color="#E65100", lw=2, label="Accuracy")
    ax.plot(ts, f1s,  "D-", color="#6A1B9A", lw=2, label="F1 score")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Default (0.5)")
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Sensitivity / Specificity / Accuracy vs Threshold", fontsize=13)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/threshold_comparison.png", dpi=150)
    plt.close()
    print("Saved -> results/threshold_comparison.png")

    # ── Print README-ready markdown table ────────────────────────────────────
    print("\n--- Copy this into your README ---")
    print(f"\n| Threshold | Sensitivity | Specificity | Accuracy | F1 |")
    print(f"|-----------|-------------|-------------|----------|----|")
    for r in rows:
        marker = " ← default" if r["threshold"] == 0.5 else ""
        print(f"| {r['threshold']}       | "
              f"{r['sensitivity']*100:.1f}%       | "
              f"{r['specificity']*100:.1f}%       | "
              f"{r['accuracy']*100:.1f}%     | "
              f"{r['f1']:.3f}{marker} |")
    print(f"\nAUROC: {rows[0]['auroc']} (threshold-independent)")


if __name__ == "__main__":
    main()
