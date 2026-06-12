"""
train_model.py — Main entry point: generate data, train, and evaluate

Run this file to train the complete pipeline:
    python train_model.py

What happens:
    1. Generate synthetic ATAC-seq training data (if not already done)
    2. Load data into PyTorch DataLoaders
    3. Build the CNN model
    4. Train with early stopping
    5. Evaluate on test set and save figures to results/
"""

import os
import sys
import json
import torch

# ── Make src importable ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.utils    import set_seed, get_device
from src.model    import ATACPeakCNN
from src.dataset  import make_dataloaders
from src.train    import train
from src.evaluate import full_evaluation


# ── Configuration ──────────────────────────────────────────────────────────────
# Change these to experiment with different settings

CONFIG = {
    # Data
    "data_dir":    "data",
    "seq_length":  500,

    # Model architecture
    "n_filters_1": 128,       # stem filters (simple motifs)
    "n_filters_2": 256,       # residual block filters (complex patterns)
    "n_residual":  4,         # number of residual blocks
    "dropout":     0.1,

    # Training
    "batch_size":  64,
    "n_epochs":    30,
    "lr":          1e-3,
    "weight_decay":1e-4,
    "patience":    5,         # early stopping patience

    # Output
    "results_dir": "results",
    "model_path":  "results/best_model.pt",

    # Reproducibility
    "seed": 42,
}


def main():
    print("\n" + "=" * 60)
    print("   ATAC-seq Peak Prediction — CNN Training Pipeline")
    print("=" * 60)

    # ── Step 1: Setup ──────────────────────────────────────────────────────────
    set_seed(CONFIG["seed"])
    device = get_device()
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    # ── Step 2: Generate data if needed ───────────────────────────────────────
    train_path = os.path.join(CONFIG["data_dir"], "train.npz")
    if not os.path.exists(train_path):
        print("\nGenerating synthetic training data...")
        import subprocess
        subprocess.run([sys.executable, "data/generate_synthetic.py"], check=True)
    else:
        print(f"\nData already exists at {CONFIG['data_dir']}/ — skipping generation")

    # ── Step 3: Load data ──────────────────────────────────────────────────────
    print("\nLoading data into DataLoaders...")
    loaders = make_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
    )

    # ── Step 4: Build model ────────────────────────────────────────────────────
    print("\nBuilding CNN model...")
    model = ATACPeakCNN(
        seq_length=CONFIG["seq_length"],
        n_filters_1=CONFIG["n_filters_1"],
        n_filters_2=CONFIG["n_filters_2"],
        n_residual=CONFIG["n_residual"],
        dropout=CONFIG["dropout"],
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    history = train(
        model=model,
        loaders=loaders,
        device=device,
        n_epochs=CONFIG["n_epochs"],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"],
        save_path=CONFIG["model_path"],
    )

    # ── Step 6: Load best model and evaluate ──────────────────────────────────
    print("\nLoading best checkpoint for evaluation...")
    checkpoint = torch.load(CONFIG["model_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Best model from epoch {checkpoint['epoch']} "
          f"(val_loss: {checkpoint['val_loss']:.4f})")

    metrics = full_evaluation(
        model=model,
        loader=loaders["test"],
        device=device,
        history=history,
        results_dir=CONFIG["results_dir"],
    )

    # ── Step 7: Save config and metrics ───────────────────────────────────────
    output = {"config": CONFIG, "metrics": metrics}
    with open(os.path.join(CONFIG["results_dir"], "results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nAll results saved to {CONFIG['results_dir']}/")
    print("\nFiles created:")
    for f in os.listdir(CONFIG["results_dir"]):
        print(f"  {CONFIG['results_dir']}/{f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
