"""
predict.py — Run the trained model on new DNA sequences

Usage:
    # Predict a single sequence
    python predict.py --sequence "ATGCATGCTGAGTCAATGCATGC"

    # Predict multiple sequences from a FASTA file
    python predict.py --fasta my_sequences.fasta

    # Use a specific model checkpoint
    python predict.py --sequence "ATGC..." --model results/best_model.pt
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.model import ATACPeakCNN
from src.utils import one_hot_encode, get_device

SEQ_LENGTH  = 500
MODEL_PATH  = "results/best_model.pt"


def pad_or_trim(sequence: str, target_length: int) -> str:
    """Pad with N or trim sequence to exactly target_length."""
    if len(sequence) < target_length:
        pad = "N" * (target_length - len(sequence))
        sequence = sequence + pad
    else:
        sequence = sequence[:target_length]
    return sequence


def load_model(model_path: str, device: torch.device) -> ATACPeakCNN:
    """Load trained model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first."
        )
    checkpoint = torch.load(model_path, map_location=device)
    model = ATACPeakCNN(seq_length=SEQ_LENGTH).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {model_path} "
          f"(trained for {checkpoint['epoch']} epochs, "
          f"val_loss: {checkpoint['val_loss']:.4f})")
    return model


def predict_sequences(sequences: list, model: ATACPeakCNN, device: torch.device):
    """
    Predict chromatin accessibility for a list of DNA sequences.

    Returns:
        list of dicts with sequence, probability, and prediction label
    """
    results = []
    with torch.no_grad():
        for seq in sequences:
            # Pad/trim to model's expected length
            seq_padded = pad_or_trim(seq, SEQ_LENGTH)
            # One-hot encode: (4, 500)
            encoded = one_hot_encode(seq_padded)
            # Add batch dimension: (1, 4, 500)
            tensor  = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(device)
            # Forward pass
            logit   = model(tensor)
            prob    = torch.sigmoid(logit).item()
            results.append({
                "sequence":    seq[:50] + "..." if len(seq) > 50 else seq,
                "probability": round(prob, 4),
                "prediction":  "OPEN (peak)" if prob >= 0.5 else "CLOSED",
                "confidence":  "high" if abs(prob - 0.5) > 0.3 else "low",
            })
    return results


def read_fasta(fasta_path: str):
    """Simple FASTA reader. Returns list of (header, sequence) tuples."""
    sequences = []
    header, seq = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    sequences.append((header, "".join(seq)))
                header, seq = line[1:], []
            else:
                seq.append(line.upper())
    if header:
        sequences.append((header, "".join(seq)))
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description="Predict ATAC-seq peak accessibility from DNA sequence"
    )
    parser.add_argument("--sequence", type=str, default=None,
                        help="Single DNA sequence string")
    parser.add_argument("--fasta",    type=str, default=None,
                        help="Path to FASTA file with sequences")
    parser.add_argument("--model",    type=str, default=MODEL_PATH,
                        help="Path to trained model checkpoint")
    args = parser.parse_args()

    if not args.sequence and not args.fasta:
        # Demo mode: show predictions for sequences with and without motifs
        print("No input provided. Running demo with example sequences...\n")
        test_seqs = [
            "A" * 246 + "TGAGTCA" + "A" * 247,   # AP-1 motif → likely open
            "C" * 246 + "GAGGAAGT" + "C" * 246,  # ETS motif → likely open
            "ATCGATCG" * 62 + "ATCG",             # random → likely closed
        ]
        names = ["AP-1 motif (positive)", "ETS motif (positive)", "Random (negative)"]
        seqs  = test_seqs
    elif args.sequence:
        seqs  = [args.sequence]
        names = ["Input sequence"]
    else:
        fasta_entries = read_fasta(args.fasta)
        names = [h for h, _ in fasta_entries]
        seqs  = [s for _, s in fasta_entries]

    device = get_device()
    model  = load_model(args.model, device)

    results = predict_sequences(seqs, model, device)

    print(f"\n{'Name':<30} {'P(open)':<10} {'Prediction':<15} {'Confidence'}")
    print("-" * 70)
    for name, res in zip(names, results):
        print(f"{name:<30} {res['probability']:<10} {res['prediction']:<15} {res['confidence']}")


if __name__ == "__main__":
    main()
