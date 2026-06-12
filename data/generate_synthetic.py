"""
generate_synthetic.py — Create synthetic ATAC-seq training data

Biology:
    ATAC-seq peaks (open chromatin) are enriched for transcription factor
    binding motifs. We simulate this by:
      - Positive sequences: random DNA with known TF motifs inserted
      - Negative sequences: random DNA with no motifs

    Real TF motifs used (JASPAR database consensus sequences):
      AP-1  (TGASTCA)   — pioneer TF, major driver of chromatin opening
      ETS   (GAGGAAGT)  — important in immune and cancer cells
      CTCF  (CCGCGAGGNGGCAG) — chromatin organizer, boundary element
      GATA  (WGATAR)    — hematopoietic TF
      SP1   (GGGCGG)    — ubiquitous activator

Run:
    python data/generate_synthetic.py

Output:
    data/train.npz, data/val.npz, data/test.npz
    Each file contains:
        sequences: (N, 4, 500) float32  — one-hot encoded DNA
        labels:    (N,)        int64    — 0=closed, 1=open chromatin
"""

import os
import sys
import random
import numpy as np

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.utils import random_sequence, insert_motif, one_hot_encode, set_seed

# ── Configuration ──────────────────────────────────────────────────────────────
SEQ_LENGTH   = 500      # bp per training example (scBasset uses 1344; 500 is good for learning)
N_TOTAL      = 10_000   # total sequences (train + val + test)
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
# Test fraction = 1 - TRAIN_FRAC - VAL_FRAC = 0.15
SEED         = 42
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__))

# ── TF binding motifs ──────────────────────────────────────────────────────────
# These are real consensus sequences from the JASPAR database.
# W = A or T,  S = C or G,  N = any base,  R = A or G
# For simplicity we use the fixed consensus (no IUPAC ambiguity codes).

MOTIFS = {
    "AP1":  "TGAGTCA",      # AP-1 family (FOS/JUN) — pioneer chromatin opener
    "ETS":  "GAGGAAGT",     # ETS family — immune / cancer relevance
    "CTCF": "CCGCGAGGCGGCAG",  # CTCF — boundary / insulator element
    "GATA": "TGATAA",       # GATA factors — hematopoietic lineage
    "SP1":  "GGGCGG",       # SP1 — ubiquitous activator
}

MOTIF_LIST = list(MOTIFS.values())


def make_positive_sequence(length: int) -> str:
    """
    Positive example: random background sequence with 1–3 TF motifs inserted.
    This simulates an open chromatin region containing TF binding sites.
    """
    seq = random_sequence(length)
    # Insert 1–3 motifs at random positions
    n_motifs = random.randint(1, 3)
    for _ in range(n_motifs):
        motif = random.choice(MOTIF_LIST)
        seq = insert_motif(seq, motif)
    return seq


def make_negative_sequence(length: int) -> str:
    """
    Negative example: purely random sequence.
    Simulates closed chromatin with no TF binding sites.
    """
    return random_sequence(length)


def generate_dataset(n_total: int, seq_length: int):
    """
    Generate balanced positive/negative sequences.
    Returns (sequences_array, labels_array).
    """
    n_pos = n_total // 2
    n_neg = n_total - n_pos

    print(f"  Generating {n_pos} positive sequences (open chromatin)...")
    pos_seqs = [make_positive_sequence(seq_length) for _ in range(n_pos)]

    print(f"  Generating {n_neg} negative sequences (closed chromatin)...")
    neg_seqs = [make_negative_sequence(seq_length) for _ in range(n_neg)]

    all_seqs  = pos_seqs + neg_seqs
    all_labels = [1] * n_pos + [0] * n_neg

    # Shuffle together
    combined = list(zip(all_seqs, all_labels))
    random.shuffle(combined)
    all_seqs, all_labels = zip(*combined)

    # One-hot encode all sequences → shape (N, 4, seq_length)
    print("  One-hot encoding sequences...")
    encoded = np.stack([one_hot_encode(s) for s in all_seqs], axis=0)  # (N, 4, L)
    labels  = np.array(all_labels, dtype=np.int64)

    return encoded, labels


def split_and_save(encoded, labels, output_dir, train_frac, val_frac):
    """Split into train/val/test and save as .npz files."""
    N = len(labels)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    splits = {
        "train": (encoded[:n_train],          labels[:n_train]),
        "val":   (encoded[n_train:n_train+n_val], labels[n_train:n_train+n_val]),
        "test":  (encoded[n_train+n_val:],     labels[n_train+n_val:]),
    }

    os.makedirs(output_dir, exist_ok=True)
    for split_name, (X, y) in splits.items():
        path = os.path.join(output_dir, f"{split_name}.npz")
        np.savez_compressed(path, sequences=X, labels=y)
        pos = y.sum()
        print(f"  Saved {split_name}: {len(y)} sequences "
              f"({pos} open / {len(y)-pos} closed) → {path}")


if __name__ == "__main__":
    set_seed(SEED)
    print(f"\nGenerating synthetic ATAC-seq dataset")
    print(f"  Sequence length : {SEQ_LENGTH} bp")
    print(f"  Total sequences : {N_TOTAL}")
    print(f"  Motifs used     : {', '.join(MOTIFS.keys())}\n")

    encoded, labels = generate_dataset(N_TOTAL, SEQ_LENGTH)
    split_and_save(encoded, labels, OUTPUT_DIR, TRAIN_FRAC, VAL_FRAC)

    print(f"\nDone. Data saved to {OUTPUT_DIR}/")
    print("Next step: python train_model.py")
