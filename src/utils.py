"""
utils.py — DNA sequence utilities and reproducibility helpers

Key concept: One-hot encoding
  A → [1, 0, 0, 0]
  C → [0, 1, 0, 0]
  G → [0, 0, 1, 0]
  T → [0, 0, 0, 1]
  N → [0, 0, 0, 0]  (unknown base)

This transforms a DNA string into a (4 × sequence_length) matrix,
which is the input format for Conv1d layers in PyTorch.
"""

import random
import numpy as np
import torch


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds so experiments are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return GPU if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ── One-hot encoding ───────────────────────────────────────────────────────────

# Map each nucleotide to its column index in the one-hot matrix
NUCLEOTIDE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(sequence: str) -> np.ndarray:
    """
    Convert a DNA string to a one-hot encoded numpy array.

    Args:
        sequence: DNA string, e.g. "ATGCATGC"

    Returns:
        np.ndarray of shape (4, len(sequence))
        Row 0 = A, Row 1 = C, Row 2 = G, Row 3 = T

    Example:
        >>> one_hot_encode("ACG")
        array([[1, 0, 0],   # A
               [0, 1, 0],   # C
               [0, 0, 1],   # G
               [0, 0, 0]])  # T
    """
    seq = sequence.upper()
    L = len(seq)
    # Start with all zeros — unknown bases (N) stay zero
    encoding = np.zeros((4, L), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in NUCLEOTIDE_TO_IDX:
            encoding[NUCLEOTIDE_TO_IDX[base], i] = 1.0
    return encoding


def decode_one_hot(encoding: np.ndarray) -> str:
    """
    Convert a one-hot matrix back to a DNA string.
    Useful for inspecting what the model learned.
    """
    idx_to_nuc = {0: "A", 1: "C", 2: "G", 3: "T"}
    seq = []
    for pos in range(encoding.shape[1]):
        col = encoding[:, pos]
        if col.sum() == 0:
            seq.append("N")
        else:
            seq.append(idx_to_nuc[int(np.argmax(col))])
    return "".join(seq)


# ── DNA sequence generation ────────────────────────────────────────────────────

BASES = ["A", "C", "G", "T"]


def random_sequence(length: int) -> str:
    """Generate a random DNA sequence of given length."""
    return "".join(random.choice(BASES) for _ in range(length))


def insert_motif(sequence: str, motif: str, position: int = None) -> str:
    """
    Insert a known TF binding motif into a sequence at a given position.
    If position is None, insert at a random position.

    Biology context:
        Transcription factor binding motifs are short (6–15 bp) sequence patterns
        that drive chromatin accessibility. The CNN should learn to detect these.
    """
    seq = list(sequence)
    if position is None:
        # Avoid the very edges
        position = random.randint(10, len(sequence) - len(motif) - 10)
    seq[position: position + len(motif)] = list(motif)
    return "".join(seq)


def reverse_complement(sequence: str) -> str:
    """
    Return the reverse complement of a DNA sequence.
    Important because TFs can bind both strands.
    """
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(complement[b] for b in reversed(sequence.upper()))
