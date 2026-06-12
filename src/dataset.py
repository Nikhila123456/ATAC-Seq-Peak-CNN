"""
dataset.py — PyTorch Dataset class for ATAC-seq sequences

What is a PyTorch Dataset?
    PyTorch's Dataset class wraps your data and defines how to get one example.
    The DataLoader then batches examples automatically during training.

    You must implement two methods:
        __len__  → total number of examples
        __getitem__ → return (input, label) for index i

    The data flows like this during training:
        DataLoader → calls __getitem__(i) → returns (tensor, label)
                   → batches N examples → feeds to model
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ATACDataset(Dataset):
    """
    Dataset for ATAC-seq peak prediction.

    Each example is:
        X : torch.Tensor of shape (4, seq_length) — one-hot encoded DNA
        y : torch.Tensor of shape ()               — 0 (closed) or 1 (open)
    """

    def __init__(self, npz_path: str):
        """
        Load pre-generated .npz file containing sequences and labels.

        Args:
            npz_path: path to .npz file created by data/generate_synthetic.py
        """
        data = np.load(npz_path)
        # sequences: (N, 4, seq_length) float32
        self.sequences = torch.tensor(data["sequences"], dtype=torch.float32)
        # labels: (N,) int64
        self.labels    = torch.tensor(data["labels"],    dtype=torch.float32)

        print(f"Loaded {len(self.labels)} sequences from {npz_path}")
        print(f"  Shape: {self.sequences.shape}  "
              f"  Positive: {int(self.labels.sum())}  "
              f"  Negative: {int(len(self.labels) - self.labels.sum())}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def make_dataloaders(data_dir: str, batch_size: int = 64, num_workers: int = 0):
    """
    Create train, validation, and test DataLoaders.

    Args:
        data_dir   : directory containing train.npz, val.npz, test.npz
        batch_size : number of sequences per training batch
        num_workers: parallel data loading workers (0 = main process)

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    import os
    loaders = {}
    for split in ("train", "val", "test"):
        path    = os.path.join(data_dir, f"{split}.npz")
        dataset = ATACDataset(path)
        shuffle = (split == "train")   # only shuffle training data
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
