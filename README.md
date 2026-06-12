<div align="center">

# 🧬 ATAC-Seq Peak Prediction with CNN

**Predicting chromatin accessibility from DNA sequence using deep learning**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AUROC](https://img.shields.io/badge/AUROC-0.908-blue)](results/roc_curve.png)

*Inspired by [scBasset](https://www.nature.com/articles/s41592-022-01562-8) (Yuan & Kelley, Nature Methods 2022)*

</div>

---

## 📌 Overview

This repository implements a **convolutional neural network (CNN)** that learns to predict
**open chromatin regions (ATAC-seq peaks)** directly from 500 bp DNA sequence windows.

The model learns transcription factor binding motifs as convolutional filters — the same
sequence patterns that drive chromatin accessibility in living cells. This is the core
prediction problem in regulatory genomics, underpinning models like
[Enformer](https://www.nature.com/articles/s41592-021-01252-x) and
[scBasset](https://www.nature.com/articles/s41592-022-01562-8).

---

## 🔬 Biology

Open chromatin regions contain **TF binding motifs** that displace nucleosomes,
making DNA accessible for gene regulation. The CNN learns these patterns as filters:

| TF Family | Motif | Role |
|-----------|-------|------|
| **AP-1** (FOS/JUN) | `TGAGTCA` | Pioneer — drives chromatin opening |
| **ETS** | `GAGGAAGT` | Immune & cancer accessibility |
| **CTCF** | `CCGCGAGGCGGCAG` | Chromatin boundary/insulator |
| **GATA** | `TGATAA` | Hematopoietic specification |
| **SP1** | `GGGCGG` | Ubiquitous activator |

---

## 🏗️ Model Architecture

```
DNA (500 bp) → One-hot encode (4 × 500)
    │
    ├─ Stem Conv (128 filters, k=15)   # motif detection
    ├─ MaxPool (/2)
    ├─ Expansion Conv (256 filters)    # complex combinations
    ├─ MaxPool (/2)
    ├─ Residual Blocks × 4             # deep feature learning
    ├─ Global Average Pool             # position-invariant
    └─ Dense → Sigmoid                 # P(open chromatin)

Parameters: ~2.1 million
```

**Key choices:** residual skip connections (He et al. 2016),
BCEWithLogitsLoss for numerical stability,
global average pooling for position invariance.

---

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| 🎯 **AUROC** | **0.908** |
| 📈 **AUPRC** | **0.927** |
| ✅ **Accuracy** | **79.7%** |

> AUROC > 0.90 is consistent with scBasset on ENCODE ATAC-seq data.

### Visualizations

<table>
  <tr>
    <td align="center"><b>ROC Curve</b></td>
    <td align="center"><b>Score Distribution</b></td>
  </tr>
  <tr>
    <td><img src="results/roc_curve.png" width="380"/></td>
    <td><img src="results/score_distribution.png" width="380"/></td>
  </tr>
  <tr>
    <td align="center"><b>Training Curves</b></td>
    <td align="center"><b>Threshold Analysis</b></td>
  </tr>
  <tr>
    <td><img src="results/training_curves.png" width="380"/></td>
    <td><img src="results/threshold_comparison.png" width="380"/></td>
  </tr>
</table>

### Threshold Sensitivity Analysis

| Threshold | Sensitivity | Specificity | Accuracy | F1 |
|-----------|------------|-------------|----------|----|
| `0.1` | 88.7% | 71.4% | 80.1% | 0.819 |
| `0.2` | 76.0% | 91.1% | 83.5% | 0.823 |
| `0.3` | 70.9% | 94.5% | 82.5% | 0.804 |
| `0.4` | 66.0% | 96.6% | 81.1% | 0.780 |
| `0.5` | 62.1% | 97.8% | 79.7% | 0.756 ← default |
| `0.6` | 58.7% | 98.5% | 78.3% | 0.733 |
| `0.7` | 55.7% | 98.8% | 76.9% | 0.710 |


---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Nikhila123456/ATAC-Seq-Peak-CNN.git
cd ATAC-Seq-Peak-CNN
conda env create -f environment.yml
conda activate atac-cnn
```

### Run Full Pipeline

```bash
# 1. Generate synthetic training data (10,000 sequences with real TF motifs)
python data/generate_synthetic.py

# 2. Train model (early stopping, ~30 epochs)
python train_model.py

# 3. Threshold analysis
python threshold_comparison.py

# 4. Predict on new sequences
python predict.py --sequence "ATGCATGCTGAGTCAATGCATGC"
python predict.py --fasta   my_peaks.fasta
```

---

## 📁 Repository Structure

```
ATAC-Seq-Peak-CNN/
│
├── 📂 src/
│   ├── model.py              # CNN architecture (start here)
│   ├── dataset.py            # PyTorch Dataset + DataLoader
│   ├── train.py              # Training loop + early stopping
│   ├── evaluate.py           # AUROC, ROC, confusion matrix
│   └── utils.py              # One-hot encoding, DNA utilities
│
├── 📂 data/
│   └── generate_synthetic.py # Synthetic ATAC-seq training data
│
├── 📂 results/               # Model checkpoint + evaluation plots
│
├── train_model.py            # ▶ Main entry point
├── predict.py                # Inference on new sequences
├── threshold_comparison.py   # Sensitivity/specificity analysis
├── environment.yml
└── requirements.txt
```

---

## 🔧 Extending to Real Data

```bash
# Download ENCODE ATAC-seq peaks (e.g. GM12878 immune cell line)
# https://www.encodeproject.org/

# Extract sequences (peak center ± 250 bp)
bedtools getfasta -fi hg38.fa -bed peaks.bed -fo positives.fasta

# Generate matched negatives (random accessible genome regions)
# Replace data/train.npz and retrain
# Expected AUROC on real data: 0.85–0.95
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥ 2.0 | Deep learning |
| NumPy | < 2.0 | Array ops |
| scikit-learn | ≥ 1.2 | AUROC, metrics |
| matplotlib | ≥ 3.6 | Plots |

---

## 📚 References

1. Yuan H & Kelley DR. **scBasset** *Nature Methods* 19, 1088 (2022).
   https://doi.org/10.1038/s41592-022-01562-8

2. Avsec Ž et al. **Enformer** *Nature Methods* 18, 1196 (2021).
   https://doi.org/10.1038/s41592-021-01252-x

3. Linder J et al. *Nature Genetics* (2025).
   https://doi.org/10.1038/s41588-024-02053-6

4. He K et al. **Deep Residual Learning** *CVPR* (2016).
   https://arxiv.org/abs/1512.03385

---

## 👩‍🔬 Author

**Nikhila T. Suresh, PhD**
Postdoctoral Research Fellow · Florida Atlantic University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-nikhilatssuresh-0077B5?logo=linkedin)](https://linkedin.com/in/nikhilatssuresh)
[![GitHub](https://img.shields.io/badge/GitHub-Nikhila123456-181717?logo=github)](https://github.com/Nikhila123456)

---

<div align="center">
<i>If you find this useful, please ⭐ the repo!</i>
</div>
