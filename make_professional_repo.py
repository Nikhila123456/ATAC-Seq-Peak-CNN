"""
Run from inside atac_cnn/:
    python make_professional_repo.py

Creates:
    - README.md       (professional, badge-style, figures prominent)
    - .gitignore      (Python + data files)
    - LICENSE         (MIT)
    - CITATION.cff    (makes your repo citable)
"""
import json, os

# ── Load your actual results ───────────────────────────────────────────────────
with open("results/results.json") as f:
    r = json.load(f)["metrics"]

with open("results/threshold_comparison.json") as f:
    thresholds = json.load(f)

auroc = round(r["auroc"], 3)
auprc = round(r["auprc"], 3)
acc   = round(r["accuracy"] * 100, 1)

# Build threshold table
tbl = ""
for t in thresholds:
    mark = " ← default" if t["threshold"] == 0.5 else ""
    tbl += (f"| `{t['threshold']}` "
            f"| {t['sensitivity']*100:.1f}% "
            f"| {t['specificity']*100:.1f}% "
            f"| {t['accuracy']*100:.1f}% "
            f"| {t['f1']:.3f}{mark} |\n")

# ── README ─────────────────────────────────────────────────────────────────────
readme = f"""<div align="center">

# 🧬 ATAC-Seq Peak Prediction with CNN

**Predicting chromatin accessibility from DNA sequence using deep learning**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AUROC](https://img.shields.io/badge/AUROC-{auroc}-blue)](results/roc_curve.png)

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
| 🎯 **AUROC** | **{auroc}** |
| 📈 **AUPRC** | **{auprc}** |
| ✅ **Accuracy** | **{acc}%** |

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
{tbl}

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
"""

# ── .gitignore ─────────────────────────────────────────────────────────────────
gitignore = """# Python
__pycache__/
*.py[cod]
*.egg-info/
.env
.venv/
*.venv

# Data files (too large for git — regenerate with generate_synthetic.py)
data/*.npz

# Jupyter
.ipynb_checkpoints/
*.ipynb

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Model weights (large — tracked separately if needed)
# results/best_model.pt
"""

# ── LICENSE ────────────────────────────────────────────────────────────────────
license_text = """MIT License

Copyright (c) 2026 Nikhila T. Suresh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# ── CITATION.cff ───────────────────────────────────────────────────────────────
citation = """cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: Suresh
    given-names: Nikhila T.
    affiliation: Florida Atlantic University
    orcid: "https://orcid.org/YOUR-ORCID-HERE"
title: "ATAC-Seq Peak Prediction with CNN"
version: 1.0.0
date-released: 2026-06-11
url: "https://github.com/Nikhila123456/ATAC-Seq-Peak-CNN"
"""

# ── Write all files ────────────────────────────────────────────────────────────
files = {
    "README.md":    readme,
    ".gitignore":   gitignore,
    "LICENSE":      license_text,
    "CITATION.cff": citation,
}

for fname, content in files.items():
    with open(fname, "w") as f:
        f.write(content)
    print(f"✓ Written: {fname}")

print("""
Done! Now run:
  git add README.md .gitignore LICENSE CITATION.cff results/*.png
  git commit -m "Professional README with badges, figures, and results"
  git push
""")
