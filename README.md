# Geometry-aware Pipe-Plate Defect Segmentation with Dynamic Attention and Star-Topology Convolution

<!-- [![Paper](https://img.shields.io/badge/Paper-IEEE%20TII-blue)](https://...) -->
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Framework](https://img.shields.io/badge/Framework-YOLOv13-orange)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red)](https://pytorch.org/)

> **Official implementation of the paper submitted to *IEEE Transactions on Industrial Informatics*.**

## 📋 Overview

We present a **geometry-aware defect segmentation framework** for pipe-plate weld inspection in heat exchanger manufacturing. The proposed method addresses three core challenges in industrial defect detection: (i) irregular geometric structures of weld seams, (ii) topological complexity under non-stationary defect distributions, and (iii) poor cross-scale geometric consistency in existing segmentation pipelines.

Built upon YOLOv13, our framework integrates three novel components:

- **MLA-DSAM** — Multi-Level Attention with Depthwise Separable Attention Module, ensuring cross-scale geometric consistency via adaptive feature metrics.
- **StellarConv** — A pentagram-shaped topological convolution operator that non-linearly extends the high-dimensional kernel space, reducing parameters by **87.9%** while capturing high-order directional correlations.
- **MASegment Head** — A Geometry-aware Segmentation Head with cross-level semantic reorganization and boundary-driven mask modulation.

We also release the **first industrial-grade pipe-plate defect dataset** (18,000 HD images, 4 defect categories) to facilitate future research.

## 🔥 Key Contributions

1. **MLA-DSAM Module**: Reconstructs a semantic weight mapping system based on an Adaptive Feature Metric. Multi-scale projection operators implicitly enforce local feature stability, ensuring stable fusion of cross-scale features while preserving local geometric consistency.

2. **StellarConv Operator**: Introduces a pentagram-shaped topological structure prior as a directional constraint for sparse reconstruction on irregular geometries. For the first time, non-linearly extends the high-dimensional convolution kernel space, overcoming the expression bottleneck of traditional convolutions within local receptive fields.

3. **Geometry-aware Segmentation Head (MASegment)**: Features cross-level semantic reorganization and boundary-driven mask modulation, with a differentiable boundary-constrained operator embedded in the gradient feedback pathway. Significantly reduces parameter overhead while improving geometric consistency and discriminative margin.

4. **Pipe-Plate Defect Dataset**: First publicly available industrial-grade benchmark with 18,000 high-definition images from nuclear power plant manufacturing, covering 4 defect categories under diverse working conditions.

## 🏗️ Architecture

```
Input Image (640×640)
       │
       ▼
┌─────────────────────────────────┐
│         YOLOv13 Backbone        │
│  (DSConv_YOLO13, A2C2f, DSC3k2) │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  MLA-DSAM (Attention Module)    │
│  ┌───────────────────────────┐  │
│  │ Adaptive Feature Metric   │  │
│  │ Multi-Scale Projection    │  │
│  │ Cross-Scale Fusion        │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  StellarConv (Topology Conv)    │
│  ┌───────────────────────────┐  │
│  │ Pentagram Topological     │  │
│  │ Prior + Sparse            │  │
│  │ Reconstruction            │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  MASegment Head                 │
│  ┌───────────────────────────┐  │
│  │ DBB Division Processing   │  │
│  │ Boundary-Constrained Op   │  │
│  │ Cross-Level Reorganize    │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
       │
       ▼
  Detection + Segmentation Masks
```

## 📊 Performance

### Main Results on Pipe-Plate Defect Dataset

| Model | P<sub>mask</sub> (%) | P<sub>box</sub> (%) | R<sub>box</sub> (%) | mAP<sup>50</sup><sub>box</sub> (%) | mAP<sup>50</sup><sub>mask</sub> (%) | Params (M) | FLOPs (G) |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mask R-CNN | 88.3 | 86.6 | 81.3 | 76.5 | 66.6 | 44.6 | 193 |
| Cascade-Mask R-CNN | 89.6 | 87.6 | 86.4 | 75.4 | 67.1 | 77.3 | 1716 |
| YOLOv8 | 93.9 | 89.4 | 88.0 | 87.0 | 67.6 | 11.7 | 42.7 |
| YOLOv11 | 93.4 | 93.4 | 90.0 | 87.1 | 67.2 | 10.0 | 35.6 |
| YOLOv12 | 95.0 | 94.7 | 89.0 | 87.0 | 66.5 | 9.75 | 33.6 |
| YOLOv13 (baseline) | 95.8 | 88.9 | 80.9 | 87.2 | 66.0 | 9.68 | 34.4 |
| **Ours** | **96.8** | **97.4** | **91.0** | **87.6** | **68.0** | **9.81** | **41.1** |

### Ablation Study

| Exp | StellarConv | MLA-DSAM | MASegment | P<sub>box</sub> (%) | P<sub>mask</sub> (%) | mAP<sup>50</sup><sub>box</sub> (%) | FPS |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | | | | 88.9 | 95.8 | 87.2 | 63.6 |
| 2 | ✓ | | | 91.3 | 95.2 | 87.3 | 64.5 |
| 3 | | ✓ | | 91.0 | 95.9 | 87.4 | 66.2 |
| 4 | | | ✓ | 92.1 | 96.5 | 87.3 | 69.7 |
| 5 | ✓ | ✓ | | 93.4 | 96.1 | 87.5 | 70.3 |
| 6 | ✓ | | ✓ | 94.6 | 96.6 | 87.4 | 70.9 |
| 7 | | ✓ | ✓ | 95.8 | 96.6 | 87.5 | 71.3 |
| **8** | **✓** | **✓** | **✓** | **96.8** | **96.8** | **87.6** | **73.5** |

### Generalization on NEU Surface Defect Dataset

| Model | P<sub>mask</sub> (%) | P<sub>box</sub> (%) | mAP<sup>50</sup><sub>box</sub> (%) | mAP<sup>50</sup><sub>mask</sub> (%) |
|-------|:---:|:---:|:---:|:---:|
| Mask R-CNN | 89.3 | 85.6 | 79.6 | 75.3 |
| YOLOv8 | 81.9 | 83.3 | 82.1 | 78.6 |
| YOLOv12 | 94.7 | 95.2 | 81.5 | 77.8 |
| **Ours** | **87.3** | **87.3** | **82.9** | **78.8** |

## 📦 Dataset

We publicly release the **Pipe-Plate Weld Defect Dataset**, the first industrial-grade benchmark specifically designed for pipe-plate weld inspection.

- **Source**: On-site collection from nuclear power plant manufacturing
- **Size**: 18,000 high-definition images (augmented from 1,000 originals)
- **Split**: Training / Validation / Testing = 8:1:1
- **Categories**:
  - **Notfull** — Incomplete weld filling
  - **Hole** — Surface porosity defects
  - **External defects** — Surface-level structural anomalies
  - **Internal defects** — Sub-surface inclusion-type defects

> 🔗 **Download**: [Baidu Netdisk](https://pan.baidu.com/s/1jM-cqot-C7zhcHpf9iwI2w) &nbsp;|&nbsp; Code: `8bma`

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/bianduche/pipe-plate-defect-segmentation-method.git
cd pipe-plate-defect-segmentation-method

# Install dependencies
pip install ultralytics
pip install torch>=2.4.0 torchvision
```

### Requirements

- Python 3.11+
- PyTorch 2.4.0+
- CUDA 12.1 (recommended)
- ultralytics (YOLO framework)
- Ubuntu 22.04 (training) / Windows (inference)

## 🚀 Quick Start

### Training

```bash
python train.py
```

Key training configuration (see `train.py` for details):

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Base LR | 0.01 |
| Input Size | 640×640 |
| Batch Size | 80 |
| Epochs | 100 |
| Mosaic | 1.0 |
| Weight Decay | 0.0005 |

### Inference

```bash
python detect.py
```

The model weights (`best.pt`) are loaded from `runs/train/exp/weights/`.

### Model Configuration

The model architecture is defined in `yolo13.yaml`, with custom modules:

| Component | File | Description |
|-----------|------|-------------|
| `StellarConv` | `StellarConv.py` | Pentagram-topology convolution operator |
| `MLA-DSAM` | `MLA_DSAM` | Multi-level attention with depthwise separable attention |
| `MASegment Head` | `MASegment Head` | Geometry-aware segmentation head |

## 📂 Repository Structure

```
pipe-plate-defect-segmentation-method/
├── yolo13.yaml              # Model architecture configuration
├── train.py                 # Training script
├── detect.py                # Inference / evaluation script
├── StellarConv.py           # StellarConv operator implementation
├── MLA_DSAM                 # MLA-DSAM attention module
├── MASegment Head           # Geometry-aware segmentation head
└── ultralytics/             # Modified ultralytics framework
```

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{anon2025geometry,
  title={Geometry-aware Pipe-Plate Defect Segmentation with Dynamic Attention and Star-Topology Convolution},
  author={Anonymous Authors},
  journal={IEEE Transactions on Industrial Informatics},
  year={2025},
  note={Under Review}
}
```

## 📄 License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## 🙏 Acknowledgements

This work was supported in part by research grants. We thank the anonymous reviewers for their constructive feedback.

---

> **For questions, please open an issue or contact the corresponding author.**
