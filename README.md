# Geometry-aware Pipe-Plate Defect Segmentation with Dynamic Attention and Star-Topology Convolution

![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)
![Framework](https://img.shields.io/badge/Framework-YOLOv13-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red)

> **Official implementation of the paper published in *IEEE Open Journal of the Computer Society*.**

## Overview

We present a **geometry-aware defect segmentation framework** for pipe-plate weld inspection in heat exchanger manufacturing. The proposed method addresses three core challenges in industrial defect detection: (i) irregular geometric structures of weld seams, (ii) topological complexity under non-stationary defect distributions, and (iii) poor cross-scale geometric consistency in existing segmentation pipelines.

Built upon YOLOv13, our framework integrates three novel components:

- **MLA-DSAM** — Multi-Level Attention with Depthwise Separable Attention Module, ensuring cross-scale geometric consistency via adaptive feature metrics.
- **StellarConv** — A pentagram-shaped topological convolution operator that non-linearly extends the high-dimensional kernel space, reducing operator-level parameters by **87.9%** while capturing high-order directional correlations, with the total network at only **9.81M parameters**.
- **MASegment Head** — A Geometry-aware Segmentation Head with cross-level semantic reorganization and boundary-driven mask modulation.

We also release the **first industrial-grade pipe-plate defect dataset** (18,000 HD images, 4 defect categories) to facilitate future research.

Experimental results demonstrate state-of-the-art performance with **96.8% mask precision**, **97.4% box precision**, and **87.6% mAP<sub>50</sub>**, outperforming existing methods while maintaining real-time inference at **73.5 FPS** with only 9.81M parameters.

## Key Contributions

1. **MLA-DSAM Module**: Reconstructs a semantic weight mapping system based on an Adaptive Feature Metric. Multi-scale projection operators implicitly enforce local feature stability, ensuring stable fusion of cross-scale features while preserving local geometric consistency.

2. **StellarConv Operator**: Introduces a pentagram-shaped topological structure prior as a directional constraint for sparse reconstruction on irregular geometries. For the first time, non-linearly extends the high-dimensional convolution kernel space, overcoming the expression bottleneck of traditional convolutions within local receptive fields.

3. **Geometry-aware Segmentation Head (MASegment)**: Features cross-level semantic reorganization and boundary-driven mask modulation, with a differentiable boundary-constrained operator embedded in the gradient feedback pathway. Significantly reduces parameter overhead while improving geometric consistency and discriminative margin.

4. **Pipe-Plate Defect Dataset**: First publicly available industrial-grade benchmark with 18,000 high-definition images from nuclear power plant manufacturing, covering 4 defect categories under diverse working conditions.

## Architecture

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
│  │ Prior + Sparse Sampling  │  │
│  │ Directional Constraints   │  │
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

## Performance

### Main Results on Pipe-Plate Defect Dataset

*Table 7 in paper. Column order: P<sub>mask</sub>, R<sub>box</sub>, P<sub>box</sub>, R<sub>mask</sub>, mAP<sub>50</sub><sup>box</sup>, mAP<sub>50</sub><sup>mask</sup>.*

| Model | P<sub>mask</sub> (%) | R<sub>box</sub> (%) | P<sub>box</sub> (%) | R<sub>mask</sub> (%) | mAP<sub>50</sub><sup>box</sup> (%) | mAP<sub>50</sub><sup>mask</sup> (%) | Params (M) | FLOPs (G) |
|:-------|---------------------:|---------------------:|---------------------:|---------------------:|------------------------:|------------------------:|------------:|----------:|
| Mask R-CNN | 88.3 | 81.3 | 86.6 | 68.5 | 76.5 | 66.6 | 44.6 | 193 |
| Cascade-Mask R-CNN | 89.6 | 86.4 | 87.6 | 71.3 | 75.4 | 67.1 | 77.3 | 1716 |
| YOLOv8 | 93.9 | 88.0 | 89.4 | 77.0 | 87.0 | 67.6 | 11.7 | 42.7 |
| YOLOv11 | 93.4 | 90.0 | 93.4 | 77.0 | 87.1 | 67.2 | 10.0 | 35.6 |
| YOLOv12 | 95.0 | 89.0 | 94.7 | 77.0 | 87.0 | 66.5 | 9.75 | 33.6 |
| YOLOv13 (baseline) | 88.9 | 80.9 | 95.8 | 78.0 | 87.2 | 66.0 | 9.68 | 34.4 |
| SAM | 80.1 | 83.1 | 79.6 | 70.3 | 71.3 | 58.6 | 141.3 | 65 |
| DeepLabV3+ | 82.6 | 81.6 | 80.1 | 72.6 | 73.6 | 60.1 | 303.0 | 51.4 |
| **Ours** | **96.8** | **91.0** | **97.4** | **79.0** | **87.6** | **68.0** | **9.81** | **41.1** |

### Ablation Study

*Table 6 in paper. SC=StellarConv, MLA=MLA-DSAM, MA=MASegment. Metrics: P<sub>box</sub>, P<sub>mask</sub>, R<sub>box</sub>, R<sub>mask</sub>, mAP<sub>50</sub><sup>box</sup>, mAP<sub>50</sub><sup>mask</sup>.*

| Exp. | SC | MLA-DSAM | MASegment | P<sub>box</sub> (%) | P<sub>mask</sub> (%) | R<sub>box</sub> (%) | R<sub>mask</sub> (%) | mAP<sub>50</sub><sup>box</sup> (%) | mAP<sub>50</sub><sup>mask</sup> (%) | FPS | Params (M) | FLOPs (G) |
|:-----:|:---:|:---------:|:-----------:|---------------------:|---------------------:|---------------------:|---------------------:|------------------------:|------------------------:|-----:|------------:|----------:|
| 1 | | | | 88.9 | 95.8 | 80.9 | 78.0 | 87.2 | 66.0 | 63.6 | 9.68 | 34.4 |
| 2 | ✓ | | | 91.3 | 95.2 | 89.6 | 78.2 | 87.3 | 66.3 | 64.5 | 9.65 | 32.1 |
| 3 | | ✓ | | 91.0 | 95.9 | 89.2 | 78.1 | 87.4 | 66.8 | 66.2 | 9.90 | 43.2 |
| 4 | | | ✓ | 92.1 | 96.5 | 89.4 | 78.0 | 87.3 | 67.1 | 69.7 | 9.69 | 34.3 |
| 5 | ✓ | ✓ | | 93.4 | 96.1 | 90.0 | 78.2 | 87.5 | 67.3 | 70.3 | 9.83 | 41.3 |
| 6 | ✓ | | ✓ | 94.6 | 96.6 | 90.9 | 78.4 | 87.4 | 67.6 | 70.9 | 9.90 | 32.2 |
| 7 | | ✓ | ✓ | 95.8 | 96.6 | 90.8 | 78.6 | 87.5 | 67.8 | 71.3 | 9.91 | 43.1 |
| 8 | ✓ | ✓ | ✓ | **97.4** | **96.8** | **91.0** | **79.0** | **87.6** | **68.0** | **73.5** | **9.81** | **41.1** |

### MLA-DSAM vs. Other Attention Mechanisms (YOLOv13, mean ± std over 5 runs)

*Table 5 in paper. Values reported as mean ± standard deviation over 5 independent runs with different random seeds. Paired t-test confirms statistical significance.*

| Method | P<sub>box</sub> (%) | P<sub>mask</sub> (%) | R<sub>box</sub> (%) | R<sub>mask</sub> (%) | mAP<sub>50</sub><sup>box</sup> (%) |
|--------|---------------------:|---------------------:|---------------------:|---------------------:|------------------------:|
| SE | 77.6 ± 0.3 | 71.8 ± 0.4 | 74.5 ± 0.3 | 68.2 ± 0.3 | 75.8 ± 0.2 |
| CBAM | 81.5 ± 0.2 | 75.7 ± 0.3 | 78.6 ± 0.2 | 71.7 ± 0.3 | 79.8 ± 0.2 |
| ECA | 75.9 ± 0.3 | 70.1 ± 0.3 | 72.8 ± 0.3 | 66.0 ± 0.4 | 74.2 ± 0.2 |
| Self-Attn | 79.7 ± 0.4 | 73.9 ± 0.3 | 76.9 ± 0.3 | 70.1 ± 0.3 | 78.3 ± 0.3 |
| Hyper-ACE | 88.9 ± 0.2 | 95.8 ± 0.1 | 80.9 ± 0.3 | 78.0 ± 0.2 | 87.1 ± 0.2 |
| **MLA-DSAM** | **91.0 ± 0.1** | **95.9 ± 0.1** | **89.2 ± 0.2** | **78.1 ± 0.2** | **87.5 ± 0.1** |

*Statistical significance (paired t-test, 5 runs): mAP<sub>50</sub><sup>box</sup> improvement over Hyper-ACE: p < 0.05; P<sub>box</sub> gain (+2.1%): p < 0.01; R<sub>box</sub> gain (+8.3%): p < 0.001.*

### StellarConv vs. Standard Convolution

*Table 1 in paper.*

| Metric | StellarConv | Standard Conv | Reduction |
|--------|-------------:|--------------:|----------:|
| 1st Layer Params | 104 | 680 | −84.7% |
| 2nd Layer Params | 608 | 5216 | −88.3% |
| Total Params | 712 | 5896 | −87.9% |
| 1st Layer GFLOPs | 0.014 | 0.123 | −88.6% |
| 2nd Layer GFLOPs | 0.028 | 0.248 | −88.7% |
| Total GFLOPs | 0.041 | 0.372 | −88.9% |
| Input Size | 581×653 | 581×653 | Same |
| Output Size | 146×164 | 146×164 | Same |

### Generalization on NEU Surface Defect Dataset

*Table 8 in paper.*

| Model | P<sub>mask</sub> (%) | R<sub>box</sub> (%) | P<sub>box</sub> (%) | R<sub>mask</sub> (%) | mAP<sub>50</sub><sup>box</sup> (%) | mAP<sub>50</sub><sup>mask</sup> (%) |
|:-------|---------------------:|---------------------:|---------------------:|---------------------:|------------------------:|------------------------:|
| Mask R-CNN | 89.3 | 86.7 | 85.6 | 81.3 | 79.6 | 75.3 |
| Cascade-Mask R-CNN | 82.6 | 84.6 | 86.6 | 79.6 | 71.6 | 72.1 |
| YOLOv8 | 81.9 | 94.0 | 83.3 | 91.0 | 82.1 | 78.6 |
| YOLOv11 | 82.6 | 94.0 | 82.6 | 91.0 | 81.6 | 77.2 |
| YOLOv12 | 94.7 | 89.0 | 95.2 | 77.0 | 81.5 | 77.8 |
| YOLOv13 | 85.2 | 94.0 | 84.9 | 91.0 | 75.9 | 72.8 |
| SAM | 71.3 | 73.5 | 70.6 | 69.5 | 70.6 | 68.3 |
| DeepLabV3+ | 78.6 | 79.6 | 74.5 | 76.5 | 73.6 | 70.1 |
| **Ours** | **87.3** | **94.0** | **87.3** | **91.0** | **82.9** | **78.8** |

## Dataset

We publicly release the **Pipe-Plate Weld Defect Dataset**, the first industrial-grade benchmark specifically designed for pipe-plate weld inspection.

- **Source**: On-site collection from nuclear power plant manufacturing
- **Size**: 18,000 high-definition images (augmented from 1,000 originals)
- **Split**: Training / Validation / Testing = 8:1:1
- **Categories**:
  - **Not full** — Incomplete weld filling
  - **Hole** — Surface porosity defects
  - **External defects** — Surface-level structural anomalies
  - **Internal defects** — Sub-surface inclusion-type defects

> **Download**: The dataset is publicly available at this repository: [https://github.com/bianduche/pipe-plate-defect-segmentation-method](https://github.com/bianduche/pipe-plate-defect-segmentation-method)

## Installation

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

## Quick Start

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
| Weight Decay | 5×10<sup>-4</sup> |

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

## Repository Structure

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

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{anonymous2026geometry,
  title={Geometry-aware Pipe-Plate Defect Segmentation with Dynamic Attention and Star-Topology Convolution},
  author={Anonymous Authors},
  journal={IEEE Open Journal of the Computer Society},
  year={2026},
  doi={10.1109/XXXX.2026.XXXXXXX}
}
```

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgements

This work was supported in part by research grants. We thank the anonymous reviewers for their constructive feedback.

---

> **For questions, please open an issue or contact the corresponding author.**
