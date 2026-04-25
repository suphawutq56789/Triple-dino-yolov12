# Triple-DINOv3-YOLOv12

**Multi-Stream Object Detection with DINOv3 Vision Foundation Model**

Department of Civil Engineering · King Mongkut's University of Technology Thonburi (KMUTT)

---

## Overview

This project extends YOLOv12 to accept **three simultaneous image streams** (9-channel input) combined with **DINOv3** — Meta's self-supervised Vision Transformer — for enhanced feature extraction. Designed for civil engineering applications where multiple camera angles or detail images improve detection accuracy.

```
┌─────────────────────────────────────────────────────────────┐
│                    Triple-DINOv3-YOLOv12                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: 3 Images (main + detail1 + detail2)                 │
│         ↓                                                   │
│  [TripleInputConv]  9ch → 64ch                              │
│         ↓                                                   │
│  YOLOv12 Backbone  +  DINOv3 (P3 / P4 / Dual)              │
│         ↓                                                   │
│  Detection Head → Output                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Contributions

| Feature | Description |
|---|---|
| **Triple Input** | 9-channel input from 3 camera views (main + 2 details) |
| **DINOv3 Integration** | Real DINOv3 ViT at P3, P4, or dual P3+P4 stages |
| **6 Integration Modes** | `initial` / `p3` / `p4` / `dual` / `p0p3` / `nodino` |
| **Multi-Scale DINOv3** | small / base / large / giant / satellite variants |
| **Civil Engineering Focus** | Crack detection, structural monitoring, infrastructure inspection |

---

## Integration Modes

```
                    9ch Input
                       ↓
              [TripleInputConv]
                       ↓
    ┌──────────────────┼──────────────────┐
    │                  │                  │
  initial             P3/P4/dual        nodino
    │                  │                  │
  DINOv3@P0       DINOv3@P3/P4/both    No DINOv3
    │                  │                  │
    └──────────────────┼──────────────────┘
                       ↓
                  Detection Head
```

| Mode | DINOv3 Position | VRAM | Best For |
|---|---|---|---|
| `initial` | Before backbone (P0) | High | Maximum semantic features at input |
| `p3` | After P3 stage | Medium | Mid-level feature enhancement |
| `p4` | After P4 stage | Medium | Abstract feature enhancement |
| `dual` | P3 + P4 both | High | Richest multi-scale DINOv3 coverage |
| `p0p3` | P0 + P3 both | High | Input + mid-level dual enhancement |
| `nodino` | None | Low | Baseline triple-input without DINOv3 |

---

## Installation

```bash
git clone https://github.com/suphawutq56789/Triple-dino-yolov12.git
cd Triple-dino-yolov12

conda create -n triple-dino python=3.11
conda activate triple-dino

pip install -r requirements.txt
pip install -e .
```

**HuggingFace token** (required for DINOv3 download):
```bash
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
# Get token from: https://huggingface.co/settings/tokens
```

---

## Dataset Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg          ← main image
│   │   ├── detail1/
│   │   │   └── img001.jpg      ← detail view 1
│   │   └── detail2/
│   │       └── img001.jpg      ← detail view 2
│   ├── val/  (same structure)
│   └── test/ (same structure)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

```yaml
# data.yaml
path: /path/to/dataset
train: images/train
val:   images/val
test:  images/test
nc: 1
names: ['your_class']
```

> If `detail1/` or `detail2/` are missing, the main image is used as fallback automatically.

---

## Training

### Recommended Command
```bash
# p3 + pretrained + DINOv3 large (best balance)
python train_triple_dinov3.py \
  --data /path/to/data.yaml \
  --integrate p3 \
  --dinov3-size large \
  --variant m \
  --batch 16 \
  --imgsz 640 \
  --workers 16 \
  --epochs 150 \
  --patience 30 \
  --pretrained yolov12m.pt
```

### All Integration Options
```bash
# p4 mode — DINOv3 at P4
python train_triple_dinov3.py --data data.yaml --integrate p4 --dinov3-size large --variant m --batch 16

# dual mode — DINOv3 at P3 + P4
python train_triple_dinov3.py --data data.yaml --integrate dual --dinov3-size large --variant m --batch 8

# initial mode — DINOv3 before backbone
python train_triple_dinov3.py --data data.yaml --integrate initial --dinov3-size large --variant m --batch 16 --pretrained yolov12m.pt

# nodino — triple input only (baseline)
python train_triple_dinov3.py --data data.yaml --integrate nodino --variant m --batch 16
```

### Key Arguments

| Argument | Options | Default | Description |
|---|---|---|---|
| `--integrate` | `initial` `p3` `p4` `dual` `p0p3` `nodino` | `initial` | DINOv3 integration mode |
| `--dinov3-size` | `small` `base` `large` `giant` `sat_large` `sat_giant` | `small` | DINOv3 model size |
| `--variant` | `n` `s` `m` `l` `x` | `s` | YOLOv12 model size |
| `--pretrained` | path to `.pt` | None | YOLOv12 pretrained weights |
| `--unfreeze-dinov3` | flag | False | Fine-tune DINOv3 (needs lower batch) |
| `--batch` | int | `8` | Batch size |
| `--imgsz` | int | `224` | Image size (640 recommended) |
| `--epochs` | int | `200` | Training epochs |
| `--patience` | int | `30` | Early stopping patience |

### DINOv3 Size Guide

| Size | Params | VRAM | Notes |
|---|---|---|---|
| `small` | 21M | ~2GB | Quick experiments |
| `base` | 86M | ~4GB | Balanced |
| `large` | 304M | ~8GB | **Recommended** |
| `giant` | 1.1B | ~16GB | Maximum accuracy |
| `sat_large` | 304M | ~8GB | Satellite/aerial images |
| `sat_giant` | 1.1B | ~16GB | Large-scale satellite |

---

## Architecture

### DINOv3FeatureEnhancer (New)

A new module that wraps DINOv3Backbone for use at any feature stage (P3/P4):

```python
# Used automatically when --integrate p3, p4, or dual
DINOv3FeatureEnhancer(
    input_channels=512,
    output_channels=512,
    model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
    freeze=True
)
```

### Model YAML Files

| File | Mode | Description |
|---|---|---|
| `yolov12_triple.yaml` | `nodino` | Triple input, no DINOv3 |
| `yolov12.yaml` | `initial` | Standard backbone (DINOv3 as preprocessor) |
| `yolov12_triple_dinov3_p3.yaml` | `p3` | Real DINOv3 at P3 |
| `yolov12_triple_dinov3_p4.yaml` | `p4` | Real DINOv3 at P4 |
| `yolov12_triple_dinov3_dual.yaml` | `dual` | Real DINOv3 at P3 + P4 |
| `yolov12_triple_dinov3_p0p3_adapter.yaml` | `p0p3` | DINOv3 at P0 + P3 |

---

## Changelog

- **2026/04/25** — Added `p4` and `dual` integration modes. Fixed `p3` to use real DINOv3 ViT (was conv attention before). Added `DINOv3FeatureEnhancer` module.
- **2025/10/02** — Added satellite DINOv3 variants (`sat_large`, `sat_giant`) trained on SAT-493M dataset.
- **2025/09/20** — Initial release: Triple input architecture + DINOv3 integration for civil engineering.

---

## Acknowledgements

- [YOLOv12](https://arxiv.org/abs/2502.12524) — Yunjie Tian, Qixiang Ye, David Doermann
- [DINOv3](https://github.com/facebookresearch/dinov3) — Meta AI Research
- [DINOV3-YOLOV12](https://github.com/Sompote/DINOV3-YOLOV12) — Sompote (architecture reference)
- [ultralytics](https://github.com/ultralytics/ultralytics) — Base framework

---

## Citation

```bibtex
@misc{triple_dinov3_yolov12,
  title   = {Triple-DINOv3-YOLOv12: Multi-Stream Object Detection with DINOv3 Vision Foundation Model},
  author  = {Research Group, Department of Civil Engineering, KMUTT},
  year    = {2025},
  url     = {https://github.com/suphawutq56789/Triple-dino-yolov12}
}

@article{tian2025yolov12,
  title   = {YOLOv12: Attention-Centric Real-Time Object Detectors},
  author  = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal = {arXiv preprint arXiv:2502.12524},
  year    = {2025}
}
```

---

*Department of Civil Engineering · KMUTT · 2025*
