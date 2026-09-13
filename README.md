<div align="center">

# Cross-modal learning for SAR target recognition using optical vision foundation models

### Code for EO-to-SAR prototype alignment

[![arXiv](https://img.shields.io/badge/arXiv-2609.07753-b31b1b.svg)](https://arxiv.org/abs/2609.07753)

**[Lucas Hirsch](https://luhirsch.github.io/) · [James R. Hopgood](https://www.research.ed.ac.uk/en/persons/james-hopgood/) · Javid Khan · [Yoann Altmann](https://researchportal.hw.ac.uk/en/persons/yoann-altmann/) · [Mike E. Davies](https://eng.ed.ac.uk/about/people/professor-michael-e-davies)**

📝 [Paper](https://arxiv.org/abs/2609.07753) · Accepted for presentation at SPIE Sensors + Imaging 2026

</div>

This repository contains the code accompanying the paper
*[Cross-modal learning for SAR target recognition using optical vision foundation models](https://arxiv.org/abs/2609.07753)*.

We investigate whether electro-optical (EO) vision foundation models (e.g. DINOv3) can
provide supervision for synthetic aperture radar (SAR) target recognition.
A frozen DINOv3 EO encoder is used to construct optical class prototypes
without requiring paired EO and SAR images. A SAR encoder is then trained to
classify SAR images while aligning its embeddings with the corresponding EO
class prototypes. At inference time, the model operates using SAR imagery
alone.

## Quickstart

1. Install PyTorch, torchvision, and the remaining dependencies
   ([Installation](#installation)).
2. Clone DINOv3 and download the ViT-S+/16 weights.
3. Copy `config-example.yaml` to `config.yaml` and update the paths
   ([Configuration](#configuration)).
4. Arrange the EO and SAR images in PyTorch `ImageFolder` format
   ([Data setup](#data-setup)).
5. Run one of the experiment scripts ([Usage](#usage)).

For example:

```bash
cp config-example.yaml config.yaml
python src/train_eo_prototype_alignment.py
```

## 📋 Method Overview

The repository compares five ways of using DINOv3 for SAR classification:

| Method | Training data | Adaptation | Objective |
|---|---|---|---|
| Frozen DINOv3 | SAR | Linear head only | Cross-entropy |
| SAR-only fine-tuning | SAR | LoRA | Cross-entropy |
| MMD alignment | Unpaired SAR and EO (labeled) | LoRA | Cross-entropy + global MMD |
| EO prototype alignment **(ours)** | Unpaired SAR and EO (labeled) | LoRA | Cross-entropy + class-prototype alignment |
| Synthetic prototype alignment | SAR | LoRA | Cross-entropy + synthetic class-prototype alignment |

The MMD experiment samples SAR and EO images independently and therefore does
not use image pairs. Prototype alignment first averages the EO features within
each class, then uses the resulting fixed class prototypes as targets for the
SAR embeddings.

Synthetic prototype alignment replaces the EO prototypes with fixed synthetic
class prototypes. This provides a control for whether improvements come from
the geometry of the EO prototypes rather than prototype regularization alone.

All methods use SAR images alone for evaluation.

## 📦 Installation

Clone the [official DINOv3 repository](https://github.com/facebookresearch/dinov3)
and request/download the web pretrained **ViT-S+/16** weights.

Install a recent version of PyTorch and torchvision suitable for your CUDA
version by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/).
Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, install them directly:

```bash
pip install numpy pyyaml scikit-learn peft
```

GPU training is strongly recommended for the LoRA experiments. The scripts use
CUDA when it is available and otherwise fall back to CPU. If several GPUs are
visible, select the required CUDA device in the training script.

## 📝 Configuration

All machine-specific paths are configured in `config.yaml`. Start by copying
the provided example:

```bash
cp config-example.yaml config.yaml
```

Then update the paths for your system:

```yaml
paths:
  dino_repo: "/path/to/dinov3"
  dino_weights: "/path/to/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"

  train_sar: "./data/train/SAR_Train"
  train_eo: "./data/train/EO_Train"
  test_sar: "./data/test/IID"

  output_dir: "./outputs"
```

Paths may be absolute or relative. Relative paths are resolved from the
repository root, regardless of where a script is launched.

Your local `config.yaml` is ignored by Git, so each user can keep their own
paths without committing them.

## 📂 Data Setup

The experiments use the UNICORNv2 EO/SAR vehicle dataset, which contains paired
EO and SAR images of 10 vehicle classes. This dataset is used for the MAVIC-C
competition, part of a CVPR workshop, and can be downloaded from the official
[competition website](https://www.codabench.org/forums/12351/).

The expected structure is:

```text
data/
├── train/
│   ├── SAR_Train/
│   │   ├── <class_name>/
│   │   └── ...
│   └── EO_Train/
│       ├── <class_name>/
│       └── ...
└── test/
    └── IID/
        ├── <class_name>/
        └── ...
```

The folder names define the class labels through
[`torchvision.datasets.ImageFolder`](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html).
The EO and SAR folders must use the same class names, but individual EO and SAR
images do not need to be paired.

You may also keep the dataset elsewhere and point `config.yaml` to its
location.

## 🖥️ Usage

Run each command from the repository root.

### Frozen DINOv3 baseline

Extract frozen SAR features once and train weighted linear classifiers:

```bash
python src/train_frozen_dino.py
```

### SAR-only LoRA fine-tuning

Fine-tune the SAR encoder using labelled SAR images and cross-entropy only:

```bash
python src/train_sar_only.py
```

### Unpaired MMD alignment

Fine-tune the SAR encoder while matching the global SAR feature distribution to
independently sampled EO features:

```bash
python src/train_mmd_alignment.py
```

### EO prototype alignment

Construct fixed EO class prototypes and align each SAR embedding with the
prototype for its class:

```bash
python src/train_eo_prototype_alignment.py
```

### Synthetic prototype alignment

Replace the EO prototypes with fixed synthetic class prototypes and train using
the same classification and alignment objectives:

```bash
python src/train_synthetic_prototype_alignment.py
```

### Outputs

Each method writes to its own directory under the configured `output_dir`:

```text
outputs/<method>/
├── results.csv
├── summary.json
└── predictions_seed_<seed>.csv
```

`results.csv` contains the best-epoch metrics for each seed,
`summary.json` contains the aggregate mean and standard deviation, and each
prediction file contains the true and predicted class for every test image.

The scripts do not save model checkpoints.

## 📖 Citation

To cite the paper, use:

```bibtex
@misc{hirsch2026crossmodal,
  title         = {Cross-modal learning for SAR target recognition using optical vision foundation models},
  author        = {Lucas Hirsch and James R. Hopgood and Javid Khan and Yoann Altmann and Mike E. Davies},
  year          = {2026},
  eprint        = {2609.07753},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.07753}
}
```
