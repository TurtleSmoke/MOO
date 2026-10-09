# MOO: Multi-view Oriented Observations

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE.md)
[![arXiv](https://img.shields.io/badge/arXiv-2603.04314-b31b1b.svg)](https://arxiv.org/abs/2603.04314)
[![CVPRW 2026](https://img.shields.io/badge/CVPRW%202026-CV4Animals-1f6feb.svg)](https://www.cv4animals.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CEAai%2FMOO-yellow.svg)](https://huggingface.co/datasets/CEAai/MOO)

Presented at the CVPR 2026 Workshop on Computer Vision for Animal Behavior Tracking and Modeling (CV4Animals), non-archival.

## Overview

MOO provides **128,000 images** of 1,000 synthetic cattle identities captured from 128 uniformly sampled viewpoints,
spanning 360° in azimuth and -25° to 90° in elevation.

## Download

**Option 1: [Download MOO.zip](https://kalisteo.cea.fr/index.php/moo-dataset-en/) from Kalisteo**

The archive contains the packed data only:

```
MOO.zip
├── data.hdf5           # RGB images and depth maps packed
└── metadata.json       # Annotations and identity labels
```

Unzip it at the root of this repository, then generate the split files with `make_splits.py` (see [Splits](#splits)):

```
MOO/                                                    # this repository
├── data.hdf5                                           # from MOO.zip
├── metadata.json                                       # from MOO.zip
└── splits/                                             # from python make_splits.py
    ├── train_{top,side,topside}.txt                    # Training splits
    └── test_{top,side,topside}_{query,gallery}.txt     # Testing splits
```

**Option 2: [Hugging Face](https://huggingface.co/datasets/CEAai/MOO)**, with the evaluation protocols as configs:

```python
from datasets import load_dataset

train = load_dataset("CEAai/MOO", "topside", split="train")
query = load_dataset("CEAai/MOO", "top", split="query")  # Top -> Side
gallery = load_dataset("CEAai/MOO", "side", split="gallery")
```

After unpacking `data.hdf5` with `pack.py` (see [Usage](#usage)), the dataset structure is as follows:

```
data/
├── cow0/
│   ├── cow0_0.png
│   ├── cow0_0.npy
│   ├── cow0_0_metadata.json
│   └── ...
└── ...  (1,000 identities)
```

Files are named `cow{id}_{view_id}`, where `view_id` indexes the viewpoint in the 128-point grid.
The background in the depth map is set to $1e^{10}$ by blender, which can also be used to mask out the background.

## Splits

Identities are split 500/500 (even IDs train, odd IDs test). Views `0-63` are side views (elevation < 30°) and
views `64-127` are top views (elevation > 30°). [`make_splits.py`](make_splits.py) defines the protocol, including the
fixed query/gallery assignment of each test identity, and writes the split files used by [`MOO.py`](MOO.py):

```bash
python make_splits.py  # splits/train_{top,side,topside}.txt, splits/test_{top,side,topside}_{query,gallery}.txt
```

## Setup

For running the unpacking script, [uv](https://docs.astral.sh/uv/) is recommended:

```bash
# Install uv, if necessary
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
source .venv/bin/activate
# or using `uv run` directly without activating the environment
```

pip is also supported using the generated requirements files:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt # with dev dependencies
```

## Usage

Unzip, then unpack the `data.hdf5` and `metadata.json`:

```bash
unzip MOO.zip
python pack.py unpack --src . --out .
```

Use `--png`, `--depth`, `--json` to extract selectively. Pack back with:

```bash
python pack.py pack --src . --out .
```

Run `python pack.py --help` for the full list of options.

## FastReID Integration

A dataloader compatible with [FastReID](https://github.com/jdai-cv/fast-Reid) is provided in [`MOO.py`](MOO.py). Copy it to `fastreid/data/datasets/MOO.py` in your FastReID installation, update the `__init__.py` to include relevant train/test splits, and use the dataset name `MOO` in your FastReID config.

## Citation

```bibtex
@article{grolleau2026moo,
    title = {MOO: A Multi-view Oriented Observations Dataset for Viewpoint Analysis in Cattle Re-Identification},
    author = {Grolleau, William and Chaouch, Achraf and Sabourin, Astrid and Lapouge, Guillaume and Achard, Catherine},
    journal = {arXiv preprint arXiv:2603.04314},
    note = {Presented at the CVPR 2026 Workshop on Computer Vision for Animal Behavior Tracking and Modeling (CV4Animals), non-archival},
    year = {2026},
    doi = {10.48550/arXiv.2603.04314}
}

```
