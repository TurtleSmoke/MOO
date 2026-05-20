# MOO: Multi-view Oriented Observations

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE.md)
[![arXiv](https://img.shields.io/badge/arXiv-2603.04314-b31b1b.svg)](https://arxiv.org/abs/2603.04314)

## Overview

MOO provides **128,000 images** of 1,000 synthetic cattle identities captured from 128 uniformly sampled viewpoints,
spanning 360° in azimuth and -25° to 90° in elevation.

## Download

**[Download MOO.zip](https://kalisteo.cea.fr/index.php/moo-dataset-en/)**

The archive contains:

```
MOO.zip
├── data.hdf5                                           # RGB images and depth maps packed
├── metadata.json                                       # Annotations and identity labels
├── LICENSE                                             # License file (CC BY 4.0)
└── splits/
    ├── train_{top,side,topside}.txt                    # Training splits
    └── test_{top,side,topside}_{query,gallery}.txt     # Testing splits
```

Once downloaded and unpacked, the dataset structure is as follows:

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
    year = {2026}
}

```
