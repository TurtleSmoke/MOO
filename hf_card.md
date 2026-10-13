---
license: cc-by-nc-4.0
pretty_name: "MOO: Multi-view Oriented Observations"
task_categories:
  - image-feature-extraction
tags:
  - re-identification
  - cattle
  - animals
  - synthetic
  - multi-view
  - depth
size_categories:
  - 100K<n<1M
configs:
  - config_name: topside
    default: true
    data_files:
      - split: train
        path: [data/train_top-*, data/train_side-*]
      - split: query
        path: [data/test_top_query-*, data/test_side_query-*]
      - split: gallery
        path: [data/test_top_gallery-*, data/test_side_gallery-*]
  - config_name: top
    data_files:
      - split: train
        path: data/train_top-*
      - split: query
        path: data/test_top_query-*
      - split: gallery
        path: data/test_top_gallery-*
  - config_name: side
    data_files:
      - split: train
        path: data/train_side-*
      - split: query
        path: data/test_side_query-*
      - split: gallery
        path: data/test_side_gallery-*
---

# MOO: Multi-view Oriented Observations

[Paper (arXiv:2603.04314)](https://arxiv.org/abs/2603.04314) · [Code](https://github.com/TurtleSmoke/MOO) · [Project page](https://turtlesmoke.dev/MOO/)

Presented at the CVPR 2026 Workshop on Computer Vision for Animal Behavior Tracking and Modeling (CV4Animals), non-archival.

MOO provides **128,000 images** of 1,000 synthetic cattle identities rendered from 128 calibrated viewpoints,
spanning 360° in azimuth and -25° to 90° in elevation, with depth maps and camera parameters for every image.

## Protocol

Identities are split 500/500: even IDs for training, odd IDs for testing.
Views `0-63` are **side** views (elevation < 30°), views `64-127` are **top** views (elevation > 30°).
For each test identity, half of its top views and half of its side views are queries, the rest are gallery.
The split definition lives in [`make_splits.py`](https://github.com/TurtleSmoke/MOO/blob/main/make_splits.py).

Each config (`top`, `side`, `topside`) has `train`, `query` and `gallery` splits. The paper's evaluation protocols
(query → gallery) combine them:

```python
from datasets import load_dataset

train = load_dataset("CEAai/MOO", "topside", split="train")

# Top -> Side
query = load_dataset("CEAai/MOO", "top", split="query")
gallery = load_dataset("CEAai/MOO", "side", split="gallery")
# Side -> Top: ("side", "query") vs ("top", "gallery"); Top-Side -> Top-Side: ("topside", ...) for both
```

The default download includes depth (~35% of the data). To skip it, stream and select columns; only the
requested columns are fetched (without `streaming=True`, whole files are downloaded and the columns dropped afterwards):

```python
query = load_dataset("CEAai/MOO", "top", split="query", streaming=True, columns=["image", "id", "view_id"])
```

Use `.with_format("numpy")` to get `depth` as a float32 array.

## Fields

| Field | Type | Description |
|---|---|---|
| `image` | image | 512×512 RGB render |
| `depth` | float32 512×512 | depth map; background is `1e10` and can be used as a mask |
| `id` | int32 | identity, `0-999` |
| `view_id` | int32 | viewpoint index in the 128-point grid |
| `view` | string | coarse direction label (`front`, `back-left`, `top-right`, …) |
| `azimuth`, `elevation` | float64 | camera angles in degrees |
| `location` | float64[3] | camera position in world coordinates |
| `cam2world` | float64 4×4 | camera-to-world matrix |
| `K` | float64 3×3 | camera intrinsics |

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
