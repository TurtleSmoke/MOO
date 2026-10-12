"""Export MOO to a Hugging Face dataset repo (parquet + dataset card).

Each of the six disjoint subsets (train/test x top/side x query/gallery) is written once as its own
split; the dataset card (hf_card.md) composes them into the `top`, `side` and `topside` configs.
Split membership comes from make_splits.py, so the Hub and the split files cannot disagree.

Shards are written with pyarrow directly (datasets' writer takes ~1.5 s per row on the 512x512 depth
arrays) and existing shards are skipped, so an interrupted run resumes where it stopped.

Usage: uv run --group hf python hf_export.py --out hf_export [--repo CEAai/MOO [--private]] [--limit 8]
"""

import argparse
import io
import json
import shutil
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Array2D, Features, Image, Sequence, Value
from datasets.table import table_cast
from huggingface_hub import HfApi
from PIL import Image as PILImage

from make_splits import splits

SUBSETS = [
    "train_top",
    "train_side",
    "test_top_query",
    "test_top_gallery",
    "test_side_query",
    "test_side_gallery",
]
FEATURES = Features(
    {
        "image": Image(),
        "depth": Array2D((512, 512), "float32"),
        "id": Value("int32"),
        "view_id": Value("int32"),
        "view": Value("string"),
        "azimuth": Value("float64"),
        "elevation": Value("float64"),
        "location": Sequence(Value("float64"), length=3),
        "cam2world": Array2D((4, 4), "float64"),
        "K": Array2D((3, 3), "float64"),
    }
)
SHARD_ROWS = 1000  # ~270 MB per shard
ROW_GROUP_ROWS = 100  # keeps dataset-viewer reads small


def nested(arrays):
    """Stack equally shaped 2D arrays into an arrow list<list<...>> column."""
    stacked = np.stack(arrays)
    n, h, w = stacked.shape
    inner = pa.ListArray.from_arrays(np.arange(0, n * h * w + 1, w, dtype=np.int32), stacked.ravel())
    return pa.ListArray.from_arrays(np.arange(0, n * h + 1, h, dtype=np.int32), inner)


META = None


def load_meta(src):
    global META
    META = json.loads((src / "metadata.json").read_text())


def write_shard(job, src):
    path, pairs = job
    if path.exists():
        return
    cols = {k: [] for k in FEATURES}
    with h5py.File(src / "data.hdf5", "r") as f:
        for cow, view in pairs:
            key, m = f"cow{cow}/cow{cow}_{view}", META[f"cow{cow}"][f"cow{cow}_{view}"]
            png = io.BytesIO()
            PILImage.fromarray(f[f"{key}/colors"][()]).save(png, format="PNG")
            cols["image"].append({"bytes": png.getvalue(), "path": None})
            cols["depth"].append(f[f"{key}/depth"][()])
            cols["id"].append(cow)
            cols["view_id"].append(view)
            cols["view"].append(m["view"])
            cols["azimuth"].append(m["azimuth"])
            cols["elevation"].append(m["elevation"])
            cols["location"].append(m["location"])
            cols["cam2world"].append(np.asarray(m["cam2world_matrix"], dtype=np.float64))
            cols["K"].append(np.asarray(m["K"], dtype=np.float64))
    for k in ("depth", "cam2world", "K"):
        cols[k] = nested(cols[k])
    table = table_cast(pa.table(cols), FEATURES.arrow_schema)
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp, row_group_size=ROW_GROUP_ROWS, compression="zstd")
    tmp.rename(path)  # only complete shards get the final name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="local folder mirrored to the Hub repo")
    parser.add_argument("--src", type=Path, default=Path("."), help="folder with data.hdf5 and metadata.json")
    parser.add_argument("--repo", help="upload to this dataset repo once all shards are written")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--limit", type=int, help="rows per subset, for a dry run")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    all_splits, jobs = splits(), []
    for name in SUBSETS:
        pairs = all_splits[name][: args.limit]
        chunks = [pairs[i : i + SHARD_ROWS] for i in range(0, len(pairs), SHARD_ROWS)]
        jobs += [(args.out / "data" / f"{name}-{i:05d}-of-{len(chunks):05d}.parquet", c) for i, c in enumerate(chunks)]
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=load_meta, initargs=(args.src,)) as pool:
        for i, _ in enumerate(pool.imap_unordered(partial(write_shard, src=args.src), jobs), 1):
            print(f"{i}/{len(jobs)} shards", flush=True)
    shutil.copy("hf_card.md", args.out / "README.md")

    if args.repo:
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_large_folder(repo_id=args.repo, folder_path=args.out, repo_type="dataset")


if __name__ == "__main__":
    main()
