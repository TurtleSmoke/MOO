"""
pack.py: Utility for packing/unpacking standard images, depth maps, and metadata.
Usage:
  python pack.py pack   --src unpacked_folder --out archive_folder
  python pack.py unpack --src archive_folder  --out unpacked_folder [--png] [--depth] [--json]

  Both pack and unpack skip already-processed items and can be safely re-run unless a file was partially written.
"""

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


@contextmanager
def _step(desc):
    tqdm.write(f"  {desc}...")
    t0 = time.monotonic()
    yield
    tqdm.write(f"  done in {time.monotonic() - t0:.1f}s")


def get_samples(data_dir):
    """Returns a sorted list of (cow_dir, stem) for all valid samples."""
    return (
        sorted(
            (d, f.stem) for d in data_dir.iterdir() if d.is_dir() for f in d.iterdir() if f.suffix in (".png", ".npy")
        )
        if data_dir.exists()
        else []
    )


def pack_archive(src, out):
    out.mkdir(parents=True, exist_ok=True)
    data_dir = src / "data" if (src / "data").is_dir() else src
    hdf5_path = out / "data.hdf5"

    with _step("Discovering samples"):
        samples = get_samples(data_dir)
        tqdm.write(f"  {len(samples)} samples found.")

    all_meta = {}
    with h5py.File(hdf5_path, "a") as f:
        with _step("Scanning HDF5 for already-packed samples"):
            packed = {f"{cow}/{stem}" for cow in f for stem in f[cow]}
            tqdm.write(f"  {len(packed)} already packed.")

        to_process = [(cow_dir, stem) for cow_dir, stem in samples if f"{cow_dir.name}/{stem}" not in packed]
        tqdm.write(f"  {len(to_process)} samples to pack.")

        for cow_dir, stem in tqdm(to_process, desc="Packing", unit="sample"):
            key = f"{cow_dir.name}/{stem}"
            png, depth, meta = (
                cow_dir / f"{stem}.png",
                cow_dir / f"{stem}.npy",
                cow_dir / f"{stem}_metadata.json",
            )

            if png.exists() and f"{key}/colors" not in f:
                f.create_dataset(f"{key}/colors", data=np.array(Image.open(png)), compression="lzf")
            if depth.exists() and f"{key}/depth" not in f:
                f.create_dataset(f"{key}/depth", data=np.load(depth), compression="lzf")
            if meta.exists():
                all_meta.setdefault(cow_dir.name, {})[stem] = json.loads(meta.read_text())

    with _step("Writing metadata.json"):
        (out / "metadata.json").write_text(json.dumps(all_meta, indent=2))


def _write_worker(out_dir, stem, colors, depth, meta):
    out_dir.mkdir(parents=True, exist_ok=True)

    if colors is not None and not (out_dir / f"{stem}.png").exists():
        Image.fromarray(colors).save(out_dir / f"{stem}.png")
    if depth is not None and not (out_dir / f"{stem}.npy").exists():
        np.save(out_dir / f"{stem}.npy", depth)
    if meta is not None and not (out_dir / f"{stem}_metadata.json").exists():
        (out_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2))


def _build_existing(data_dir):
    """Walk output dir once into a set — avoids per-file stat() calls on resume."""
    if not data_dir.exists():
        return set()
    base = str(data_dir)
    return {
        os.path.join(rel, fn) if (rel := os.path.relpath(dirpath, base)) != "." else fn
        for dirpath, _, filenames in os.walk(data_dir)
        for fn in filenames
    }


def unpack_archive(src, dst, png, depth, meta):
    all_meta = json.loads((src / "metadata.json").read_text()) if (src / "metadata.json").exists() else {}
    data_dir = dst / "data"

    with h5py.File(src / "data.hdf5", "r") as f:
        with _step("Listing HDF5 keys"):
            keys = [(cow, stem) for cow in f for stem in f[cow]]
            hdf5_ds = {f"{cow}/{stem}/{ds}" for cow, stem in keys for ds in f[f"{cow}/{stem}"]}
            tqdm.write(f"  {len(keys)} samples in archive.")

        with _step("Scanning output directory"):
            existing = _build_existing(data_dir)
            tqdm.write(f"  {len(existing)} files already on disk.")

        def needs(cow, stem, ds, ext):
            return f"{cow}/{stem}/{ds}" in hdf5_ds and os.path.join(cow, f"{stem}{ext}") not in existing

        with _step("Computing resume list"):
            to_process = [
                (cow, stem)
                for cow, stem in keys
                if (png and needs(cow, stem, "colors", ".png"))
                or (depth and needs(cow, stem, "depth", ".npy"))
                or (
                    meta
                    and all_meta.get(cow, {}).get(stem)
                    and os.path.join(cow, f"{stem}_metadata.json") not in existing
                )
            ]
            tqdm.write(f"  {len(keys) - len(to_process)} already done, {len(to_process)} to unpack.")

        with (
            ThreadPoolExecutor() as executor,
            tqdm(total=len(to_process), desc="Unpacking", unit="sample") as pbar,
        ):
            for cow, stem in to_process:
                group = f[f"{cow}/{stem}"]
                c_data = group["colors"][:] if png and "colors" in group else None
                d_data = group["depth"][:] if depth and "depth" in group else None
                m_data = all_meta.get(cow, {}).get(stem) if meta else None
                future = executor.submit(_write_worker, data_dir / cow, stem, c_data, d_data, m_data)
                future.add_done_callback(lambda _: pbar.update(1))


def main():
    p = argparse.ArgumentParser(description="Pack/unpack dataset images, depth maps, and metadata.")
    p.add_argument("cmd", choices=["pack", "unpack"])
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--png", action="store_true", help="Extract RGB images (default if no flags).")
    p.add_argument("--depth", action="store_true", help="Extract depth maps (default if no flags).")
    p.add_argument("--json", action="store_true", help="Extract metadata   (default if no flags).")
    args = p.parse_args()

    if args.cmd == "pack":
        pack_archive(args.src, args.out)
    else:
        any_flag = any([args.png, args.depth, args.json])
        unpack_archive(
            args.src,
            args.out,
            png=args.png or not any_flag,
            depth=args.depth or not any_flag,
            meta=args.json or not any_flag,
        )


if __name__ == "__main__":
    main()
