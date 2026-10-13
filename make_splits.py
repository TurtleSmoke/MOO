"""Generate the MOO train/test split files in splits/.

Protocol (paper, Sec. 3): identities are split 500/500, even IDs train and odd IDs test.
Views 0-63 are side views (elevation < 30 deg), views 64-127 are top views (elevation > 30 deg).
Each test identity has half of its top views and half of its side views in the query set, and
the rest in the gallery. That half was drawn at random when the benchmark was built, so it is
stored in query_views.txt, one line per test identity: `<id>: <query view ids>`.

Usage: python make_splits.py [--out splits]
"""

import argparse
from pathlib import Path

NUM_IDS = 1000
VIEWS = {"side": range(64), "top": range(64, 128), "topside": range(128)}

QUERY_VIEWS = {
    int(cow): {int(v) for v in views.split()}
    for cow, views in (line.split(":") for line in (Path(__file__).parent / "query_views.txt").read_text().splitlines())
}


def is_train(cow):
    return cow % 2 == 0


def is_query(cow, view):
    return view in QUERY_VIEWS[cow]


def splits():
    """Return {split_name: [(cow, view), ...]}, sorted by cow then view."""
    out = {}
    for name, views in VIEWS.items():
        out[f"train_{name}"] = [(c, v) for c in range(NUM_IDS) if is_train(c) for v in views]
        test = [(c, v) for c in range(NUM_IDS) if not is_train(c) for v in views]
        out[f"test_{name}_query"] = [(c, v) for c, v in test if is_query(c, v)]
        out[f"test_{name}_gallery"] = [(c, v) for c, v in test if not is_query(c, v)]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("splits"))
    out = parser.parse_args().out
    out.mkdir(exist_ok=True)
    for name, rows in splits().items():
        (out / f"{name}.txt").write_text("".join(f"data/cow{c}/cow{c}_{v}.png\n" for c, v in rows))
        print(f"{out / name}.txt: {len(rows)} rows")


if __name__ == "__main__":
    main()
