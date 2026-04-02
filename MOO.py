# encoding: utf-8
import re
import os.path as osp

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = [
    # Training
    "MOO_TRAIN_TOP",
    "MOO_TRAIN_SIDE",
    "MOO_TRAIN_TOPSIDE",
    # Evaluation (Query -> Gallery)
    "MOO_TEST_TOP_TO_SIDE",
    "MOO_TEST_SIDE_TO_TOP",
    "MOO_TEST_TOPSIDE_TO_TOPSIDE",
]


class MOOBase(ImageDataset):
    """Base class for the MOO dataset.

    Reference:
       Grolleau, et al. MOO: A Multi-view Oriented Observations Dataset for Viewpoint Analysis
        in Cattle Re-Identification.

    URL: `<https://github.com/TurtleSmoke/MOO>`

    Dataset statistics:
        - identities: 1,000.
        - Train images: 64,000 (top/side), 128,000 (topside)
        - Test images: 16,000 (top/side query/gallery), 32,000 (topside query/gallery)

    Notes:
        Since all images use the same camera, we force different `camid` for query and gallery
        to avoid market-1501-like evaluation ignoring same-camera matches.
    """

    dataset_dir = "MOO"

    def __init__(
        self,
        root="datasets",
        train_split=None,
        query=None,
        gallery=None,
        **kwargs,
    ):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        splits_dir = osp.join(self.dataset_dir, "splits")
        train = (
            self._load_split(osp.join(splits_dir, f"train_{train_split}.txt"))
            if train_split
            else []
        )
        query = (
            self._load_split(osp.join(splits_dir, f"test_{query}_query.txt"), camid=0)
            if query
            else []
        )
        gallery = (
            self._load_split(
                osp.join(splits_dir, f"test_{gallery}_gallery.txt"), camid=1
            )
            if gallery
            else []
        )

        super().__init__(train, query, gallery, **kwargs)

    def _load_split(self, txt_path, camid=0):
        if not osp.exists(txt_path):
            raise FileNotFoundError(f"Split file not found: {txt_path}")

        paths = map(str.strip, open(txt_path).readlines())
        rows = [
            (
                osp.join(self.dataset_dir, line),
                int(re.search(r"cow(\d+)", line).group(1)),
                camid,
            )
            for line in paths
        ]

        return rows


@DATASET_REGISTRY.register()
class MOO_TRAIN_TOP(MOOBase):
    def __init__(self, root="datasets", **kwargs):
        super().__init__(root=root, train_split="top", **kwargs)


@DATASET_REGISTRY.register()
class MOO_TRAIN_SIDE(MOOBase):
    def __init__(self, root="datasets", **kwargs):
        super().__init__(root=root, train_split="side", **kwargs)


@DATASET_REGISTRY.register()
class MOO_TRAIN_TOPSIDE(MOOBase):
    def __init__(self, root="datasets", **kwargs):
        super().__init__(root=root, train_split="topside", **kwargs)


@DATASET_REGISTRY.register()
class MOO_TEST_TOP_TO_SIDE(MOOBase):
    def __init__(self, root="datasets", **kwargs):
        super().__init__(root=root, query="top", gallery="side", **kwargs)


@DATASET_REGISTRY.register()
class MOO_TEST_SIDE_TO_TOP(MOOBase):
    def __init__(self, root="datasets", **kwargs):
        super().__init__(root=root, query="side", gallery="top", **kwargs)


@DATASET_REGISTRY.register()
class MOO_TEST_TOPSIDE_TO_TOPSIDE(MOOBase):
    def __init__(self, root="datasets", **kwargs):
        super().__init__(root=root, query="topside", gallery="topside", **kwargs)
