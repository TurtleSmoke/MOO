"""Build a web-ready glb for one MOO identity from the base cow glb and that identity's textures.

- swaps the coat colour/opacity texture for the identity's own (downscaled for the web)
- drops the normal map: the dataset renders are smooth, and model-viewer renders it as deep wrinkles
- body materials use alpha MASK: the coat texture's alpha cuts the eye sockets, so OPAQUE hides the eyes
- smooth normals are accumulated per vertex *position*, so vertices split along UV seams share one
  normal; per-vertex smoothing leaves creases ("folds") along every seam

The base can be any cow glb, e.g. src/assets/cows/cow74.glb (all identities share the mesh); the
script is idempotent on its own output.

Usage: python make_glb.py src/assets/cows/cow74.glb texture_dir out.glb [--size 1024]
"""

import argparse
import io

import numpy as np
import pygltflib
from PIL import Image

TEXTURES = {"colorOpacityCow": "colorOpacityCow.png"}


def read(glb, blob, idx):
    acc = glb.accessors[idx]
    bv = glb.bufferViews[acc.bufferView]
    dtype = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}[acc.componentType]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc.type]
    start = bv.byteOffset + (acc.byteOffset or 0)
    return np.frombuffer(blob, dtype, acc.count * width, start).reshape(acc.count, width)


def png(path, size):
    im = Image.open(path)
    im = im.resize((size, size), Image.LANCZOS) if im.width > size else im
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("textures")
    parser.add_argument("out")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    glb = pygltflib.GLTF2().load(args.base)
    blob = glb.binary_blob()

    # every bufferView's bytes, with the identity's textures swapped in; normals are appended below
    views = [blob[bv.byteOffset : bv.byteOffset + bv.byteLength] for bv in glb.bufferViews]
    for img in glb.images:
        if img.name in TEXTURES:
            views[img.bufferView] = png(f"{args.textures}/{TEXTURES[img.name]}", args.size)

    for mesh in glb.meshes:
        prims = mesh.primitives
        pos = [read(glb, blob, p.attributes.POSITION).astype(np.float64) for p in prims]
        tris = [read(glb, blob, p.indices).reshape(-1, 3) for p in prims]
        # weld identical positions across seams and primitives of the same mesh
        keys, inverse = np.unique(np.round(np.concatenate(pos), 5), axis=0, return_inverse=True)
        inverse = inverse.ravel()
        offsets = np.cumsum([0] + [len(p) for p in pos])
        acc = np.zeros((len(keys), 3))
        for p, t, o in zip(pos, tris, offsets):
            face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])  # area-weighted
            for k in range(3):
                np.add.at(acc, inverse[t[:, k] + o], face)
        acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
        for prim, o, n in zip(prims, offsets, np.diff(offsets)):
            normals = acc[inverse[o : o + n]].astype(np.float32)
            if prim.attributes.NORMAL is not None:  # already processed: overwrite in place
                views[glb.accessors[prim.attributes.NORMAL].bufferView] = normals.tobytes()
                continue
            glb.bufferViews.append(pygltflib.BufferView(buffer=0, byteLength=normals.nbytes, target=34962))
            views.append(normals.tobytes())
            glb.accessors.append(
                pygltflib.Accessor(
                    bufferView=len(glb.bufferViews) - 1,
                    componentType=pygltflib.FLOAT,
                    count=len(normals),
                    type=pygltflib.VEC3,
                    min=normals.min(0).tolist(),
                    max=normals.max(0).tolist(),
                )
            )
            prim.attributes.NORMAL = len(glb.accessors) - 1

    # drop the normal map texture, its image and its bufferView, shifting later indices
    for k in [i for i, img in enumerate(glb.images) if img.name == "colorOpacityCowNorm"]:
        (t,) = [i for i, tex in enumerate(glb.textures) if tex.source == k]
        view = glb.images[k].bufferView
        del glb.images[k], glb.textures[t], glb.bufferViews[view], views[view]
        for item in glb.accessors + glb.images:
            if item.bufferView is not None and item.bufferView > view:
                item.bufferView -= 1
        for tex in glb.textures:
            tex.source -= tex.source > k
        for m in glb.materials:
            m.normalTexture = None
            if m.pbrMetallicRoughness.baseColorTexture:
                m.pbrMetallicRoughness.baseColorTexture.index -= m.pbrMetallicRoughness.baseColorTexture.index > t

    for m in glb.materials:
        if "Eye" in m.name:
            m.alphaMode, m.doubleSided = "OPAQUE", True
        else:
            m.alphaMode, m.alphaCutoff, m.doubleSided = "MASK", 0.5, True

    # re-pack the binary chunk, 4-byte aligned
    packed = bytearray()
    for bv, data in zip(glb.bufferViews, views):
        packed += b"\0" * (-len(packed) % 4)
        bv.byteOffset, bv.byteLength = len(packed), len(data)
        packed += data
    glb.buffers[0].byteLength = len(packed)
    glb.set_binary_blob(bytes(packed))
    glb.save(args.out)
    print(f"{args.out}: {len(packed) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
