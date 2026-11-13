"""Export edits as ONLY the changed tensors (~85 MB instead of a 12 GB checkpoint), and load them back.

    python -m wsp_vla.export --edits edits/final.json --out edits/pi05_libero_edited.npz      (openpi env, CPU ok)
    serve:  WSP_EDITED=edits/pi05_libero_edited.npz python ~/openpi/scripts/serve_policy.py ...   via `patch_restore()`
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .edits import KV, IMG_HEAD, apply_edits, load_edits

LEAVES = {"img_head_kernel": IMG_HEAD + ("kernel",), "img_head_bias": IMG_HEAD + ("bias",), "llm_kv_einsum_w": KV + ("w",)}
CKPT_PARAMS = "gs://openpi-assets/checkpoints/pi05_libero/params"


def _get(d, path):
    for k in path:
        d = d[k]
    return d


def export(edits, out, params_path=CKPT_PARAMS):
    import openpi.models.model as _model
    from openpi.shared import download
    p = _model.restore_params(download.maybe_download(params_path), restore_type=np.ndarray)
    q = apply_edits(p, edits)
    tensors = {n: np.asarray(_get(q, path)) for n, path in LEAVES.items() if not np.array_equal(_get(p, path), _get(q, path))}
    np.savez(out, **tensors)
    return list(tensors)


def load_into(params, path):
    """Replace the exported tensors inside a restored params dict (in place)."""
    f = np.load(path)
    for n, pth in LEAVES.items():
        if n in f:
            node = _get(params, pth[:-1]); old = node[pth[-1]]
            assert old.shape == f[n].shape, (n, old.shape, f[n].shape)
            node[pth[-1]] = f[n].astype(old.dtype)
    return params


def patch_restore(path=None):
    """Make every openpi `restore_params` call return the edited weights (use before create_trained_policy)."""
    import openpi.models.model as _model
    path = path or os.environ["WSP_EDITED"]
    orig = _model.restore_params
    _model.restore_params = lambda *a, **kw: load_into(orig(*a, **kw), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits", required=True, help="JSON list or .json path of edit specs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print("exported", export(load_edits(a.edits), a.out), "->", a.out)


if __name__ == "__main__":
    main()
