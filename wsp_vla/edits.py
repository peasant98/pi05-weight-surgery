"""Weight edits for pi0.5, folded EXACTLY into existing tensors (same architecture, same speed).

ImgHeadEdit  z' = z + (z - mu) B^T on SigLIP tokens after `img/head`      ->  W' = W + W B^T,  b' = b + (b - mu) B^T
ValueEdit    v' = v (I + B_l^T) on the VLM attention VALUES of layer l    ->  W_v' = W_v + W_v B_l^T
             (values carry no RoPE and kv_einsum has no bias, so this is exact)
HeadScale    scale one attention head's output projection (blog-style keep-if-wins search primitive)

Edits operate on the *pure dict* of model params (openpi `nnx.State.to_pure_dict()` or `restore_params(...)`), with
numpy or JAX arrays. Each edit serializes to a small JSON dict that points at an .npz with its matrices.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

IMG_HEAD = ("PaliGemma", "img", "head")
KV = ("PaliGemma", "llm", "layers", "attn", "kv_einsum")                    # w: (layers, 2[K,V], kv_heads, width, head_dim)
ATTN_OUT = {"vlm": "attn_vec_einsum", "expert": "attn_vec_einsum_1"}       # w: (layers, heads, head_dim, width)


def _node(params, path):
    for k in path:
        params = params[k]
    return params


def _xp(a):
    if type(a).__module__.startswith("jax"):
        import jax.numpy as jnp
        return jnp
    return np


def _mm(xp, a, b):
    """Full-precision matmul (JAX on GPU defaults to TF32 for float32, ~3 significant digits)."""
    return xp.matmul(a, b, precision="highest") if xp is not np else a @ b


def _set(a, idx, val):
    """Functional for JAX arrays, copy-and-assign for numpy."""
    if type(a).__module__.startswith("jax"):
        return a.at[idx].set(val)
    a = np.array(a); a[idx] = val
    return a


@dataclass
class ImgHeadEdit:
    B: np.ndarray                     # (2048, 2048)
    mu: np.ndarray                    # (2048,)
    scale: float = 1.0
    kind: str = "img_head"

    def apply(self, params):
        head = _node(params, IMG_HEAD); W, b = head["kernel"], head["bias"]; xp = _xp(W)
        B = xp.asarray(self.B * self.scale, xp.float32); mu = xp.asarray(self.mu, xp.float32)
        Wf, bf = W.astype(xp.float32), b.astype(xp.float32)
        head["kernel"] = (Wf + _mm(xp, Wf, B.T)).astype(W.dtype)
        head["bias"] = (bf + _mm(xp, bf - mu, B.T)).astype(b.dtype)
        return params


@dataclass
class ValueEdit:
    B: np.ndarray                     # (layers, head_dim, head_dim)
    layers: Sequence[int] = field(default_factory=lambda: list(range(18)))
    scale: float = 1.0
    kind: str = "value"

    def apply(self, params):
        node = _node(params, KV); w = node["w"]; xp = _xp(w)
        for l in self.layers:
            Wv = w[l, 1, 0].astype(xp.float32)
            w = _set(w, (l, 1, 0), (Wv + _mm(xp, Wv, xp.asarray(self.B[l] * self.scale, xp.float32).T)).astype(w.dtype))
        node["w"] = w
        return params


@dataclass
class HeadScale:
    layer: int
    head: int
    scale: float
    part: str = "vlm"
    kind: str = "head_scale"

    def apply(self, params):
        node = _node(params, ("PaliGemma", "llm", "layers", "attn", ATTN_OUT[self.part])); w = node["w"]
        node["w"] = _set(w, (self.layer, self.head), w[self.layer, self.head] * self.scale)
        return params


def apply_edits(params, edits):
    """Apply edits in order to a shallow copy of the params dict (unchanged leaves are shared, not copied)."""
    import copy
    out = copy.copy(params)
    def clone(d):                                         # copy dict structure along the edited paths only
        return {k: clone(v) if isinstance(v, dict) else v for k, v in d.items()}
    out = clone(out)
    for e in edits:
        out = e.apply(out)
    return out


# ------------------------------------------------------------------------------------------------- serialization
def save_edit(edit, path, **extra):
    """Save an edit's arrays to .npz and return its JSON spec."""
    path = str(path)
    if isinstance(edit, ImgHeadEdit):
        np.savez(path, B=edit.B, mu=edit.mu, **extra)
        return {"op": "img_head", "path": path, "scale": edit.scale}
    if isinstance(edit, ValueEdit):
        np.savez(path, B=edit.B, **extra)
        return {"op": "value", "path": path, "layers": list(edit.layers), "scale": edit.scale}
    raise TypeError(edit)


def load_edit(spec):
    """JSON spec -> edit. Also accepts the research-code format {"op": "kv_value"/"img_head", "key": ...}."""
    op = spec["op"]
    if op == "head":
        return HeadScale(spec["layer"], spec["head"], spec["scale"], spec.get("part", "vlm"))
    f = np.load(spec["path"]); key = spec.get("key", "B")
    if op == "img_head":
        return ImgHeadEdit(f[key], f["mu"], spec.get("scale", 1.0))
    if op in ("value", "kv_value"):
        return ValueEdit(f[key], spec.get("layers", list(range(18))), spec.get("scale", 1.0))
    raise ValueError(op)


def load_edits(spec):
    """Edits from a JSON string, a path to a .json file, or an already-parsed list of specs."""
    if isinstance(spec, Path) or (isinstance(spec, str) and spec.strip().endswith(".json") and Path(spec).exists()):
        spec = Path(spec).read_text()
    specs = json.loads(spec) if isinstance(spec, str) else spec
    return [load_edit(s) for s in specs]
