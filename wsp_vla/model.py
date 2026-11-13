"""pi0.5 side of the recipe (openpi env, JAX): probes + closed-form fits.

    pi = Pi05()                                        # loads gs://openpi-assets/checkpoints/pi05_libero
    bg = fit_img_head_edit(pi, ["pairs/bg_object.npz"], beta=1.0)                       # background edit
    cam = fit_value_edit(pi, ["pairs/cam_*.npz"], pre_edits=[bg], beta=2.0, layers=range(9, 18))   # camera edit
    save_edit(bg, "edits/bg.npz"); save_edit(cam, "edits/cam.npz")

Pairs files come from `wsp_vla.sim.pairs` (LIBERO env). All statistics are accumulated on the GPU.
"""
from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from .edits import apply_edits
from .fisher import fisher_shrink, fisher_shrink_layers

CKPT = "gs://openpi-assets/checkpoints/pi05_libero"
IMG_TOKENS = 256                                     # SigLIP tokens per 224 px image; prefix = 3 images then prompt


class Pi05:
    """Wraps an openpi pi0.5 policy; `edits` are applied to a copy of the weights (no recompile on change)."""

    def __init__(self, config="pi05_libero", ckpt=CKPT):
        from openpi.policies import policy_config
        from openpi.training import config as _config
        self.policy = policy_config.create_trained_policy(_config.get_config(config), ckpt)
        self.graphdef, self.state = nnx.split(self.policy._model)
        self.base = self.state.to_pure_dict()
        self._cur = self.state
        self._enc = jax.jit(lambda st, x: nnx.merge(self.graphdef, st).PaliGemma.img(x, train=False)[0])
        self._vals = jax.jit(self._values)

    def set_edits(self, edits=()):
        st = jax.tree.map(lambda x: x, self.state)
        st.replace_by_pure_dict(apply_edits(self.base, list(edits)))
        self._cur = st
        return self

    # ------------------------------------------------------------------------------------------------ probes
    def image_tokens(self, images):
        """uint8 (B, 224, 224, 3) -> (B, 256, 2048) tokens after img/head (what the Gemma LLM reads)."""
        return self._enc(self._cur, jnp.asarray(images, jnp.float32) / 127.5 - 1.0).astype(jnp.float32)

    def _values(self, st, obs):
        from openpi.models.pi0 import make_attn_mask
        m = nnx.merge(self.graphdef, st)
        toks, pmask, ar = m.embed_prefix(obs)
        _, kv = m.PaliGemma.llm([toks, None], mask=make_attn_mask(pmask, ar), positions=jnp.cumsum(pmask, axis=1) - 1)
        return kv[1][:, :, :, 0, :].astype(jnp.float32), pmask              # values (L, B, S, head_dim)

    def prefix_values(self, images, wrist, state, prompts):
        """Per-layer attention values of the whole prefix (3 images x 256 tokens, then prompt) + valid mask."""
        from openpi.models import model as _model
        els = [self.policy._input_transform({"observation/image": i, "observation/wrist_image": w, "observation/state": s,
                                             "prompt": str(p)}) for i, w, s, p in zip(images, wrist, state, prompts)]
        obs = _model.Observation.from_dict(jax.tree.map(lambda *x: jnp.stack([jnp.asarray(v) for v in x]), *els))
        return self._vals(self._cur, obs)


# ---------------------------------------------------------------------------------------------------- fits
def _bilinear(grid, c):
    """grid (L, B, 16, 16, H) sampled at sub-token (row, col) c (B, T, 2) -> (L, B, T, H), valid mask (B, T)."""
    ok = (c[..., 0] >= -0.5) & (c[..., 1] >= -0.5) & (c[..., 0] <= 15.5) & (c[..., 1] <= 15.5)
    c = jnp.clip(c, 0, 15); r0 = jnp.floor(c[..., 0]).astype(int); q0 = jnp.floor(c[..., 1]).astype(int)
    r1, q1 = jnp.minimum(r0 + 1, 15), jnp.minimum(q0 + 1, 15)
    wr, wq = (c[..., 0] - r0)[None, ..., None], (c[..., 1] - q0)[None, ..., None]
    bi = jnp.arange(grid.shape[1])[:, None]
    g = lambda r, q: grid[:, bi, r, q]
    return (1 - wr) * (1 - wq) * g(r0, q0) + (1 - wr) * wq * g(r0, q1) + wr * (1 - wq) * g(r1, q0) + wr * wq * g(r1, q1), ok


def fit_img_head_edit(pi, bg_pair_files, beta=1.0, bs=8, **kw):
    """Background edit from background pairs (same token = same patch). Returns ImgHeadEdit."""
    from .edits import ImgHeadEdit
    s1, s2, sw, nt, nw = 0.0, 0.0, 0.0, 0, 0
    for path in bg_pair_files:
        D = np.load(path)
        for cam_clean, cam_bg in (("img", "bg_img"), ("wr", "bg_wr")):
            for i in range(0, len(D[cam_clean]), bs):
                z = pi.image_tokens(D[cam_clean][i:i + bs]).reshape(-1, 2048)
                s1 = s1 + z.sum(0); s2 = s2 + z.T @ z; nt += len(z)
                for k in range(D[cam_bg].shape[1]):
                    d = z - pi.image_tokens(D[cam_bg][i:i + bs, k]).reshape(-1, 2048)
                    sw = sw + d.T @ d; nw += len(d)
    mu = np.asarray(s1, np.float64) / nt
    St = np.asarray(s2, np.float64) / nt - np.outer(mu, mu); Sw = np.asarray(sw, np.float64) / nw / 2
    B, lam, g = fisher_shrink(St, Sw, beta, **kw)
    edit = ImgHeadEdit(B, mu)
    edit.stats = {"St": St, "Sw": Sw, "lam": lam}            # saved alongside the edit (plots / inspection)
    return edit


def fit_value_edit(pi, cam_pair_files, bg_pair_files=(), pre_edits=(), beta=2.0, layers=range(18), bs=4, **kw):
    """Camera (+ optionally background) edit on the VLM attention values, fit IN ORDER on top of `pre_edits`.
    Camera pairs: third-person tokens matched across views via depth (sub-token, bilinear). Uncentered moments
    (the value projection has no bias). Returns ValueEdit."""
    from .edits import ValueEdit
    pi.set_edits(pre_edits)
    L, H = 18, 256
    St, Sw, Sb = jnp.zeros((L, H, H)), jnp.zeros((L, H, H)), jnp.zeros((L, H, H)); nt = nw = nb = 0
    for path in cam_pair_files:
        D = np.load(path)
        for i in range(0, len(D["img"]), bs):
            sl = slice(i, i + bs); args = (D["wr"][sl], D["state"][sl], D["prompt"][sl])
            V0, pm = pi.prefix_values(D["img"][sl], *args)
            Vm = V0 * pm[None, :, :, None]
            St = St + jnp.einsum("lbsh,lbsk->lhk", Vm, Vm); nt += int(pm.sum())
            for k in range(D["cam_img"].shape[1]):
                V1, _ = pi.prefix_values(D["cam_img"][sl, k], *args)
                Vo, ok = _bilinear(V1[:, :, :IMG_TOKENS].reshape(L, -1, 16, 16, H), jnp.asarray(D["corr"][sl, k], jnp.float32))
                d = (V0[:, :, :IMG_TOKENS] - Vo) * ok[None, ..., None]
                Sw = Sw + jnp.einsum("lbsh,lbsk->lhk", d, d); nw += int(ok.sum())
    for path in bg_pair_files:                                                     # joint nuisance (optional)
        D = np.load(path)
        for i in range(0, len(D["img"]), bs):
            sl = slice(i, i + bs); args = (D["state"][sl], D["prompt"][sl])
            V0, pm = pi.prefix_values(D["img"][sl], D["wr"][sl], *args)
            Vm = V0 * pm[None, :, :, None]; St = St + jnp.einsum("lbsh,lbsk->lhk", Vm, Vm); nt += int(pm.sum())
            for k in range(D["bg_img"].shape[1]):
                V1, _ = pi.prefix_values(D["bg_img"][sl, k], D["bg_wr"][sl, k], *args)
                d = (V0 - V1)[:, :, :2 * IMG_TOKENS]
                Sb = Sb + jnp.einsum("lbsh,lbsk->lhk", d, d); nb += d.shape[1] * d.shape[2]
    St = np.asarray(St, np.float64) / nt
    Sw = np.asarray(Sw, np.float64) / nw / 2 + (np.asarray(Sb, np.float64) / nb / 2 if nb else 0)
    B, lam = fisher_shrink_layers(St, Sw, beta, **kw)
    pi.set_edits(())
    edit = ValueEdit(B, list(layers))
    edit.stats = {"St": St, "Sw": Sw, "lam": lam}
    return edit
