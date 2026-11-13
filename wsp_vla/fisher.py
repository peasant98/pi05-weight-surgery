"""Closed-form nuisance removal (no gradients).

Given paired features -- the same state rendered normally and with a nuisance changed (background photo, camera pose) --
find the directions that move mostly because of the nuisance and shrink them.

    S_t : scatter of normal features          (how much each direction varies at all)
    S_w : scatter of within-pair differences  (how much it varies ONLY because of the nuisance)

Generalized eigenvectors of (S_w, S_t) give lambda = nuisance share per direction. The gain
g = (lambda / mean(lambda)) ** (-beta / 2), clipped to [gmin, gmax], shrinks nuisance-dominated directions
(gmax = 1: shrink only, never amplify). The map is returned as B with  z' = z + (z - mu) B^T.
"""
from __future__ import annotations

import numpy as np


def fisher_shrink(St, Sw, beta=1.0, eps=0.05, gmin=0.1, gmax=1.0):
    """Return (B, lam, g). St, Sw: (n, n) symmetric. eps regularizes St by eps * mean variance."""
    St, Sw = np.asarray(St, np.float64), np.asarray(Sw, np.float64)
    n = len(St)
    S = St + eps * np.trace(St) / n * np.eye(n)
    Li = np.linalg.inv(np.linalg.cholesky(S))
    lam, Q = np.linalg.eigh(Li @ Sw @ Li.T)
    V = Li.T @ Q                                           # V^T S V = I
    lam = np.clip(lam, 1e-9, None)
    g = np.clip((lam / lam.mean()) ** (-beta / 2), gmin, gmax)
    return S @ V @ np.diag(g - 1) @ V.T, lam, g


def fisher_shrink_layers(St, Sw, beta=1.0, **kw):
    """Per-layer version: St, Sw (L, n, n) -> B (L, n, n), lam (L, n)."""
    out = [fisher_shrink(St[l], Sw[l], beta, **kw) for l in range(len(St))]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])
