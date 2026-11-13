"""Static figures (matplotlib): what the edit shrinks, and results bars. Blue theme to match the videos."""
from __future__ import annotations

import numpy as np

NAVY, BLUE, CORAL, GREY = "#080e1e", "#60b2ff", "#ff7878", "#a0a8b9"


def _style(ax, fig):
    fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY)
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.tick_params(colors=GREY); ax.xaxis.label.set_color(GREY); ax.yaxis.label.set_color(GREY); ax.title.set_color("white")


def fisher_spectrum(St, Sw, beta=1.0, out="fisher_spectrum.png", title="what the edit shrinks", **kw):
    """Per-direction nuisance share lambda (sorted) and the resulting gain g for one feature space."""
    import matplotlib.pyplot as plt
    from ..fisher import fisher_shrink
    _, lam, g = fisher_shrink(St, Sw, beta, **kw)
    o = np.argsort(-lam)
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=200); _style(ax, fig)
    ax.semilogy(lam[o], color=CORAL, lw=1.5, label="nuisance share  λ")
    ax2 = ax.twinx(); ax2.plot(g[o], color=BLUE, lw=1.5, label="gain g"); ax2.set_ylim(0, 1.05); ax2.tick_params(colors=GREY)
    ax.set_xlabel("direction (sorted by nuisance share)"); ax.set_ylabel("λ"); ax2.set_ylabel("gain", color=GREY)
    ax.set_title(title)
    fig.legend(loc="upper right", frameon=False, labelcolor="white", fontsize=8)
    fig.tight_layout(); fig.savefig(out, facecolor=NAVY); plt.close(fig)
    return out


def layer_nuisance(St, Sw, out="layer_nuisance.png", title="viewpoint share of value variance, per VLM layer"):
    """tr(S_w)/tr(S_t) per layer (e.g. camera pairs on attention values)."""
    import matplotlib.pyplot as plt
    frac = [np.trace(Sw[l]) / np.trace(St[l]) for l in range(len(St))]
    fig, ax = plt.subplots(figsize=(7, 3), dpi=200); _style(ax, fig)
    ax.bar(range(len(frac)), frac, color=BLUE); ax.set_xlabel("layer"); ax.set_ylabel("nuisance share"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, facecolor=NAVY); plt.close(fig)
    return out


def results_bars(conditions, original, edited, out="results.png", title="pi0.5 on LIBERO, held-out episodes"):
    """conditions: names; original / edited: success rates in % (same order)."""
    import matplotlib.pyplot as plt
    x = np.arange(len(conditions)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=200); _style(ax, fig)
    b1 = ax.bar(x - w / 2, original, w, color=GREY, label="original"); b2 = ax.bar(x + w / 2, edited, w, color=BLUE, label="+ weight edit")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, "%d%%" % round(b.get_height()), ha="center", color="white", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(conditions, color="white"); ax.set_ylim(0, 110); ax.set_ylabel("success (%)")
    ax.set_title(title); ax.legend(frameon=False, labelcolor="white", fontsize=8)
    fig.tight_layout(); fig.savefig(out, facecolor=NAVY); plt.close(fig)
    return out
