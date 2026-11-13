"""Figures: python scripts/plot.py results results.jsonl --out results.png    (bars per condition, original vs edited tags)
           python scripts/plot.py spectrum stats.npz --out spectrum.png          (needs keys St, Sw; per-layer -> layer plot)"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp_vla.viz import plots  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["results", "spectrum"]); ap.add_argument("path"); ap.add_argument("--out", required=True)
ap.add_argument("--orig-tag", default="orig"); ap.add_argument("--edit-tag", default="final")
a = ap.parse_args()
if a.what == "results":
    agg = collections.defaultdict(list)
    for r in map(json.loads, open(a.path)):
        agg[(r["tag"], r["condition"])].append(r["success"])
    conds = sorted({c for _, c in agg})
    pct = lambda t, c: 100 * np.mean(agg[(t, c)]) if agg[(t, c)] else 0
    print(plots.results_bars(conds, [pct(a.orig_tag, c) for c in conds], [pct(a.edit_tag, c) for c in conds], a.out))
else:
    f = np.load(a.path)
    print(plots.layer_nuisance(f["St"], f["Sw"], a.out) if f["St"].ndim == 3 else plots.fisher_spectrum(f["St"], f["Sw"], out=a.out))
