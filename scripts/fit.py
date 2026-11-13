"""Fit closed-form edits from paired data (openpi env, GPU ~10 GB).
usage: python scripts/fit.py background --pairs pairs/bg_object.npz --beta 1 --out edits/bg.npz
       python scripts/fit.py camera --pairs pairs/cam_*.npz --pre edits/bg.json --beta 2 --layers 9-17 --out edits/cam.npz
Writes <out>.npz and <out>.json (the edit spec; pass it to the server / evaluator / export)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp_vla.edits import load_edits, save_edit  # noqa: E402
from wsp_vla.model import Pi05, fit_img_head_edit, fit_value_edit  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["background", "camera"])
ap.add_argument("--pairs", nargs="+", required=True); ap.add_argument("--bg-pairs", nargs="*", default=[])
ap.add_argument("--pre", default="", help="edit spec .json applied before measuring (fit in order)")
ap.add_argument("--beta", type=float, default=1.0); ap.add_argument("--layers", default="0-17"); ap.add_argument("--out", required=True)
a = ap.parse_args()
pi = Pi05()
if a.kind == "background":
    edit = fit_img_head_edit(pi, a.pairs, beta=a.beta)
else:
    lo, hi = [int(x) for x in a.layers.split("-")]
    edit = fit_value_edit(pi, a.pairs, a.bg_pairs, load_edits(a.pre) if a.pre else (), beta=a.beta, layers=range(lo, hi + 1))
spec = save_edit(edit, a.out, **getattr(edit, "stats", {}))
pre = json.loads(Path(a.pre).read_text()) if a.pre else []
Path(a.out).with_suffix(".json").write_text(json.dumps(pre + [spec], indent=1))
print("wrote", a.out, "and", Path(a.out).with_suffix(".json"))
