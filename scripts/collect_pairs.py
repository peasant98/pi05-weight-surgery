"""Collect paired observations for fitting (LIBERO env; needs a running policy server, any edits off).
usage: python scripts/collect_pairs.py --suite libero_object --tasks 0-9 --episodes 10-14 --kind background --out pairs/bg_object.npz
       python scripts/collect_pairs.py --suite libero_goal --episodes 13-15 --kind camera --out pairs/cam_goal.npz
Use PRACTICE episodes only (e.g. 10+); keep 0-9 for testing."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp_vla.sim.pairs import collect  # noqa: E402

rng = lambda s: range(int(s.split("-")[0]), int(s.split("-")[-1]) + 1)
ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_object"); ap.add_argument("--tasks", default="0-9"); ap.add_argument("--episodes", default="10-14")
ap.add_argument("--kind", choices=["background", "camera"], default="background"); ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=8001); ap.add_argument("--every", type=int, default=20)
a = ap.parse_args()
print("wrote", collect(a.suite, rng(a.tasks), rng(a.episodes), a.out, kind=a.kind, port=a.port, every=a.every))
