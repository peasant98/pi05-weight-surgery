"""Closed-loop LIBERO evaluation through the live-edit server (LIBERO env).
usage: python scripts/evaluate.py --suite libero_object --episodes 0-5 --condition background --edits edits/final.json --tag final --out results.jsonl
conditions: clean | background | orbit90 (orbitDEG) | background+orbit90
add --video-dir videos --video-res 512 to save sharp display videos (policy inputs are unchanged)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp_vla.sim.rollout import Condition, Evaluator, success_table  # noqa: E402

rng = lambda s: range(int(s.split("-")[0]), int(s.split("-")[-1]) + 1)
ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_object"); ap.add_argument("--tasks", default="0-9"); ap.add_argument("--episodes", default="0-2")
ap.add_argument("--condition", default="clean"); ap.add_argument("--edits", default="[]", help="JSON list or .json path")
ap.add_argument("--tag", default=""); ap.add_argument("--out", default="results.jsonl"); ap.add_argument("--port", type=int, default=8001)
ap.add_argument("--video-dir", default=None); ap.add_argument("--video-res", type=int, default=0); ap.add_argument("--photo", default=None)
a = ap.parse_args()
parts = a.condition.split("+")
cond = Condition(background="background" in parts, orbit=float(next((p[5:] for p in parts if p.startswith("orbit")), 0)), photo=a.photo)
edits = Path(a.edits).read_text() if a.edits.endswith(".json") else a.edits
recs = Evaluator(a.port).run(a.suite, rng(a.tasks), rng(a.episodes), cond, json.dumps(json.loads(edits)), a.tag, a.out, a.video_dir, a.video_res)
for k, (s, n) in success_table(recs).items():
    print(k, "%d/%d" % (s, n))
