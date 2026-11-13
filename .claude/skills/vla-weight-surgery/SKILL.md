---
name: vla-weight-surgery
description: Make a VLA policy (pi0.5 on LIBERO) robust to a visual nuisance by editing its weights in closed form (no gradients) - diagnose failures, collect paired sim renders, fit Fisher-shrink edits into existing tensors, select with a pre-registered rule, test once on held-out episodes. Use when asked to "fix", "harden" or "edit the weights of" a robot policy against backgrounds, camera pose or similar distractors.
---

# VLA weight surgery

Follow `agent/AGENT.md` in this repo exactly (ground rules, tools, loop, lessons). Short version:

1. Start the live-edit server: `cd ~/openpi && XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 .venv/bin/python -m wsp_vla.server --port 8001`
   (with this repo on `PYTHONPATH`). Stop it when idle.
2. Measure: `scripts/evaluate.py` on selection episodes 10-11, clean vs perturbation, in the LIBERO venv.
3. Watch failure videos (`--video-dir v --video-res 512`), name the failure, pick the locus (`img_head` = appearance,
   `value` = geometry).
4. `scripts/collect_pairs.py` on practice episodes 13+ of every suite -> `scripts/fit.py` (fit in order with `--pre`).
5. Append the selection rule to `PREREG.md` **before** running candidates; select; test once on episodes 0-9.
6. Report per suite and condition, including failures. Export the winner with `python -m wsp_vla.export`.
