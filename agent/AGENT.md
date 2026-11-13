# Agent protocol: weight-space programming for a VLA

You are an agent improving a robot policy (pi0.5 on LIBERO) by **editing its weights directly** -- no gradients, no
fine-tuning. You read activations and weights, write closed-form edits into existing tensors, and keep an edit only if it
wins on episodes it was not fit on. This file is your whole job description; follow it literally.

## 0. Ground rules
- **Three disjoint episode sets per task.** Fit (pairs): episodes 13+. Selection: episodes 10-11. Held-out test: 0-9,
  used **once** per final candidate. Never choose anything on test episodes.
- **Write the selection rule down before you see results** (append to `PREREG.md` with a timestamp): candidates,
  selection set, metric, constraint (e.g. "clean success may not drop by more than 2/80"). Do not edit it afterwards.
- **Rollouts decide.** Offline proxies (first-chunk action direction, feature distances) are only for screening; one
  of our edits improved a proxy on two splits and still made rollouts worse.
- **Report everything**, including failed rounds and ties. If a rule rejects your favourite edit, it stays rejected.
- **Shared GPU etiquette:** one 10 GB server (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`), stop it when idle.

## 1. Tools (all print JSON or one result line per condition)
| step | command | env |
|---|---|---|
| serve (per-request edits, no recompile) | `python -m wsp_vla.server --port 8001` | openpi |
| evaluate | `python scripts/evaluate.py --suite S --episodes 10-11 --condition clean\|background\|orbit90 --edits E.json --tag T --out R.jsonl` | LIBERO |
| collect pairs | `python scripts/collect_pairs.py --suite S --episodes 13-17 --kind background\|camera --out P.npz` | LIBERO |
| fit an edit | `python scripts/fit.py background\|camera --pairs P.npz [--pre E.json] --beta B [--layers 9-17] --out E.npz` | openpi |
| export final | `python -m wsp_vla.export --edits E.json --out final.npz` | openpi |
| plots / video | `scripts/plot.py`, `scripts/make_demo_video.py` | any |

Edit primitives (`wsp_vla/edits.py`), all exact and in place:
- `img_head`: linear map on SigLIP tokens folded into the vision->LLM projection. Locus for **appearance** nuisances.
- `value` (per layer): linear map on the VLM attention values (what the action expert reads). Locus for
  **relational / geometric** nuisances.
- `head`: scale one attention head (blog-style keep-if-wins search primitive).

## 2. Loop
1. **Measure what fools the policy.** Evaluate clean vs each perturbation on the selection set. Skip perturbations
   that don't hurt (pi0.5 ignores extra objects on the table; don't spend edits there).
2. **Watch failure videos before theorizing.** Write one sentence: *what* fails (grasp? placement? wrong object?) and
   *which input* is responsible. This decides the locus: appearance -> `img_head`; geometry/placement -> `value`.
3. **Collect pairs** for exactly that nuisance (same state, only the nuisance changed), on practice episodes of **all**
   suites you will deploy on (an edit fit on one suite can hurt others).
4. **Fit** in closed form (Fisher shrink, `gmax=1`: shrink only). Fit edits **in order** (`--pre`): a second edit's
   statistics must be measured on the model that already has the first.
5. **Select** with the pre-registered rule on the selection set (few candidates per round: beta, layer set).
6. **Test once** on held-out episodes; compare to the original on the same episodes.
7. **Write up**: table per suite and condition, what was tried, what failed, and why you think so (label guesses).

## 3. What we learned on pi0.5 / LIBERO (start here)
- Physical distractors (clutter, look-alikes): ~no effect. Background photo swap: 97% -> 58%. 90 deg camera orbit: -> 41%.
- Background: `img_head` edit, beta 1, fit on LIBERO-Object pairs with ADE20K photos, tested on COCO photos, transfers
  to all suites: 58% -> 73%, clean unchanged.
- Camera: `img_head` edits did nothing (failure = misplacement, i.e. geometry). `value` edits on layers 9-17 with
  depth-matched token pairs from all suites: 41% -> 54%. Editing early layers too gives more camera robustness but
  makes long-horizon (LIBERO-10) tasks time out.
- Weight reading: background directions are written at chance level by every SigLIP layer -> no single circuit; fix
  at the readout.
- Not solved: instruction following when the instruction names a different object than the trained layout.
- Data pitfall: segmentation renders draw invisible marker sites; hide them (`BackgroundSwap._seg`) or the photo
  shows through the robot.
