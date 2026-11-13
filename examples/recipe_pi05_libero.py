"""The full recipe as API calls. Two environments are involved (openpi = JAX model, LIBERO = Python 3.8 simulator),
so the steps are split; run each block in its env (or use the equivalent scripts/ commands).

# --- LIBERO env, with `python -m wsp_vla.server --port 8001` running (no edits) -------------------------------
from wsp_vla.sim.pairs import collect
for s in ["libero_object", "libero_spatial", "libero_goal", "libero_10"]:
    collect(s, range(10), range(13, 16), f"pairs/bg_{s}.npz", kind="background")   # ADE20K photos
    collect(s, range(10), range(13, 16), f"pairs/cam_{s}.npz", kind="camera")       # 30/60/90 deg orbits + depth matches
"""
import glob
import json

from wsp_vla.edits import save_edit
from wsp_vla.model import Pi05, fit_img_head_edit, fit_value_edit

# --- openpi env ------------------------------------------------------------------------------------------------
pi = Pi05()                                                                   # stock pi0.5-LIBERO
bg = fit_img_head_edit(pi, ["pairs/bg_libero_object.npz"], beta=1.0)          # 1) background: appearance -> img/head
cam = fit_value_edit(pi, sorted(glob.glob("pairs/cam_*.npz")), pre_edits=[bg],  # 2) camera: geometry -> VLM values,
                     beta=2.0, layers=range(9, 18))                            #    fit IN ORDER on top of the bg edit
spec = [save_edit(bg, "edits/bg.npz"), save_edit(cam, "edits/cam.npz")]
json.dump(spec, open("edits/final.json", "w"), indent=1)

# --- evaluate (LIBERO env, server running) -------------------------------------------------------------------------
# from wsp_vla.sim.rollout import Evaluator, Condition, success_table
# ev = Evaluator(port=8001); edits = open("edits/final.json").read()
# for cond in [Condition(), Condition(background=True), Condition(orbit=90)]:
#     for tag, e in [("orig", "[]"), ("final", edits)]:
#         print(tag, cond.name, success_table(ev.run("libero_object", range(10), range(0, 3), cond, e, tag)))

# --- export only the changed tensors (~85 MB) and serve with the stock openpi stack --------------------------------
# python -m wsp_vla.export --edits edits/final.json --out edits/pi05_libero_edited.npz
# from wsp_vla.export import patch_restore; patch_restore("edits/pi05_libero_edited.npz")  # before create_trained_policy
