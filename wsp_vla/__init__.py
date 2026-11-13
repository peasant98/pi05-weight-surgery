"""wsp_vla: closed-form weight-space edits that make pi0.5 robust to visual nuisances (no gradients).

Model side (openpi env):  wsp_vla.model (Pi05, fit_img_head_edit, fit_value_edit), wsp_vla.server, wsp_vla.export
Shared (numpy):           wsp_vla.fisher, wsp_vla.edits
Simulator side (LIBERO):  wsp_vla.sim.perturb, wsp_vla.sim.pairs, wsp_vla.sim.rollout
Visualization:            wsp_vla.viz
"""
__version__ = "0.1.0"
