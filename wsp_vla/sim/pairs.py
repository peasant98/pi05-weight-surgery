"""Paired observations for fitting edits (LIBERO env, Python 3.8).

During CLEAN rollouts of the stock policy (practice episodes, never used for testing), every `every` steps the same
state is rendered again with only the nuisance changed:
    background pairs: k background photos (ADE20K by default; tests use COCO -> different photo distribution)
    camera pairs:     k orbits of the third-person camera + depth-based sub-token correspondences
Output .npz keys:  img, wr, state, prompt  + (bg_img, bg_wr) or (cam_img, corr).
"""
from __future__ import annotations

import collections
import os
import pathlib

import numpy as np

from . import perturb as P

ADE = os.environ.get("WSP_ADE_DIR", "data/ADEChallengeData2016/images/training")   # fit-time backgrounds (!= test photos)


def collect(suite, tasks, episodes, out, kind="background", port=8001, every=20, k=2, angles=(30, 60, 90),
            photo_dir=ADE, seed=123):
    from libero.libero import benchmark, get_libero_path
    from openpi_client import websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", port)
    bench = benchmark.get_benchmark_dict()[suite]()
    rng = np.random.RandomState(seed); D = collections.defaultdict(list)
    for tid in tasks:
        task = bench.get_task(tid)
        env = P.make_env(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        inits = bench.get_task_init_states(tid)
        for ep in episodes:
            env.reset(); obs = env.set_init_state(inits[ep])
            plan, t, done = collections.deque(), 0, False
            while t < P.MAX_STEPS[suite] and not done:
                if t < 10:
                    obs, _, done, _ = env.step(P.DUMMY); t += 1; continue
                el = P.policy_inputs(obs, task.language)
                if t % every == 0:
                    D["img"].append(el["observation/image"]); D["wr"].append(el["observation/wrist_image"])
                    D["state"].append(el["observation/state"]); D["prompt"].append(str(task.language))
                    if kind == "background":
                        sw = P.BackgroundSwap(env, photo_dir=photo_dir, seed=rng.randint(1 << 30))
                        views = []
                        for _ in range(k):
                            sw.new_photo(); e2 = P.policy_inputs(sw(obs), task.language)
                            views.append((e2["observation/image"], e2["observation/wrist_image"]))
                        D["bg_img"].append(np.stack([v[0] for v in views])); D["bg_wr"].append(np.stack([v[1] for v in views]))
                    else:
                        sim = env.env.sim; m = sim.model; cid = m.camera_name2id("agentview")
                        pos0, quat0 = m.cam_pos[cid].copy(), m.cam_quat[cid].copy()
                        _, d0, f0 = P.render_depth(sim, "agentview")
                        views, corrs = [], []
                        for a in angles:
                            P.orbit_camera(env, a * rng.choice([-1, 1]))
                            rgb1, d1, f1 = P.render_depth(sim, "agentview")
                            views.append(P.to_policy_image(rgb1)); corrs.append(P.token_correspondence(d0, f0, d1, f1))
                            m.cam_pos[cid], m.cam_quat[cid] = pos0, quat0; sim.forward()
                        D["cam_img"].append(np.stack(views)); D["corr"].append(np.stack(corrs))
                if not plan:
                    plan.extend(client.infer(el)["actions"][:5])
                obs, _, done, _ = env.step(plan.popleft().tolist()); t += 1
            print(suite, tid, ep, len(D["img"]), flush=True)
        env.close()
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{key: np.stack(v) for key, v in D.items()})
    return out
