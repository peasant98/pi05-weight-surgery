"""Closed-loop LIBERO evaluation against a (live-edit) policy server (LIBERO env, Python 3.8).

    ev = Evaluator(port=8001)
    recs = ev.run("libero_object", tasks=range(10), episodes=range(0, 3), condition=Condition(background=True),
                  edits=json_spec, video_dir="videos", video_res=512)
Condition: clean | background photo swap | camera orbit (degrees). Records are dicts (also appended to `out` jsonl).
"""
from __future__ import annotations

import collections
import json
import pathlib
from dataclasses import dataclass
from typing import Optional

import imageio
import numpy as np

from . import perturb as P


@dataclass
class Condition:
    background: bool = False
    orbit: float = 0.0                       # degrees; sign is randomized per episode (seeded) when > 0
    photo: Optional[str] = None              # fixed background photo (demos); else a seeded random COCO photo

    @property
    def name(self):
        return "+".join(n for n, on in (("bg", self.background), ("orbit%g" % self.orbit, self.orbit)) if on) or "clean"


class Evaluator:
    def __init__(self, port=8001, host="127.0.0.1", replan=5):
        from openpi_client import websocket_client_policy
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.replan = replan

    def episode(self, env, init_state, prompt, suite, condition, seed, edits="[]", video_res=0):
        obs = env.reset(); obs = env.set_init_state(init_state)
        swap = P.BackgroundSwap(env, seed=seed, photo=condition.photo) if condition.background else None
        if condition.orbit:
            P.orbit_camera(env, condition.orbit * np.random.RandomState(seed + 2000).choice([-1, 1]))   # same draw as the paper runs
        plan, frames, done, t = collections.deque(), [], False, 0
        while t < P.MAX_STEPS[suite] + 10:
            if t < 10:                                                            # let objects settle
                obs, _, done, _ = env.step(P.DUMMY); t += 1; continue
            if video_res:
                frames.append(swap.render(video_res) if swap else
                              np.ascontiguousarray(env.env.sim.render(video_res, video_res, camera_name="agentview")[::-1, ::-1]))
            if not plan:
                el = P.policy_inputs(swap(obs) if swap else obs, prompt); el["__edit__"] = edits
                plan.extend(self.client.infer(el)["actions"][: self.replan])
            obs, _, done, _ = env.step(plan.popleft().tolist()); t += 1
            if done:
                break
        return dict(success=bool(done), steps=t, photo=swap.file.name if swap else ""), frames

    def run(self, suite, tasks, episodes, condition=Condition(), edits="[]", tag="", out=None, video_dir=None, video_res=0):
        from libero.libero import benchmark, get_libero_path
        bench = benchmark.get_benchmark_dict()[suite]()
        vdir = pathlib.Path(video_dir) if video_dir else None
        if vdir:
            vdir.mkdir(parents=True, exist_ok=True)
        recs = []
        for tid in tasks:
            task = bench.get_task(tid)
            env = P.make_env(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
            inits = bench.get_task_init_states(tid)
            for ep in episodes:
                seed = 5000 + 100 * tid + ep
                r, frames = self.episode(env, inits[ep], task.language, suite, condition, seed, edits, video_res)
                r.update(suite=suite, task=tid, ep=ep, condition=condition.name, tag=tag, language=task.language)
                recs.append(r)
                if out:
                    with open(out, "a") as fh:
                        fh.write(json.dumps(r) + "\n")
                if vdir and frames:
                    imageio.mimwrite(vdir / ("%s_t%d_ep%d_%s_%s%s.mp4" % (suite, tid, ep, condition.name, "ok" if r["success"] else "fail", tag)),
                                     frames[::2], fps=10)
            env.close()
        return recs


def success_table(records, key=("tag", "suite", "condition")):
    agg = collections.defaultdict(list)
    for r in records:
        agg[tuple(r[k] for k in key)].append(r["success"])
    return {k: (sum(v), len(v)) for k, v in sorted(agg.items())}
