"""Live-edit policy server (openpi env): a standard openpi websocket server whose weights can be edited per request.

The model state is a jit ARGUMENT, so an edited copy with the same shapes reuses the compiled program: switching
edits costs a few ms, not a reload or recompile. Clients add to the observation dict:
    "__edit__": JSON list of edit specs (see wsp_vla.edits), "__seed__": int (fixed flow-matching noise).
Usage: python -m wsp_vla.server --port 8001 [--default-edits edits.json]
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict

import flax.nnx as nnx
import jax

from .edits import load_edits
from .model import Pi05


class LiveEditServer:
    def __init__(self, pi: Pi05, default="[]", cache_size=4):
        self.pi, self.default, self.cache = pi, default, OrderedDict()
        self.cache_size = cache_size
        p = pi.policy
        jitted = jax.jit(lambda st, rng, obs, **kw: nnx.merge(pi.graphdef, st).sample_actions(rng, obs, **kw))
        self._cur = pi.state
        p._sample_actions = lambda rng, obs, **kw: jitted(self._cur, rng, obs, **kw)
        self._infer = p.infer
        p.infer = self.infer

    def _state(self, spec):
        if spec not in self.cache:
            self.cache[spec] = self.pi.set_edits(load_edits(spec))._cur
            logging.info("built edit %s", spec[:100])
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(spec)
        return self.cache[spec]

    def infer(self, obs, **kw):
        obs = dict(obs)
        spec = obs.pop("__edit__", self.default) or "[]"
        self._cur = self.pi.state if spec == "[]" else self._state(spec)
        if "__seed__" in obs:
            self.pi.policy._rng = jax.random.key(int(obs.pop("__seed__")))
        return self._infer(obs, **kw)

    def serve(self, port):
        from openpi.serving import websocket_policy_server
        websocket_policy_server.WebsocketPolicyServer(policy=self.pi.policy, host="0.0.0.0", port=port,
                                                      metadata=self.pi.policy.metadata).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--default-edits", default="[]", help="JSON (or .json path) applied when a request has no __edit__")
    a = ap.parse_args()
    default = open(a.default_edits).read() if a.default_edits.endswith(".json") else a.default_edits
    srv = LiveEditServer(Pi05(), default=json.dumps(json.loads(default)))
    logging.info("live-edit server on %d (default edits: %s)", a.port, default[:80])
    srv.serve(a.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
