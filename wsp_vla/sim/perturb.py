"""LIBERO perturbations (LIBERO env, Python 3.8): background swap, camera orbit, cross-view token correspondences."""
from __future__ import annotations

import math
import os
import pathlib

import numpy as np
from openpi_client import image_tools

RES = 256                                  # LIBERO camera resolution used by the openpi example
POLICY_RES = 224                           # policy input (SigLIP), 16 x 16 tokens of 14 px
BG_KEYS = ("table", "floor", "wall")
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
DUMMY = [0.0] * 6 + [-1.0]


def make_env(bddl, seed=7, res=RES):
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=res, camera_widths=res)
    env.seed(seed)
    return env


def quat2axisangle(q):
    q = np.array(q, dtype=np.float64); q[3] = min(max(q[3], -1.0), 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den


def policy_inputs(obs, prompt):
    """openpi LIBERO example preprocessing (images rotated 180 deg, resized with padding to 224)."""
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
    wr = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
    st = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    return {"observation/image": img, "observation/wrist_image": wr, "observation/state": st, "prompt": str(prompt)}


class BackgroundSwap:
    """Replace background pixels (floor / walls / table, from the simulator's segmentation) with a photo, in both
    cameras. Invisible marker SITES are hidden during the segmentation pass (otherwise they occlude the robot and the
    photo shows through it as a thin line)."""

    def __init__(self, env, photo_dir=None, seed=0, photo=None):
        photo_dir = photo_dir or os.environ.get("WSP_COCO_DIR", "data/coco/val2017")     # test-time backgrounds
        m = env.env.sim.model
        self.env, self.rng = env, np.random.RandomState(seed)
        self.is_bg = np.array([any(k in m.body_id2name(m.geom_bodyid[g]) for k in BG_KEYS) or m.body_id2name(m.geom_bodyid[g]) == "world"
                               for g in range(m.ngeom)] + [True])               # last entry: id -1 (nothing / site)
        self.files = sorted(pathlib.Path(photo_dir).expanduser().glob("*.jpg"))
        self.fixed = photo
        self.new_photo()

    def new_photo(self):
        from PIL import Image
        self.file = pathlib.Path(self.fixed) if self.fixed else self.files[self.rng.randint(len(self.files))]
        self._img, self._cache = Image.open(self.file).convert("RGB"), {}
        return self.file.name

    def _photo(self, res):
        if res not in self._cache:
            from PIL import Image
            self._cache[res] = np.asarray(self._img.resize((res, res), Image.LANCZOS))[::-1]   # raw frames are upside down
        return self._cache[res]

    def _seg(self, res, cam):
        m = self.env.env.sim.model                                                 # env.reset() rebuilds the sim: look up
        old = m.site_size.copy(); m.site_size[:] = 0
        try:
            return self.env.env.sim.render(res, res, camera_name=cam, segmentation=True)
        finally:
            m.site_size[:] = old

    def mask(self, res, cam):
        seg = self._seg(res, cam)
        return self.is_bg[np.where(seg[..., 0] == 5, seg[..., 1], -1)]

    def __call__(self, obs):
        """Policy observations with the background replaced (returns a new dict)."""
        obs = dict(obs)
        for key, cam in (("agentview_image", "agentview"), ("robot0_eye_in_hand_image", "robot0_eye_in_hand")):
            obs[key] = np.where(self.mask(RES, cam)[..., None], self._photo(RES), obs[key])
        return obs

    def render(self, res, cam="agentview"):
        """Display-only high-res composite (policy inputs are unaffected)."""
        rgb = self.env.env.sim.render(res, res, camera_name=cam)
        return np.ascontiguousarray(np.where(self.mask(res, cam)[..., None], self._photo(res), rgb)[::-1, ::-1])


def orbit_camera(env, degrees, cam="agentview"):
    """Rotate the third-person camera about the vertical axis through the workspace centre (world origin), keeping it
    aimed inward. Call after every env.reset() (reset rebuilds the model)."""
    m = env.env.sim.model; cid = m.camera_name2id(cam); th = np.deg2rad(degrees)
    x, y, z = m.cam_pos[cid]
    m.cam_pos[cid] = [np.cos(th) * x - np.sin(th) * y, np.sin(th) * x + np.cos(th) * y, z]
    h = th / 2; w1, x1, y1, z1 = np.cos(h), 0, 0, np.sin(h); w2, x2, y2, z2 = m.cam_quat[cid]
    m.cam_quat[cid] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                       w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    env.env.sim.forward()


def render_depth(sim, cam, res=RES):
    rgb, d = sim.render(res, res, camera_name=cam, depth=True)
    m = sim.model; near, far = m.vis.map.znear * m.stat.extent, m.vis.map.zfar * m.stat.extent
    cid = m.camera_name2id(cam)
    frame = (sim.data.cam_xpos[cid].copy(), sim.data.cam_xmat[cid].reshape(3, 3).copy(), np.deg2rad(m.cam_fovy[cid]))
    return rgb, near / (1.0 - d * (1.0 - near / far)), frame


def token_correspondence(depth0, frame0, depth1, frame1, grid=16, res=RES, tol=0.03):
    """For each nominal policy-image token centre: lift to 3D with depth, project into the other view.
    Returns (grid*grid, 2) sub-token (row, col) in the other view's token grid, or (-1, -1) if out of view/occluded."""
    (p0, R0, fov), (p1, R1, _) = frame0, frame1
    fpx = (res / 2) / np.tan(fov / 2); c = (res - 1) / 2; tok = POLICY_RES / grid
    out = -np.ones((grid * grid, 2))
    for i in range(grid * grid):
        r, q = divmod(i, grid)
        u, v = res - 1 - (q + 0.5) * tok * res / POLICY_RES, res - 1 - (r + 0.5) * tok * res / POLICY_RES   # raw frame
        z = depth0[int(round(v)), int(round(u))]
        P = p0 + R0 @ np.array([(u - c) / fpx * z, (v - c) / fpx * z, -z])
        pc = R1.T @ (P - p1); z1 = -pc[2]
        if z1 <= 0:
            continue
        u1, v1 = c + fpx * pc[0] / z1, c + fpx * pc[1] / z1
        if not (0 <= u1 < res and 0 <= v1 < res) or abs(depth1[int(v1), int(u1)] - z1) > tol:
            continue
        out[i] = ((res - 1 - v1) * POLICY_RES / res / tok - 0.5, (res - 1 - u1) * POLICY_RES / res / tok - 0.5)
    return out


def to_policy_image(raw_rgb):
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(raw_rgb[::-1, ::-1]), POLICY_RES, POLICY_RES))
