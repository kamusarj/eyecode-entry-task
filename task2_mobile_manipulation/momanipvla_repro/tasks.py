"""OVMM-style pick-and-place episodes driven by VLA waypoints.

Task: "move the cube from the table to the bowl". As in MoManipVLA's OVMM protocol, navigation is
done by a heuristic navigator (with realistic stopping errors) and the evaluated policy generates
the manipulation trajectories at the end of each navigation phase.

Waypoint source ("VLA"): the paper uses OpenVLA-7B fine-tuned on 200 fixed-base demos. A 7B VLA does
not fit our 6 GB GPU, so waypoints come from either
  * OracleVLA  – ground-truth keyposes + Gaussian prediction noise (models VLA imprecision), or
  * a learned fixed-base waypoint policy (see vla_policy.py) queried with the robot's camera image.
Both predict the next end-effector waypoint Q_i in the *base* frame from the current observation;
Γ(P0_i, Q_i) maps it to the world frame (Eq. 1).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from esdf import ESDF, BoxSDF
from kinematics import Kinematics, base_to_world, world_to_base, yaw_rot
from planner import PlannerConfig, SegmentPlanner, wrap
from scene import ARM_HOME, ARM_MOUNT_X, build_model, random_scene

GRIPPER_OPEN, GRIPPER_CLOSED = 0.04, 0.0
NOMINAL_REACH = ARM_MOUNT_X + 0.45   # heuristic: base-center -> target horizontal distance


def topdown_R(yaw):
    """TCP orientation for a top-down grasp: approach (z) pointing down, fingers along yaw."""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)


@dataclass
class Waypoint:
    p: np.ndarray       # world position
    R: np.ndarray       # world orientation
    gripper: float
    name: str


class OracleVLA:
    """Ground-truth keyposes, predicted in the base frame with noise (σ_pos, σ_yaw)."""

    NAMES = ("pregrasp", "grasp", "lift", "preplace", "place", "retreat")

    def __init__(self, sigma_pos=0.008, sigma_yaw_deg=3.0, rng=None, seed=None):
        self.sp, self.sy = sigma_pos, np.radians(sigma_yaw_deg)
        # the prediction error of each waypoint is fixed per episode, so every method sees the same VLA
        self.seed = int((rng or np.random.default_rng(0)).integers(1 << 31)) if seed is None else seed

    def predict(self, name, base, scene_state):
        cube, cube_yaw, bowl = scene_state["cube"], scene_state["cube_yaw"], scene_state["bowl"]
        hand_yaw = base[2] + np.pi / 2                          # fingers across the base heading
        if name in ("pregrasp", "grasp", "lift"):
            # cube is 90° symmetric: choose the face pair closest to the comfortable hand yaw
            k = np.round(wrap(hand_yaw - cube_yaw) / (np.pi / 2))
            yaw = cube_yaw + k * np.pi / 2
            dz = {"pregrasp": 0.12, "grasp": 0.005, "lift": 0.15}[name]
            p_w, R_w = cube + np.array([0, 0, dz]), topdown_R(yaw)
        else:
            dz = {"preplace": 0.15, "place": 0.06, "retreat": 0.18}[name]
            p_w, R_w = bowl + np.array([0, 0, dz]), topdown_R(hand_yaw)
        p_b, R_b = world_to_base(base, p_w, R_w)                # the VLA reasons in the base frame
        rng = np.random.default_rng([self.seed, self.NAMES.index(name)])
        p_b = p_b + rng.normal(0, self.sp, 3)
        R_b = yaw_rot(rng.normal(0, self.sy)) @ R_b
        return p_b, R_b


def nav_stop_pose(scene, target_xy, kin, esdf, rng, lateral_sigma=0.25, extra_standoff=(0.0, 0.45),
                  yaw_sigma_deg=20.0, receptacle=None):
    """Heuristic navigator: stop in front of the receptacle facing it, with OVMM-like errors."""
    f = next(x for x in scene.furniture if x.name == receptacle)
    n = np.array([np.cos(f.yaw), np.sin(f.yaw)])                # outward normal of the front edge
    t = np.array([-n[1], n[0]])
    front_off = np.dot(target_xy - f.center[:2], t)
    for _ in range(200):
        d = f.half[0] + 0.32 + rng.uniform(*extra_standoff)
        xy = f.center[:2] + n * d + t * (front_off + rng.normal(0, lateral_sigma))
        yaw = np.arctan2(-n[1], -n[0]) + np.radians(rng.normal(0, yaw_sigma_deg))
        base = np.array([xy[0], xy[1], yaw])
        kin.set(base, ARM_HOME)
        if esdf.query(kin.query_points()).min() > 0.05:
            return base
    raise RuntimeError("no collision-free navigation stop pose")


def heuristic_base(base0, target_p):
    """OVMM-style heuristic: face the target and stand at a nominal reach distance from it."""
    v = target_p[:2] - base0[:2]
    yaw = np.arctan2(v[1], v[0])
    xy = target_p[:2] - NOMINAL_REACH * np.array([np.cos(yaw), np.sin(yaw)])
    return np.array([xy[0], xy[1], yaw])


@dataclass
class EpisodeResult:
    seed: int
    method: str
    pick: bool = False
    place: bool = False
    steps: int = 0
    latency_ms: list = field(default_factory=list)
    fail_reason: str = ""
    base_travel: float = 0.0
    min_clearance: float = 1.0
    smooth: float = 0.0
    trajectory: list = field(default_factory=list)     # (base, q, fingers, cube_pose) for rendering

    @property
    def success(self):
        return self.pick and self.place


class Episode:
    def __init__(self, seed, vla_factory=None, cfg: PlannerConfig | None = None, exact_sdf=True):
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.scene = random_scene(rng)
        self.spec, self.model = build_model(self.scene)
        self.kin = Kinematics(self.model)
        self.esdf = (BoxSDF if exact_sdf else ESDF)(self.scene.furniture, room_half=self.scene.room_half)
        self.cfg = cfg or PlannerConfig()
        self.vla_factory = vla_factory or (lambda r: OracleVLA(seed=seed))
        # navigation stop poses are part of the episode definition (identical for all methods)
        nav_rng = np.random.default_rng(seed + 10_000)
        self.nav_pick = nav_stop_pose(self.scene, self.scene.cube_pos[:2], self.kin, self.esdf, nav_rng,
                                      receptacle=self.scene.pick_table)
        self.nav_place = nav_stop_pose(self.scene, self.scene.bowl_pos[:2], self.kin, self.esdf, nav_rng,
                                       receptacle=self.scene.place_table)

    def run(self, method, cfg: PlannerConfig | None = None, record=False, vla_seed=None, vla=None) -> EpisodeResult:
        cfg = cfg or self.cfg
        planner = SegmentPlanner(self.kin, self.esdf, cfg)
        rng = np.random.default_rng(self.seed if vla_seed is None else vla_seed)
        if vla is None:
            vla = self.vla_factory(np.random.default_rng(rng.integers(1 << 31)))
        res = EpisodeResult(self.seed, method)
        state = dict(cube=self.scene.cube_pos.copy(), cube_yaw=self.scene.cube_yaw, bowl=self.scene.bowl_pos.copy())
        base, q, fingers = self.nav_pick.copy(), ARM_HOME.copy(), GRIPPER_OPEN
        attached = None   # cube pose in the TCP frame once grasped

        def cube_pose():
            if attached is None:
                return state["cube"].copy()
            p, R = self.kin.fk(base, q)
            return p + R @ attached

        def log(b, qq, fg):
            if record:
                self.kin.set(b, qq)
                cp = cube_pose() if attached is None else self.kin.tcp()[0] + self.kin.tcp()[1] @ attached
                res.trajectory.append((b.copy(), qq.copy(), fg, cp))

        def teleport_base(new_base, reason):
            nonlocal base
            res.base_travel += float(np.linalg.norm(new_base[:2] - base[:2]))
            if record:
                for a in np.linspace(0, 1, 25)[1:]:
                    b = base + a * np.array([*(new_base[:2] - base[:2]), wrap(new_base[2] - base[2])])
                    log(b, q, fingers)
            base = new_base.copy()

        def observe():
            # current robot / object state, visible to learned VLAs that render the robot camera
            state.update(q=q.copy(), fingers=fingers, cube_now=cube_pose())

        def run_phase(names):
            nonlocal base, q, fingers, attached
            observe()
            if method == "heuristic":
                p_b, R_b = vla.predict(names[0], base, state)
                p_w, _ = base_to_world(base, p_b, R_b)
                teleport_base(heuristic_base(base, p_w), "heuristic")
            for name in names:
                observe()
                p_cur, R_cur = self.kin.fk(base, q)
                p_b, R_b = vla.predict(name, base, state)                    # Q_i (base frame)
                p_w, R_w = base_to_world(base, p_b, R_b)                     # Q̂_i = Γ(P0_i, Q_i)
                seg = planner.plan(method, base, q, (p_cur, R_cur), (p_w, R_w), rng)
                res.latency_ms.append(seg["latency_ms"])
                res.steps += seg["T"]
                res.min_clearance = min(res.min_clearance, float(seg["min_d"].min()))
                res.smooth += seg["Fs"]
                for t in range(1, len(seg["bases"])):
                    log(seg["bases"][t], seg["q"][t], fingers)
                res.base_travel += float(np.sum(np.linalg.norm(np.diff(seg["bases"][:, :2], axis=0), axis=1)))
                base, q = seg["bases"][-1].copy(), seg["q"][-1].copy()
                if not seg["success"]:
                    res.fail_reason = (f"{name}:collision({seg['collided']})" if seg["collision"]
                                       else f"{name}:unreachable")
                    return False
                if name == "grasp":
                    p, R = self.kin.fk(base, q)
                    err = R.T @ (state["cube"] - p)                          # cube center in TCP frame
                    if np.linalg.norm(err[:2]) > 0.018 or abs(err[2]) > 0.02:
                        res.fail_reason = "grasp:misaligned"
                        return False
                    fingers = GRIPPER_CLOSED
                    attached = err
                    log(base, q, fingers)
                if name == "place":
                    p, R = self.kin.fk(base, q)
                    cube_now = p + R @ attached
                    attached = None
                    state["cube"] = cube_now
                    fingers = GRIPPER_OPEN
                    log(base, q, fingers)
                    if np.linalg.norm(cube_now[:2] - state["bowl"][:2]) > 0.06:
                        res.fail_reason = "place:missed_bowl"
                        return False
            return True

        log(base, q, fingers)
        res.pick = run_phase(["pregrasp", "grasp", "lift"])
        if not res.pick:
            return res
        teleport_base(self.nav_place, "navigate")
        res.place = run_phase(["preplace", "place", "retreat"])
        return res


if __name__ == "__main__":
    import sys
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    for s in seeds:
        ep = Episode(s)
        for m in ("fixed", "heuristic", "bilevel", "direct"):
            t0 = time.time()
            r = ep.run(m)
            print(s, m, "pick", r.pick, "place", r.place, r.fail_reason, "steps", r.steps,
                  f"lat {np.mean(r.latency_ms):.0f}ms", f"wall {time.time() - t0:.1f}s")
