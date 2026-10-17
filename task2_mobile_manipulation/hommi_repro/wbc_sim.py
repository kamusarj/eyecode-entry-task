"""Kinematic replay of human demonstrations through HoMMI's official whole-body controller.

This harness drives the *official* RB-Y1 whole-body IK (`rby1.whole_body_ik.RBY1WholeBodyIK`,
Mink + DAQP QP) and the *official* look-at head controller (`RBY1WBC._compute_head_lookat_angles`)
from `deps/rby1-wbc`, without the robot SDK. The realtime driver is replaced by perfect
kinematic tracking: the IK solution of one tick becomes the measured state of the next tick.

Head control modes (same names as `head_control.mode` in `deps/rby1-wbc/config/wbc.yaml`):
  * ``direct_lookat`` – HoMMI: head excluded from IK, yaw/pitch computed from the 3D look-at point.
  * ``ik``            – RGB-Only (UMI+Ego) baseline action space: 6-DoF head pose tracked inside the IK.
  * ``hold_init``     – w/o Active Neck: head joints frozen at the initial configuration.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
WBC_ROOT = REPO_ROOT / "third_party" / "hommi" / "deps" / "rby1-wbc"
if str(WBC_ROOT) not in sys.path:
    sys.path.insert(0, str(WBC_ROOT))

# `control/__init__.py` imports the compiled realtime-driver bindings (needs the RB-Y1 SDK).
# Only pure-python helpers are used here, so register an empty stand-in for the binary module.
if "control.rby1_controller" not in sys.modules:
    _stub = types.ModuleType("control.rby1_controller")
    for _name in ("Config", "RealtimeDriver", "RobotSnapshot", "AdmittanceController",
                  "AdmittanceControllerConfig", "debug_echo"):
        setattr(_stub, _name, object)
    sys.modules["control.rby1_controller"] = _stub

import yaml  # noqa: E402
import rby1.ee_targets as ee_targets_mod  # noqa: E402
from control.rby1_wbc import RBY1WBC  # noqa: E402
from rby1.ee_targets import EETargets  # noqa: E402
from rby1.frame_transforms import MODEL_TO_TCP_FRAME, TCP_TO_MODEL_FRAME  # noqa: E402
from rby1.whole_body_ik import RBY1WholeBodyIK  # noqa: E402

DEMO_PLAN = WBC_ROOT / "demo" / "dataset_plan.pkl"
HEAD_MODES = ("direct_lookat", "ik", "hold_init")


class SimClock:
    """Replaces time.monotonic() inside rby1.ee_targets so interpolation follows simulated time."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def pose6_to_mat(pose):
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_rotvec(pose[3:6]).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def mat_to_pos_quat_wxyz(mat):
    q = Rotation.from_matrix(mat[:3, :3]).as_quat()  # xyzw
    return mat[:3, 3].copy(), np.array([q[3], q[0], q[1], q[2]])


def rot_angle_deg(r_a, r_b):
    cos = (np.trace(r_a.T @ r_b) - 1.0) / 2.0
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


@dataclass
class Episode:
    name: str
    left: np.ndarray          # (T,4,4) TCP poses in demo world frame
    right: np.ndarray
    head: np.ndarray          # (T,4,4) head camera pose (OpenCV convention: z optical axis)
    lookat: np.ndarray | None = None   # (T,3) look-at points in demo world frame (if available)
    fps: float = 60.0
    up_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))


def load_plan_episodes(path=DEMO_PLAN):
    with open(path, "rb") as f:
        plan = pickle.load(f)
    eps = []
    for ep in plan:
        g = {k: ep[f"grippers_{k}"][0]["tcp_pose"] for k in ("left", "right", "head")}
        eps.append(Episode(
            name=ep["episode_name"],
            left=np.stack([pose6_to_mat(p) for p in g["left"]]),
            right=np.stack([pose6_to_mat(p) for p in g["right"]]),
            head=np.stack([pose6_to_mat(p) for p in g["head"]]),
        ))
    return eps


class WBCReplay:
    """Replays one episode through the official IK with a chosen head-control mode."""

    def __init__(self, wbc_cfg=WBC_ROOT / "config" / "wbc.yaml"):
        self.ik = RBY1WholeBodyIK()
        self.model, self.data = self.ik.model, self.ik.data
        cfg = yaml.safe_load(open(wbc_cfg))
        self.init_position = cfg["init_position"]
        self.ik_hz = float(cfg["ik_frequency_hz"])            # 100 Hz
        self.cmd_hz = float(cfg["trajectory_frequency_hz"])   # 10 Hz policy/command rate
        # light-weight object exposing the attributes RBY1WBC._compute_head_lookat_angles reads
        self._lookat_ctx = types.SimpleNamespace(
            model=self.model, data=mujoco.MjData(self.model),
            _head_ee_site_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "head"),
            _head_parent_body_id=int(self.model.body_parentid[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_head_1")]),
            _head_joint_ids={n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                             for n in ("head_0", "head_1")},
            _model_lock=_NullLock(),
        )
        self.site = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
                     for n in ("end_effector_l", "end_effector_r", "head")}
        self.jnt_lo = self.model.jnt_range[:, 0]
        self.jnt_hi = self.model.jnt_range[:, 1]

    # ------------------------------------------------------------------ setup
    def initial_qpos(self):
        q = self.model.qpos0.copy()
        q[:7] = [0, 0, 0, 1, 0, 0, 0]
        ik = self.ik
        for idxs, key in ((ik.torso_qpos_indices, "torso"), (ik.left_arm_qpos_indices, "left_arm"),
                          (ik.right_arm_qpos_indices, "right_arm"), (ik.head_qpos_indices, "head")):
            q[idxs] = self.init_position[key]
        return q

    def fk(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        out = {}
        for name, key in (("end_effector_l", "left"), ("end_effector_r", "right"), ("head", "head")):
            sid = self.site[name]
            m = np.eye(4)
            m[:3, :3] = self.data.site_xmat[sid].reshape(3, 3)
            m[:3, 3] = self.data.site_xpos[sid]
            out[key] = m @ MODEL_TO_TCP_FRAME[key]  # model site -> TCP / camera frame
        return out

    def alignment(self, ep: Episode, q0, mode="relative"):
        """Per-body transforms demo-world -> robot-world.

        ``relative`` (official `demo/trajectory_loader.py` semantics): every body m gets
            T_m = robot_pose0_m @ demo_pose0_m^-1,
        i.e. each effector replays the demonstrator's *motion* starting from the robot's own pose.
        ``gripper_centric``: hands as in ``relative``; head pose and look-at point are mapped with the
        left-gripper transform, as HoMMI does at deployment (``reference_frame: left``), so the
        demonstrator's head pose *relative to the hands* is commanded (keeps the embodiment gap).
        ``anchored`` (HoMMI deployment with ``reference_frame: left``): one gravity-aligned transform
        that maps the demonstrator's left gripper onto the robot's left gripper is applied to all
        bodies, so the human hand-head geometry (height / viewpoint gap) is kept.
        Returns dict body -> 4x4 transform (the look-at point uses the head transform)."""
        tcp0 = self.fk(q0)
        rel = {k: tcp0[k] @ np.linalg.inv(getattr(ep, k)[0]) for k in ("left", "right", "head")}
        if mode == "relative":
            return rel
        if mode == "gripper_centric":
            return {"left": rel["left"], "right": rel["right"], "head": rel["left"]}
        up = ep.up_axis / np.linalg.norm(ep.up_axis)
        fwd = ep.head[0][:3, 2] - np.dot(ep.head[0][:3, 2], up) * up
        fwd /= np.linalg.norm(fwd)
        m = np.stack([fwd, np.cross(up, fwd), up])  # rows: robot x,y,z expressed in demo coords
        T = np.eye(4)
        T[:3, :3] = m
        T[:3, 3] = tcp0["left"][:3, 3] - m @ ep.left[0][:3, 3]
        return {k: T for k in ("left", "right", "head")}

    @staticmethod
    def planar_lookat(head_tcp, table_z, fallback_dist=1.0):
        """Look-at proxy when no depth is available: intersect the optical axis with the work plane."""
        o, d = head_tcp[:3, 3], head_tcp[:3, 2]
        if d[2] < -1e-3:
            s = (table_z - o[2]) / d[2]
            if 0.1 < s < 3.0:
                return o + s * d
        return o + fallback_dist * d

    # ------------------------------------------------------------------ replay
    def run(self, ep: Episode, mode: str, align="relative", head_height_offset=0.0, settle_s=2.0,
            max_s=None, record_qpos=False):
        assert mode in HEAD_MODES
        clock = SimClock()
        ee_targets_mod.time = clock  # EETargets interpolates on time.monotonic()
        targets = EETargets()
        q = self.initial_qpos()
        T_align = self.alignment(ep, q, align)

        stride = int(round(ep.fps / self.cmd_hz))
        idx = np.arange(0, len(ep.left), stride)
        if max_s is not None:
            idx = idx[: int(max_s * self.cmd_hz)]
        L = T_align["left"] @ ep.left[idx]
        Rr = T_align["right"] @ ep.right[idx]
        H = T_align["head"] @ ep.head[idx]
        if ep.lookat is not None:
            Th = T_align["head"]
            look = (Th[:3, :3] @ ep.lookat[idx].T).T + Th[:3, 3]
        else:
            # look-at proxy: optical axis hits the work plane (just below the lowest gripper pose)
            table_z = min(L[:, 2, 3].min(), Rr[:, 2, 3].min()) - 0.02
            look = np.stack([self.planar_lookat(h, table_z) for h in H])
        # taller / shorter demonstrator (paper F6): the head moves, the looked-at scene point does not
        H[:, 2, 3] += head_height_offset

        def send(k, duration):
            lp, lq = mat_to_pos_quat_wxyz(L[k] @ TCP_TO_MODEL_FRAME["left"])
            rp, rq = mat_to_pos_quat_wxyz(Rr[k] @ TCP_TO_MODEL_FRAME["right"])
            hp, hq = mat_to_pos_quat_wxyz(H[k] @ TCP_TO_MODEL_FRAME["head"])
            targets.set_targets(lp, lq, rp, rq, head_pos=hp, head_quat=hq,
                                head_lookatpoint=look[k], duration=duration)

        dt = 1.0 / self.ik_hz
        ticks_per_cmd = int(round(self.ik_hz / self.cmd_hz))
        head_hold = np.asarray(self.init_position["head"], float)
        log = {k: [] for k in ("t", "pos_err_l", "pos_err_r", "rot_err_l", "rot_err_r", "gaze_err",
                               "head_pos_err", "head_rot_err", "near_limit", "base_xy", "base_yaw",
                               "torso", "success")}
        qlog, tgt_log = [], []

        def tick(k_cmd, measure):
            nonlocal q
            lp, lq, _, rp, rq, _, hp, hq = targets.get_for_ik(use_interpolation=True)
            look_now = targets.get_head_lookatpoint(use_interpolation=True)
            head_override = None
            hp_ik, hq_ik = hp, hq
            if mode == "hold_init":
                head_override, hp_ik, hq_ik = head_hold, None, None
            elif mode == "direct_lookat":
                head_override = RBY1WBC._compute_head_lookat_angles(self._lookat_ctx, q, look_now)
                hp_ik, hq_ik = None, None
            sol, _vel, ok, _ = self.ik.solve(left_target_pos=lp, left_target_quat=lq,
                                             right_target_pos=rp, right_target_quat=rq,
                                             head_target_pos=hp_ik, head_target_quat=hq_ik,
                                             current_qpos=q, dt=dt)
            if head_override is not None:
                sol[self.ik.head_qpos_indices] = head_override
            q = sol
            clock.t += dt
            if not measure:
                return
            tcp = self.fk(q)
            tgt_l = _pq_to_mat(lp, lq) @ MODEL_TO_TCP_FRAME["left"]
            tgt_r = _pq_to_mat(rp, rq) @ MODEL_TO_TCP_FRAME["right"]
            tgt_h = _pq_to_mat(hp, hq) @ MODEL_TO_TCP_FRAME["head"]
            log["t"].append(clock.t)
            log["pos_err_l"].append(np.linalg.norm(tcp["left"][:3, 3] - tgt_l[:3, 3]))
            log["pos_err_r"].append(np.linalg.norm(tcp["right"][:3, 3] - tgt_r[:3, 3]))
            log["rot_err_l"].append(rot_angle_deg(tcp["left"][:3, :3], tgt_l[:3, :3]))
            log["rot_err_r"].append(rot_angle_deg(tcp["right"][:3, :3], tgt_r[:3, :3]))
            v = look_now - tcp["head"][:3, 3]
            cos = np.dot(v / np.linalg.norm(v), tcp["head"][:3, 2])
            log["gaze_err"].append(math.degrees(math.acos(np.clip(cos, -1, 1))))
            log["head_pos_err"].append(np.linalg.norm(tcp["head"][:3, 3] - tgt_h[:3, 3]))
            log["head_rot_err"].append(rot_angle_deg(tcp["head"][:3, :3], tgt_h[:3, :3]))
            lim = self._near_limit(q)
            log["near_limit"].append(lim)
            log["base_xy"].append(q[:2].copy())
            log["base_yaw"].append(2 * math.atan2(q[6], q[3]))
            log["torso"].append(q[self.ik.torso_qpos_indices].copy())
            log["success"].append(ok)
            if record_qpos:
                qlog.append(q.copy())
                tgt_log.append(dict(left=tgt_l, right=tgt_r, head=tgt_h, look=look_now.copy()))

        # settle on the first command (robot moves to the demo start pose), not measured
        send(0, 0.0)
        for _ in range(int(settle_s * self.ik_hz)):
            tick(0, measure=False)
        for k in range(1, len(idx)):
            send(k, 1.0 / self.cmd_hz)
            for _ in range(ticks_per_cmd):
                tick(k, measure=True)

        out = {k: np.asarray(v) for k, v in log.items()}
        if record_qpos:
            out["qpos"] = np.asarray(qlog)
            out["targets"] = tgt_log
        return out

    def _near_limit(self, q, margin=math.radians(5.0)):
        """True if any limited hinge joint is within `margin` of its range (wbc.yaml warning margin)."""
        cnt = False
        for j in range(self.model.njnt):
            if not self.model.jnt_limited[j] or self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = self.model.joint(j).name
            if name.startswith(("wheel", "gripper")):
                continue
            v = q[self.model.jnt_qposadr[j]]
            if v < self.jnt_lo[j] + margin or v > self.jnt_hi[j] - margin:
                cnt = True
                break
        return cnt


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _pq_to_mat(p, q_wxyz):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    m[:3, 3] = p
    return m


def summarize(log):
    smooth = np.diff(np.asarray(log["torso"]), n=2, axis=0)
    return {
        "ee_pos_err_mm": 1000 * float(np.mean(np.r_[log["pos_err_l"], log["pos_err_r"]])),
        "ee_pos_err_p95_mm": 1000 * float(np.percentile(np.r_[log["pos_err_l"], log["pos_err_r"]], 95)),
        "ee_rot_err_deg": float(np.mean(np.r_[log["rot_err_l"], log["rot_err_r"]])),
        "gaze_err_deg": float(np.mean(log["gaze_err"])),
        "gaze_in_10deg": float(np.mean(np.asarray(log["gaze_err"]) < 10.0)),
        "head_pos_err_mm": 1000 * float(np.mean(log["head_pos_err"])),
        "head_rot_err_deg": float(np.mean(log["head_rot_err"])),
        "near_limit_frac": float(np.mean(log["near_limit"])),
        "qp_fail_frac": float(1.0 - np.mean(log["success"])),
        "torso_jerk": float(np.mean(np.abs(smooth))) if len(smooth) else 0.0,
    }


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    eps = load_plan_episodes()
    sim = WBCReplay()
    for mode in HEAD_MODES:
        log = sim.run(eps[0], mode, max_s=5.0)
        print(mode, {k: round(v, 3) for k, v in summarize(log).items()})
