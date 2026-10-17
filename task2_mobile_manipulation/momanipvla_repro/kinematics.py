"""Kinematics for the mobile Panda: FK, iterative IK with iteration count, collision query points.

MoManipVLA solves arm joint angles "iteratively by the Pinocchio IK solver" and uses the number of
IK iterations N_IK as the reachability signal (Eq. 3). We use the same closed-loop IK algorithm as
Pinocchio's CLIK example (damped least squares on the SE(3) log error, joint-limit clamping),
implemented on MuJoCo kinematics so the planner and the renderer share one model.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scene import ARM_HOME, RobotIndex


def yaw_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def base_to_world(base, p_b, R_b):
    """Γ: pose in the base frame -> world frame (base = [x, y, yaw])."""
    Rz = yaw_rot(base[2])
    return Rz @ p_b + np.array([base[0], base[1], 0.0]), Rz @ R_b


def world_to_base(base, p_w, R_w):
    Rz = yaw_rot(base[2])
    return Rz.T @ (p_w - np.array([base[0], base[1], 0.0])), Rz.T @ R_w


def rot_err(R_cur, R_tgt):
    """Rotation error vector (world frame) such that R_tgt = exp(e) R_cur (SO(3) log map)."""
    E = R_tgt @ R_cur.T
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    c = np.clip((E[0, 0] + E[1, 1] + E[2, 2] - 1.0) * 0.5, -1.0, 1.0)
    ang = np.arccos(c)
    if ang < 1e-6:
        return 0.5 * v
    if np.pi - ang < 1e-3:   # near pi: fall back to the exact (slow) path
        return Rotation.from_matrix(E).as_rotvec()
    return v * (ang / (2.0 * np.sin(ang)))


class Kinematics:
    def __init__(self, model: mujoco.MjModel, n_query_per_geom=14, seed=0):
        self.m = model
        self.d = mujoco.MjData(model)
        self.idx = RobotIndex(model)
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self._build_query_points(n_query_per_geom, np.random.default_rng(seed))

    # ---------------------------------------------------------------- state
    def set(self, base, q_arm, fingers=0.04):
        d = self.d
        d.qpos[self.idx.base_q] = base
        d.qpos[self.idx.arm_q] = q_arm
        d.qpos[self.idx.fingers_q] = fingers
        mujoco.mj_kinematics(self.m, d)

    def tcp(self):
        return self.d.site_xpos[self.idx.tcp].copy(), self.d.site_xmat[self.idx.tcp].reshape(3, 3).copy()

    def fk(self, base, q_arm):
        self.set(base, q_arm)
        return self.tcp()

    # ---------------------------------------------------------------- IK
    def ik(self, base, p_tgt, R_tgt, q_init=None, max_iter=100, tol_pos=4e-3, tol_rot=0.03, damping=1e-2,
           max_step=0.25):
        """Arm IK for a fixed base pose. Returns (q, n_iter, converged, pos_err, rot_err)."""
        q = (ARM_HOME if q_init is None else q_init).copy()
        lo, hi = self.idx.arm_lo, self.idx.arm_hi
        m, d = self.m, self.d
        d.qpos[self.idx.base_q] = base
        cols = self.idx.arm_v
        damp = damping * np.eye(6)
        for it in range(1, max_iter + 1):
            d.qpos[self.idx.arm_q] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.idx.tcp]
            R = d.site_xmat[self.idx.tcp].reshape(3, 3)
            e_p = p_tgt - p
            e_r = rot_err(R, R_tgt)
            ep, er = np.sqrt(e_p @ e_p), np.sqrt(e_r @ e_r)
            if ep < tol_pos and er < tol_rot:
                return q, it, True, ep, er
            mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.idx.tcp)
            J = np.vstack([self.jacp[:, cols], self.jacr[:, cols]])
            e = np.concatenate([e_p, 0.5 * e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damp, e)
            n = np.sqrt(dq @ dq)
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, lo, hi)
        return q, max_iter, False, ep, er

    # ---------------------------------------------------------------- collision query points
    def _build_query_points(self, n_per_geom, rng):
        """Sample points on the robot collision geometry (mesh vertices / box surface), in geom frames."""
        m = self.m
        self.q_geom, self.q_local = [], []
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b not in self.idx.robot_bodies:
                continue
            name = m.geom(g).name
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 3:   # panda collision meshes
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                pts = v[rng.choice(len(v), size=min(n_per_geom, len(v)), replace=False)]
            elif name == "base_box":
                # side faces of the base box on a ~5 cm grid
                h = m.geom_size[g]
                pts = []
                for z in np.linspace(-h[2], h[2], 5):
                    for x in np.linspace(-h[0], h[0], 13):
                        pts += [[x, -h[1], z], [x, h[1], z]]
                    for y in np.linspace(-h[1], h[1], 11)[1:-1]:
                        pts += [[-h[0], y, z], [h[0], y, z]]
                pts = np.array(pts, float)
            else:
                continue
            self.q_geom += [g] * len(pts)
            self.q_local.append(pts)
        self.q_geom = np.array(self.q_geom)
        self.q_local = np.concatenate(self.q_local)
        names = [m.body(m.geom_bodyid[g]).name for g in self.q_geom]
        self.q_is_gripper = np.array([("finger" in n) or n.endswith("hand") for n in names])

    def query_points(self):
        """World coordinates of robot query points for the current state (after set/ik)."""
        d = self.d
        R = d.geom_xmat[self.q_geom].reshape(-1, 3, 3)
        return np.einsum("nij,nj->ni", R, self.q_local) + d.geom_xpos[self.q_geom]
