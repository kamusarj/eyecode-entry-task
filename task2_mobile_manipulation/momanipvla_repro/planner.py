"""MoManipVLA motion planning objectives (Sec. 3.3) and bi-level trajectory optimization (Sec. 3.4).

Between consecutive VLA waypoints Q̂_i -> Q̂_{i+1} (world frame) we plan T steps of whole-body
poses {x_b^t (base x,y,yaw), x_e^t (end-effector)} with

    O = λ1·F_r + λ2·F_s + λ3·F_c                                          (Eq. 6)
    F_r = N_IK/N_max if IK converges within N_max iterations else C0       (Eq. 3)
    F_s = Σ_t ||θ^{t+1}-θ^t||² + ||x_b^{t+1}-x_b^t||²                       (Eq. 4)
    F_c = Σ_t Σ_j max(0, ε0 - D(q_j^t, Ω))                                   (Eq. 5)

with λ = (10, 1, 0.6), ε0 = 0.1, step size 0.05 (paper Sec. 4.1).

Upper level: Dual Annealing (+ SLSQP local refinement) over the base end pose of the segment; each
candidate base path is scored by the quality of the arm search space it induces,
    J_up = Σ_{x_e∈M} O(x_b, x_e) + α Σ_{x_e∈N} O(x_b, x_e)                  (Eq. 7)
(M: sampled end-effector paths, N: their top-k). Lower level: pick the best end-effector path in
the search space of the optimal base path. "Direct" searches base and end-effector jointly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import dual_annealing
from scipy.spatial.transform import Rotation, Slerp

from kinematics import Kinematics


@dataclass
class CostWeights:
    reach: float = 10.0
    smooth: float = 1.0
    collision: float = 0.6


@dataclass
class PlannerConfig:
    weights: CostWeights = field(default_factory=CostWeights)
    eps0: float = 0.1            # collision safety margin ε0
    n_max: int = 100             # IK iteration budget N_max
    c0: float = 10.0             # cost of an unreachable pose
    step_pos: float = 0.05       # interpolation step (m), paper: 0.05
    step_rot: float = 0.15       # interpolation step (rad)
    max_steps: int = 14
    n_upper_samples: int = 4     # |M| arm-path samples per base candidate
    top_k: int = 1               # |N|
    alpha: float = 1.0
    n_lower_samples: int = 12    # arm-path candidates in the lower level
    path_sigma: float = 0.05     # std of mid-path end-effector detours (m)
    base_bound_xy: float = 0.6   # search radius of the base end pose around its initialization (m)
    base_bound_yaw: float = 0.8  # rad
    nominal_reach: float = 0.65  # heuristic initialization: base-center -> target distance (m)
    maxfun: int = 150            # dual annealing function budget (upper level)
    maxiter: int = 25
    collision_tol: float = 0.0   # executed pose is in collision if any query point has D < tol
    max_attempts: int = 3        # Algorithm 1 "repeat ... until" the waypoint is reached feasibly
    # Our fix for the soft hinge collision cost (not in the paper): among all evaluated candidates,
    # prefer the lowest-cost one that is *verified* feasible (all poses reachable, no penetration).
    select_feasible: bool = False


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SegmentPlanner:
    def __init__(self, kin: Kinematics, esdf, cfg: PlannerConfig, early_exit=True):
        self.kin, self.esdf, self.cfg = kin, esdf, cfg
        self.early_exit = early_exit

    # ------------------------------------------------------------------ helpers
    def n_steps(self, p0, R0, p1, R1):
        ang = np.linalg.norm(Rotation.from_matrix(R1 @ R0.T).as_rotvec())
        n = max(np.ceil(np.linalg.norm(p1 - p0) / self.cfg.step_pos), np.ceil(ang / self.cfg.step_rot))
        return int(np.clip(n, 2, self.cfg.max_steps))

    @staticmethod
    def base_path(b0, b1, T):
        a = np.linspace(0, 1, T + 1)[:, None]
        d = np.array([b1[0] - b0[0], b1[1] - b0[1], wrap(b1[2] - b0[2])])
        return b0 + a * d

    def ee_path(self, p0, R0, p1, R1, T, detour):
        """World-frame end-effector path: straight line + smooth detour, slerped orientation."""
        a = np.linspace(0, 1, T + 1)
        P = p0 + a[:, None] * (p1 - p0) + np.sin(np.pi * a)[:, None] * detour
        Rs = Slerp([0, 1], Rotation.from_matrix([R0, R1]))(a).as_matrix()
        return P, Rs

    def evaluate(self, bases, P, Rs, q0, weights=None):
        """Objective O summed over the planned steps t=1..T (t=0 is the current, fixed state)."""
        w = self.cfg.weights if weights is None else weights
        cfg, kin = self.cfg, self.kin
        q_prev, b_prev = q0, bases[0]
        Fr = Fs = Fc = 0.0
        qs, ok, min_d = [q0], [], []
        for t in range(1, len(bases)):
            q, n_it, conv, _, _ = kin.ik(bases[t], P[t], Rs[t], q_init=q_prev, max_iter=cfg.n_max)
            if not conv and self.early_exit:
                # the IK seed chain is broken: count this and all remaining poses as unreachable
                n_left = len(bases) - t
                Fr += cfg.c0 * n_left
                qs += [q] * n_left
                ok += [False] * n_left
                min_d += [float(self.esdf.query(kin.query_points()).min())] * n_left
                break
            Fr += n_it / cfg.n_max if conv else cfg.c0
            db = bases[t] - b_prev
            db[2] = wrap(db[2])
            Fs += float(np.sum((q - q_prev) ** 2) + np.sum(db ** 2))
            D = self.esdf.query(kin.query_points())          # kin state was left at the IK solution
            Fc += float(np.sum(np.maximum(0.0, cfg.eps0 - D)))
            qs.append(q)
            ok.append(conv)
            min_d.append(float(D.min()))
            q_prev, b_prev = q, bases[t]
        cost = w.reach * Fr + w.smooth * Fs + w.collision * Fc
        return cost, dict(q=np.array(qs), reach_ok=np.array(ok), min_d=np.array(min_d), Fr=Fr, Fs=Fs, Fc=Fc)

    # ------------------------------------------------------------------ methods
    def plan(self, method, base0, q0, W0, W1, rng):
        """Plan one segment W0 -> W1 (each a (p, R) world pose). Returns a result dict."""
        t0 = time.perf_counter()
        (p0, R0), (p1, R1) = W0, W1
        T = self.n_steps(p0, R0, p1, R1)
        if method in ("fixed", "heuristic"):
            bases = np.repeat(base0[None], T + 1, 0)
            P, Rs = self.ee_path(p0, R0, p1, R1, T, np.zeros(3))
            cost, info = self.evaluate(bases, P, Rs, q0)
        elif method in ("bilevel", "direct"):
            solve = self._bilevel if method == "bilevel" else self._direct
            for attempt in range(self.cfg.max_attempts):
                bases, P, Rs, cost, info = solve(base0, q0, p0, R0, p1, R1, T, rng)
                # the planner can verify its own solution: all poses reachable and outside obstacles
                if np.all(info["reach_ok"]) and np.min(info["min_d"]) >= self.cfg.collision_tol:
                    break
            info["attempts"] = attempt + 1
        else:
            raise ValueError(method)
        info.update(bases=bases, P=P, Rs=Rs, cost=cost, T=T, latency_ms=1000 * (time.perf_counter() - t0))
        info["collision"] = bool(np.any(info["min_d"] < self.cfg.collision_tol))
        info["success"] = bool(np.all(info["reach_ok"]) and not info["collision"])
        info["collided"] = ""
        if info["collision"]:
            t = int(np.argmin(info["min_d"])) + 1
            self.kin.set(bases[t], info["q"][t])
            D = self.esdf.query(self.kin.query_points())
            m = self.kin.m
            info["collided"] = "+".join(sorted({m.body(m.geom_bodyid[g]).name.replace("panda/", "")
                                                for g in self.kin.q_geom[D < self.cfg.collision_tol]}))
        return info

    def _bounds(self, center):
        c = self.cfg
        return [(center[0] - c.base_bound_xy, center[0] + c.base_bound_xy),
                (center[1] - c.base_bound_xy, center[1] + c.base_bound_xy),
                (center[2] - c.base_bound_yaw, center[2] + c.base_bound_yaw)]

    def _init_base(self, base0, q0, p1, R1):
        """Paper (Sec. 4.4): initialize the base optimization with heuristic navigation. Keep the current
        base if the next waypoint is already reachable, else face the waypoint at a nominal reach."""
        if self.kin.ik(base0, p1, R1, q_init=q0, max_iter=self.cfg.n_max)[2]:
            return base0.copy()
        v = p1[:2] - base0[:2]
        yaw = np.arctan2(v[1], v[0])
        xy = p1[:2] - self.cfg.nominal_reach * np.array([np.cos(yaw), np.sin(yaw)])
        return np.array([xy[0], xy[1], base0[2] + wrap(yaw - base0[2])])

    def _bilevel(self, base0, q0, p0, R0, p1, R1, T, rng):
        c = self.cfg
        # common random numbers: the same arm-path samples are used for every base candidate
        det_up = np.vstack([np.zeros(3), rng.normal(0, c.path_sigma, (c.n_upper_samples - 1, 3))])
        det_low = np.vstack([np.zeros(3), rng.normal(0, c.path_sigma, (c.n_lower_samples - 1, 3))])
        paths_up = [self.ee_path(p0, R0, p1, R1, T, d) for d in det_up]

        feasible_seen = []                                            # (J_up, u) of verified-feasible bases

        def j_up(u):                                                  # Eq. 7
            bases = self.base_path(base0, u, T)
            evals = [self.evaluate(bases, P, Rs, q0) for P, Rs in paths_up]
            costs = np.sort([e[0] for e in evals])
            j = float(costs.sum() + c.alpha * costs[: c.top_k].sum())
            if any(np.all(i["reach_ok"]) and i["min_d"].min() >= c.collision_tol for _, i in evals):
                feasible_seen.append((j, np.array(u, float)))
            return j

        x0 = self._init_base(base0, q0, p1, R1)
        res = dual_annealing(j_up, self._bounds(x0), x0=x0, maxiter=c.maxiter, maxfun=c.maxfun,
                             seed=int(rng.integers(1 << 31)), minimizer_kwargs=dict(method="SLSQP"))
        u_best = res.x
        if c.select_feasible and feasible_seen:
            u_best = min(feasible_seen, key=lambda x: x[0])[1]
        bases = self.base_path(base0, u_best, T)
        best = None
        for d in det_low:                                              # lower level: select x_e
            P, Rs = self.ee_path(p0, R0, p1, R1, T, d)
            cost, info = self.evaluate(bases, P, Rs, q0)
            key = cost
            if c.select_feasible and not (np.all(info["reach_ok"]) and info["min_d"].min() >= c.collision_tol):
                key = cost + 1e6
            if best is None or key < best[5]:
                best = (bases, P, Rs, cost, info, key)
        best = best[:5]
        best[4]["n_evals"] = res.nfev
        return best

    def _direct(self, base0, q0, p0, R0, p1, R1, T, rng):
        c = self.cfg
        s = 2.5 * c.path_sigma
        xb0 = self._init_base(base0, q0, p1, R1)
        bounds = self._bounds(xb0) + [(-s, s)] * 3

        feasible_seen = []

        def f(x):
            bases = self.base_path(base0, x[:3], T)
            P, Rs = self.ee_path(p0, R0, p1, R1, T, x[3:])
            cost, info = self.evaluate(bases, P, Rs, q0)
            if np.all(info["reach_ok"]) and info["min_d"].min() >= c.collision_tol:
                feasible_seen.append((cost, np.array(x, float)))
            return cost

        x0 = np.concatenate([xb0, np.zeros(3)])
        # same compute budget as the bi-level search (each upper evaluation scores |M| arm paths)
        res = dual_annealing(f, bounds, x0=x0, maxiter=c.maxiter, maxfun=c.maxfun * c.n_upper_samples,
                             seed=int(rng.integers(1 << 31)), minimizer_kwargs=dict(method="SLSQP"))
        x_best = res.x
        if c.select_feasible and feasible_seen:
            x_best = min(feasible_seen, key=lambda x: x[0])[1]
        bases = self.base_path(base0, x_best[:3], T)
        P, Rs = self.ee_path(p0, R0, p1, R1, T, x_best[3:])
        cost, info = self.evaluate(bases, P, Rs, q0)
        info["n_evals"] = res.nfev
        return bases, P, Rs, cost, info
