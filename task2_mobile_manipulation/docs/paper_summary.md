# Paper summaries — Mobile Manipulation

## Paper 1 — MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation (CVPR 2025)
Wu, Zhou, Xu, Wang, Yan. arXiv:2503.13446. The [official project page](https://gary3410.github.io/momanipVLA/) lists its code link as "Coming Soon" (checked during final assembly). This submission uses its own reduced implementation.

**Problem.** Pre-trained VLA models such as OpenVLA, RT-2 and π0 generalize across tasks and scenes, but they are trained on fixed-base arms. Mobile-manipulation policies trained end-to-end by imitation need many expensive whole-body demonstrations, and modular pipelines (navigate, then manipulate) compound errors. The paper asks whether a fixed-base VLA can be reused on a mobile manipulator with almost no new training.

**Key idea.** Keep the VLA as a *waypoint generator*. At each step it predicts the next end-effector waypoint Q_i in the base frame, and Eq. 1 maps it to the world frame (Q̂_i = Γ(P_i^0, Q_i)). A motion planner then decides how the **base and the arm jointly** reach the next waypoint, so the base moves exactly when the waypoint would otherwise be infeasible.

* **Motion-planning objective** (Eq. 2–6), defined over T interpolated whole-body poses between consecutive waypoints:
  * reachability F_r = N_IK/N_max if IK converges, otherwise a large constant C0. More IK iterations mean the arm is closer to its joint limits.
  * smoothness F_s = Σ(‖θ^{t+1}−θ^t‖₂ + ‖x_b^{t+1}−x_b^t‖₂)
  * collision F_c = ΣΣ max(0, ε0 − D(q_j^t)), a hinge on the ESDF distance of robot query points (nvblox)
  * O = λ1 F_r + λ2 F_s + λ3 F_c with λ = (10, 1, 0.6) and ε0 = 0.1.
* **Bi-level optimization** (Alg. 1). The 10-DoF search (7 arm + 3 base) is split into two levels. The *upper* level runs Dual Annealing with SLSQP refinement over base poses. It scores each base candidate by the quality of the arm search space it induces: J_up = Σ_M O + α Σ_top-k O (Eq. 7). The *lower* level then picks the best end-effector trajectory inside that space.

**Contributions / results.** On the OVMM benchmark (Stretch robot, 60 scenes, "move object from A to B"), MoManipVLA gets 15.8 % overall SR. That is 4.2 points above the best baseline (KUZHUM, 11.6 %), and its Pick SR is 62.6 %. Every cost term helps; reachability matters most (Table 2). Bi-level search beats direct search on both SR and latency (693 ms vs 743 ms). In the real world (hexmove base + RM65 arm) it reaches 30–40 % SR with only 50 fine-tuning demos. Most failures are navigation failures, with "orient to place" at 72 %.

**Stated limitations.** Performance is bounded by the VLA, deployment needs a constrained motion space, and long-horizon tasks lack a task planner.

## Paper 2 — HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations (RSS 2026)
Xu, Park, Zhang, Cousineau, Bhat, Barreiros, Wang, Bohg, Song. Code: github.com/xxm19/hommi. Data: huggingface.co/datasets/real-stanford/hommi.

**Problem.** Teleoperated demonstrations for bimanual mobile manipulators are slow and costly. Handheld interfaces such as UMI scale well, but their wrist cameras miss the global context needed for navigation, bimanual coordination and *active perception*. Adding a head camera widens the **human-to-robot embodiment gap** in two ways:
* a *visual* gap: different arms, height and viewpoint;
* a *kinematic* gap: human neck and torso vs. a 2-DoF robot neck, plus different height.

**Key ideas.**
1. **Data interface.** Bimanual UMI grippers plus a cap-mounted iPhone. ARKit multi-device collaboration gives synchronized RGB, depth, 6-DoF pose and gripper width at 60 Hz in one world frame. No robot is needed.
2. **Embodiment-agnostic 3D visual representation.** The head RGB-D frame is lifted to a point map. DINOv3 patch features are fused with sinusoidal 3D encodings, and arm points are masked out by the gripper-frame test z < 0. Attention pooling produces the head embedding, combined with wrist DINOv3 CLS features and proprioception.
3. **Relaxed head action.** Instead of a 6-DoF head pose, the policy predicts a **3D look-at point**, the place where the center camera ray hits the scene. The controller turns it into a feasible gaze.
4. **Gripper-centric frame.** All observations and actions are expressed in the left-gripper frame.
5. **Constraint-aware whole-body controller.** This is a Mink differential-IK QP with high-weight bimanual SE(3) tracking, nominal and current posture regularization, CoM-over-base, an upright torso and self-collision limits. It runs at 100 Hz between the 10 Hz policy and the 500 Hz robot loop, with asynchronous latency-matched policy inference.

**Results.** On an RB-Y1 (two 7-DoF arms, 6-DoF torso, 2-DoF neck, holonomic base), HoMMI reaches Laundry 90 %, Delivery 85 % and Tablescape 80 % success. The baselines score as follows:
* Wrist-Only (UMI): 20 / 15 / 0 %
* RGB-Only (UMI + ego RGB + 6-DoF head action): 0 / 45 / 0 %
* Head-Only: 0 / 5 / 0 %
* w/o active neck: 75 / 55 / 55 %

Findings F1–F6 attribute the gains to egocentric context, the 3D representation, look-at actions and active neck control. HoMMI also transfers across demonstrators of different heights (167 / 182 cm).

**Stated limitations.** Short observation history (no memory), vision-only sensing (no force or touch), and a remaining hardware embodiment gap.

## Differences between the two papers

| | MoManipVLA | HoMMI |
|---|---|---|
| Where generalization comes from | Pre-trained fixed-base **VLA** (internet-scale robot data) | Scalable **robot-free human demonstrations** (UMI + egocentric) |
| Training needed | Small fine-tune of the VLA (200 sim / 50 real demos); the planner is training-free | Full visuomotor diffusion policy (DiT) trained per task (115–200 demos) |
| Action / output | End-effector waypoints → **optimization-based** whole-body planner | Gripper SE(3) trajectories + head look-at point → **QP differential IK** |
| How base/arm coordination is produced | Bi-level search (Dual Annealing + SLSQP) over base poses, scored by reachability / smoothness / collision | Base, torso and arms solved jointly in one weighted QP every 10 ms (no search) |
| Perception | Fixed camera RGB to the VLA; ESDF (nvblox) for collision | Wrist RGB + head 3D point-map tokens (DINOv3), arm masking |
| Active perception | No | Yes (2-DoF neck driven by look-at points) |
| Embodiment | Single arm (Stretch / RM65) | Bimanual + torso + neck (RB-Y1) |
| Closed loop | Waypoint by waypoint, re-planned | 10 Hz policy, asynchronous, latency-matched |
| Latency | ~0.6–0.7 s per planning step | 100 Hz IK, policy chunks every 0.1 s |
| Main failure mode reported | Navigation / orientation before manipulation (72 %) | Weak grasps, misalignment after long navigation |

Both papers start from the same observation: whole-body coordination should be produced by a controller or planner that knows the robot's constraints, *not* learned directly in joint space. MoManipVLA does this at **plan time**: it searches for base poses that make the VLA's end-effector targets feasible. HoMMI does it at **control time**: it relaxes the head action (look-at) and lets a constrained QP distribute end-effector motion over base, torso and arms. Our experiments test each mechanism on its own: bi-level search vs. fixed-base and heuristic base placement, and look-at vs. 6-DoF head tracking.

## Reproduction deviation checked during final assembly

Paper Eq. 4 uses unsquared Euclidean norms. The recorded implementation in `momanipvla_repro/planner.py` uses squared joint/base increments. The reported CSVs come from that squared variant and have not been relabelled as an exact Eq. 4 reproduction. DINOv3 plus a learned two-instruction embedding is a small waypoint surrogate, rather than OpenVLA or an open-vocabulary language model.
