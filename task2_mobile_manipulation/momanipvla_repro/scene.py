"""Procedural household scene + mobile manipulator for the MoManipVLA re-implementation.

Robot: Franka Panda (MuJoCo Menagerie) mounted on a holonomic base with planar joints
(base_x, base_y, base_yaw) — a 10-DoF mobile manipulator (7 arm + 3 base), the search space
size discussed in MoManipVLA Sec. 3.4.
Scene: OVMM-style "move object from receptacle A to receptacle B" layout: a pick table with a cube,
a place counter with a bowl, and clutter (cabinet, shelf board, stool) that the base / arm can hit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_XML = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

BASE_HALF = np.array([0.30, 0.25, 0.21])   # base box half extents (m)
BASE_TOP = 0.42                            # height of the arm mount
ARM_MOUNT_X = 0.20                         # arm mount forward offset from base center
CUBE_HALF = 0.02
ARM_HOME = np.array([0.0, -0.6, 0.0, -2.3, 0.0, 1.75, 0.785])   # "ready" arm configuration


@dataclass
class Furniture:
    name: str
    center: np.ndarray        # box center (world)
    half: np.ndarray          # half extents
    yaw: float = 0.0
    rgba: tuple = (0.6, 0.45, 0.3, 1.0)
    collidable: bool = True   # enters the ESDF


@dataclass
class SceneSpec:
    furniture: list[Furniture] = field(default_factory=list)
    cube_pos: np.ndarray = None
    cube_yaw: float = 0.0
    bowl_pos: np.ndarray = None
    pick_table: str = "table_a"
    place_table: str = "counter_b"
    room_half: float = 3.5


def _yaw_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def random_scene(rng: np.random.Generator) -> SceneSpec:
    """Pick table on one wall, place counter on the opposite wall, clutter next to both."""
    spec = SceneSpec()
    walls = rng.permutation(4)
    wall_yaw = {0: 0.0, 1: np.pi / 2, 2: np.pi, 3: -np.pi / 2}   # inward facing direction of a wall

    def wall_box(wall, along, depth_half, length_half, height, name, rgba):
        # box against wall `wall`; its front edge faces the room center
        n = np.array([np.cos(wall_yaw[wall]), np.sin(wall_yaw[wall])])        # inward normal
        t = np.array([-n[1], n[0]])                                           # along-wall tangent
        c2 = -n * (spec.room_half - depth_half) + t * along
        return Furniture(name, np.array([c2[0], c2[1], height / 2]),
                         np.array([depth_half, length_half, height / 2]), yaw=wall_yaw[wall], rgba=rgba)

    a_wall, b_wall = walls[0], walls[1] if (walls[1] - walls[0]) % 2 == 0 else walls[2]
    if (b_wall - a_wall) % 4 != 2:
        b_wall = (a_wall + 2) % 4
    along_a = rng.uniform(-1.0, 1.0)
    along_b = rng.uniform(-1.0, 1.0)
    table = wall_box(a_wall, along_a, rng.uniform(0.30, 0.40), rng.uniform(0.55, 0.75),
                     rng.uniform(0.70, 0.78), "table_a", (0.55, 0.38, 0.22, 1))
    counter = wall_box(b_wall, along_b, rng.uniform(0.28, 0.35), rng.uniform(0.5, 0.7),
                       rng.uniform(0.82, 0.92), "counter_b", (0.35, 0.35, 0.42, 1))
    spec.furniture += [table, counter]

    # clutter: tall cabinet beside the table, a wall shelf board above the counter, a stool
    side = rng.choice([-1, 1])
    cab = wall_box(a_wall, along_a + side * (table.half[1] + 0.30), 0.25, 0.25, 1.6, "cabinet", (0.8, 0.8, 0.75, 1))
    shelf_h = counter.half[2] * 2 + rng.uniform(0.38, 0.55)
    shelf = wall_box(b_wall, along_b, 0.14, counter.half[1], 0.03, "shelf", (0.75, 0.6, 0.4, 1))
    shelf.center[2] = shelf_h
    spec.furniture += [cab, shelf]
    n_a = np.array([np.cos(wall_yaw[a_wall]), np.sin(wall_yaw[a_wall])])
    t_a = np.array([-n_a[1], n_a[0]])
    stool_xy = table.center[:2] + n_a * (table.half[0] + rng.uniform(0.35, 0.6)) - t_a * side * rng.uniform(0.5, 0.8)
    spec.furniture.append(Furniture("stool", np.array([*stool_xy, 0.23]), np.array([0.18, 0.18, 0.23]),
                                    rgba=(0.3, 0.5, 0.6, 1)))

    # cube on the table: depth measured from the front edge, lateral offset along the table
    depth = rng.uniform(0.06, 0.35)   # reachable from a collision-free base pose
    lateral = rng.uniform(-table.half[1] + 0.08, table.half[1] - 0.08)
    front = table.center[:2] + n_a * table.half[0]
    xy = front - n_a * depth + t_a * lateral
    spec.cube_pos = np.array([xy[0], xy[1], 2 * table.half[2] + CUBE_HALF])
    spec.cube_yaw = rng.uniform(-np.pi / 4, np.pi / 4)

    n_b = np.array([np.cos(wall_yaw[b_wall]), np.sin(wall_yaw[b_wall])])
    t_b = np.array([-n_b[1], n_b[0]])
    depth_b = rng.uniform(0.08, 0.30)
    xy_b = counter.center[:2] + n_b * counter.half[0] - n_b * depth_b + t_b * rng.uniform(-counter.half[1] + 0.1, counter.half[1] - 0.1)
    spec.bowl_pos = np.array([xy_b[0], xy_b[1], 2 * counter.half[2]])
    return spec


def build_model(scene: SceneSpec):
    """Compose the MuJoCo model: scene boxes + bowl + free cube + mobile base with the Panda attached."""
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    w = spec.worldbody
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.82, 0.8, 0.76], rgb2=[0.74, 0.72, 0.68], width=512, height=512)
    mat = spec.add_material(name="floor", texrepeat=[8, 8], reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[scene.room_half, scene.room_half, 0.05], material="floor")
    w.add_light(pos=[0, 0, 3.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], castshadow=False)
    w.add_light(pos=[2, -2, 3], dir=[-0.4, 0.4, -1], diffuse=[0.35, 0.35, 0.35], castshadow=False)
    for i, (x, y) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        w.add_geom(name=f"wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02 if x else scene.room_half, scene.room_half if x else 0.02, 0.6],
                   pos=[x * (scene.room_half + 0.02), y * (scene.room_half + 0.02), 0.6], rgba=[0.9, 0.9, 0.88, 0.25],
                   contype=0, conaffinity=0)
    for f in scene.furniture:
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), f.yaw)
        w.add_geom(name=f.name, type=mujoco.mjtGeom.mjGEOM_BOX, size=f.half, pos=f.center, quat=q, rgba=f.rgba)
    bowl = w.add_body(name="bowl", pos=scene.bowl_pos)
    for k in range(12):  # bowl rim from small boxes (visual only; target receptacle is excluded from the ESDF)
        a = 2 * np.pi * k / 12
        qq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(qq, np.array([0, 0, 1.0]), a)
        bowl.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.03, 0.025], pos=[0.075 * np.cos(a), 0.075 * np.sin(a), 0.025],
                      quat=qq, rgba=[0.85, 0.3, 0.25, 1], contype=0, conaffinity=0)
    bowl.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.07, 0.004, 0], pos=[0, 0, 0.004], rgba=[0.85, 0.3, 0.25, 1],
                  contype=0, conaffinity=0)
    cube = w.add_body(name="cube", pos=scene.cube_pos)
    cube.add_freejoint(name="cube_free")
    cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, rgba=[0.15, 0.65, 0.25, 1], contype=0, conaffinity=0)

    base = w.add_body(name="base", pos=[0, 0, 0])
    base.add_joint(name="base_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    base.add_joint(name="base_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    base.add_joint(name="base_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
    base.add_geom(name="base_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=BASE_HALF, pos=[0, 0, BASE_HALF[2] + 0.0],
                  rgba=[0.2, 0.22, 0.25, 1], contype=0, conaffinity=0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            base.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.06, 0.02, 0], pos=[sx * 0.2, sy * 0.27, 0.06],
                          euler=[np.pi / 2, 0, 0], rgba=[0.05, 0.05, 0.05, 1], contype=0, conaffinity=0)
    mount = base.add_site(name="mount", pos=[ARM_MOUNT_X, 0, BASE_TOP])
    # camera on a side mast, looking down at the workspace in front of the base (the VLA's RGB input)
    base.add_camera(name="base_cam", pos=[0.0, 0.32, 1.7], xyaxes=[-0.3714, -0.9285, 0.0, 0.6877, -0.2751, 0.6718], fovy=70)

    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    hand = panda.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.01] * 3, rgba=[1, 0, 0, 1])
    for k in list(panda.keys):
        k.delete()
    spec.attach(panda, prefix="panda/", site=mount)
    model = spec.compile()
    return spec, model


class RobotIndex:
    """Joint / dof bookkeeping for the composed model."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self.base_q = np.array([model.jnt_qposadr[jid(n)] for n in ("base_x", "base_y", "base_yaw")])
        self.base_v = np.array([model.jnt_dofadr[jid(n)] for n in ("base_x", "base_y", "base_yaw")])
        arm = [jid(f"panda/joint{i}") for i in range(1, 8)]
        self.arm_q = np.array([model.jnt_qposadr[j] for j in arm])
        self.arm_v = np.array([model.jnt_dofadr[j] for j in arm])
        self.arm_lo = model.jnt_range[arm, 0].copy()
        self.arm_hi = model.jnt_range[arm, 1].copy()
        self.fingers_q = np.array([model.jnt_qposadr[jid(f"panda/finger_joint{i}")] for i in (1, 2)])
        self.cube_q = model.jnt_qposadr[jid("cube_free")]
        self.tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "panda/tcp")
        self.robot_bodies = {b for b in range(model.nbody)
                             if model.body(b).name.startswith("panda/") or model.body(b).name == "base"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sc = random_scene(rng)
    _, m = build_model(sc)
    idx = RobotIndex(m)
    print("nq", m.nq, "nv", m.nv, "nbody", m.nbody, "base_q", idx.base_q, "arm_q", idx.arm_q, "tcp", idx.tcp)
