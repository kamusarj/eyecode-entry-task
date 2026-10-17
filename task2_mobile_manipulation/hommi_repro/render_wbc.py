"""Render a side-by-side video of the three head-control modes on one demonstration.

  MUJOCO_GL=egl python hommi_repro/render_wbc.py --episode 3 --offset 0.15 --out results/hommi_wbc/wbc_modes.mp4
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wbc_sim import HEAD_MODES, WBCReplay, load_plan_episodes  # noqa: E402

TITLES = {"direct_lookat": "HoMMI: 3D look-at point", "ik": "Baseline: 6-DoF head pose in IK",
          "hold_init": "Ablation: no active neck"}


def add_geom(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def add_line(scn, a, b, rgba, width=0.006):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def add_frame(scn, T, length=0.08, alpha=1.0):
    for k, c in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        add_line(scn, T[:3, 3], T[:3, 3] + length * T[:3, k], [*c, alpha], 0.004)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=3)
    ap.add_argument("--offset", type=float, default=0.15, help="demonstrator head height offset (m)")
    ap.add_argument("--align", default="gripper_centric")
    ap.add_argument("--max-s", type=float, default=None)
    ap.add_argument("--out", default="results/hommi_wbc/wbc_head_modes.mp4")
    ap.add_argument("--source", default="cup1", help="'cup1' or a processed HoMMI session dir (Laundry)")
    ap.add_argument("--human-frames", default=None, help="npz from extract_head_frames.py (adds the human's view)")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=400)
    args = ap.parse_args()

    if args.source == "cup1":
        eps = load_plan_episodes()
    else:
        from laundry_episodes import load_laundry_episodes
        eps = load_laundry_episodes(args.source)
    human = np.load(args.human_frames) if args.human_frames else None
    sim = WBCReplay()
    runs = {m: sim.run(eps[args.episode], m, align=args.align, head_height_offset=args.offset,
                       max_s=args.max_s, record_qpos=True) for m in HEAD_MODES}
    m = sim.model
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 150, -18, 2.4
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    n = min(len(r["qpos"]) for r in runs.values())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=20, quality=8, macro_block_size=8)
    for t in range(0, n, 5):                       # 100 Hz IK ticks -> 20 fps video
        panels = []
        for mode in HEAD_MODES:
            r = runs[mode]
            d.qpos[:] = r["qpos"][t]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = [d.qpos[0] + 0.45, d.qpos[1], 1.0]
            renderer.update_scene(d, cam)
            scn = renderer.scene
            tg = r["targets"][t]
            tcp = sim.fk(r["qpos"][t])
            ok_l = r["pos_err_l"][t] < 0.02
            ok_r = r["pos_err_r"][t] < 0.02
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0], tg["left"][:3, 3], np.eye(3),
                     [0.1, 0.8, 0.2, 0.6] if ok_l else [0.9, 0.1, 0.1, 0.8])
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0], tg["right"][:3, 3], np.eye(3),
                     [0.1, 0.8, 0.2, 0.6] if ok_r else [0.9, 0.1, 0.1, 0.8])
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0, 0], tg["look"], np.eye(3), [1.0, 0.85, 0.1, 0.9])
            head = tcp["head"]
            dist = np.linalg.norm(tg["look"] - head[:3, 3])
            add_line(scn, head[:3, 3], head[:3, 3] + dist * head[:3, 2], [1.0, 0.85, 0.1, 0.8], 0.005)
            if mode == "ik":
                add_frame(scn, tg["head"], 0.1, 0.9)          # the human head pose the IK has to track
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.035, 0, 0], tg["head"][:3, 3], np.eye(3),
                         [0.8, 0.2, 0.8, 0.5])
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            e = 1000 * max(r["pos_err_l"][t], r["pos_err_r"][t])
            draw.rectangle([0, 0, args.width, 58], fill=(255, 255, 255))
            draw.text((8, 4), TITLES[mode], fill=(0, 0, 0), font=font)
            draw.text((8, 22), f"hand tracking error: {e:6.1f} mm", fill=(180, 0, 0) if e > 20 else (0, 110, 0), font=font)
            draw.text((8, 40), f"gaze error: {r['gaze_err'][t]:5.1f} deg", fill=(180, 0, 0) if r["gaze_err"][t] > 10 else (0, 110, 0), font=font)
            panels.append(np.asarray(img))
        if human is not None:
            k = min(1 + t // 10, len(human["head"]) - 1)        # 10 Hz commands <-> 10 Hz extracted frames
            hh = args.height
            head = np.asarray(Image.fromarray(human["head"][k]).resize((hh - hh // 3, hh - hh // 3)))
            wl = np.asarray(Image.fromarray(human["left"][k]).resize((hh // 3, hh // 3)))
            wr = np.asarray(Image.fromarray(human["right"][k]).resize((hh // 3, hh // 3)))
            col = np.full((hh, hh - hh // 3, 3), 255, np.uint8)
            col[: hh - hh // 3] = head
            col[hh - hh // 3:, : hh // 3] = wl
            col[hh - hh // 3:, hh // 3: 2 * (hh // 3)] = wr
            img = Image.fromarray(col)
            ImageDraw.Draw(img).text((6, 4), "human demo: head + wrist cams", fill=(255, 255, 255), font=font)
            hum = np.full((args.height, args.width, 3), 255, np.uint8)
            hum[:, (args.width - img.width) // 2:(args.width - img.width) // 2 + img.width] = np.asarray(img)
            panels = [hum] + panels
            # 2x2 grid: human view | HoMMI  //  6-DoF baseline | no neck
            frame = np.concatenate([np.concatenate(panels[:2], 1), np.concatenate(panels[2:], 1)], 0)
        else:
            frame = np.concatenate(panels, axis=1)
        writer.append_data(frame)
        if t == n // 2:
            imageio.imwrite(out.with_suffix(".png"), frame)
    writer.close()
    renderer.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
