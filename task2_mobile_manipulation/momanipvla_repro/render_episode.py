"""Render MoManipVLA episodes (several methods side by side) to an mp4.

  MUJOCO_GL=egl python momanipvla_repro/render_episode.py --seed 1005 --variants fixed_base_vla momanip_bilevel
"""
import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner import PlannerConfig  # noqa: E402
from run_eval import VARIANTS  # noqa: E402
from tasks import Episode  # noqa: E402

LABEL = {"fixed_base_vla": "Fixed-base VLA (no base motion)", "heuristic_base": "Heuristic base placement",
         "momanip_bilevel": "MoManipVLA bi-level (paper)", "momanip_direct": "MoManipVLA direct search",
         "bilevel_feasible_select": "Bi-level + feasibility-first (ours)"}


def render_rollout(ep, res, label, width, height, font):
    m = ep.model
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height, width)
    rs = mujoco.Renderer(m, height // 3, width // 3)
    idx = ep.kin.idx
    cam = mujoco.MjvCamera()
    frames = []
    for k, (b, q, fingers, cube) in enumerate(res.trajectory):
        d.qpos[idx.base_q] = b
        d.qpos[idx.arm_q] = q
        d.qpos[idx.fingers_q] = fingers
        d.qpos[idx.cube_q: idx.cube_q + 3] = cube
        d.qpos[idx.cube_q + 3: idx.cube_q + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = [b[0], b[1], 0.6]
        cam.distance, cam.elevation = 2.9, -42
        cam.azimuth = np.degrees(b[2]) + 30          # from behind-left of the robot, looking at the wall
        r.update_scene(d, cam)
        img = Image.fromarray(r.render())
        rs.update_scene(d, "base_cam")
        inset = Image.fromarray(rs.render())
        img.paste(inset, (width - inset.width - 6, height - inset.height - 6))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width, 24], fill=(255, 255, 255))
        draw.text((6, 4), label, fill=(0, 0, 0), font=font)
        frames.append(np.asarray(img))
    status = "SUCCESS" if res.success else f"FAIL: {res.fail_reason}"
    for _ in range(25):                                # hold the last frame with the outcome
        img = Image.fromarray(frames[-1].copy())
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, height - 30, width // 2 + 120, height], fill=(255, 255, 255))
        draw.text((6, height - 25), status, fill=(0, 120, 0) if res.success else (190, 0, 0), font=font)
        frames.append(np.asarray(img))
    r.close()
    rs.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1005)
    ap.add_argument("--variants", nargs="+", default=["fixed_base_vla", "momanip_bilevel", "bilevel_feasible_select"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--vla", default="oracle", choices=["oracle", "learned"])
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    ep = Episode(args.seed)
    vla = None
    if args.vla == "learned":
        from vla_policy import LearnedVLA
        vla = LearnedVLA(ep, device="cpu")
    clips = []
    for v in args.variants:
        method, w, feas = VARIANTS[v]
        res = ep.run(method, cfg=replace(PlannerConfig(), weights=w, select_feasible=feas), record=True, vla=vla)
        print(v, "success" if res.success else res.fail_reason)
        label = LABEL.get(v, v) + (" | learned VLA" if args.vla == "learned" else "")
        clips.append(render_rollout(ep, res, label, args.width, args.height, font))
    n = max(len(c) for c in clips)
    clips = [c + [c[-1]] * (n - len(c)) for c in clips]
    out = Path(args.out or f"results/momanipvla/episode_{args.seed}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    def tile(frames):
        if len(frames) == 4:   # 2x2 grid keeps a 4:3-ish frame
            return np.concatenate([np.concatenate(frames[:2], 1), np.concatenate(frames[2:], 1)], 0)
        return np.concatenate(frames, axis=1)

    w = imageio.get_writer(out, fps=10, quality=8, macro_block_size=8)
    for t in range(n):
        w.append_data(tile([c[t] for c in clips]))
    w.close()
    imageio.imwrite(out.with_suffix(".png"), tile([c[min(len(c) - 30, n - 1)] for c in clips]))
    print("wrote", out)


if __name__ == "__main__":
    main()
