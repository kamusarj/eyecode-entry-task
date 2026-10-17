"""Video of the trained HoMMI policy on held-out demonstrations: predicted vs. human future actions
drawn on the egocentric head image.

Projection without intrinsics: the head point map (left-gripper frame) gives the 3D point seen by every
pixel, so a 3D point X is drawn at the pixel whose viewing ray from the head camera centre is closest
in angle to the ray towards X.

  source scripts/hommi_env.sh
  python hommi_repro/visualize_policy.py --ckpt outputs/runs/.../latest.ckpt --episodes 0 3
"""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from eval_policy_offline import load_policy

ROOT = Path(__file__).resolve().parents[1]
C_LEFT, C_RIGHT, C_LOOK = (42, 120, 214), (235, 104, 52), (27, 175, 122)


def project(points, pm, head):
    """points (N,3), pm (3,H,W) point map, head (3,) camera centre -> (N,2) pixel coords (or None if behind)."""
    H, W = pm.shape[1:]
    rays = pm.reshape(3, -1).T - head
    n = np.linalg.norm(rays, axis=1)
    valid = n > 1e-3
    rays = rays / np.maximum(n, 1e-6)[:, None]
    out = []
    for p in points:
        d = p - head
        d = d / (np.linalg.norm(d) + 1e-9)
        cos = rays @ d
        cos[~valid] = -2
        i = int(np.argmax(cos))
        out.append(None if cos[i] < np.cos(np.radians(4)) else (i % W, i // W))
    return out


def draw_traj(draw, pix, color, alpha_fill, r=3):
    pts = [p for p in pix if p is not None]
    for a, b in zip(pts[:-1], pts[1:]):
        draw.line([a, b], fill=color + (alpha_fill,), width=2)
    for p in pts[::4]:
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color + (alpha_fill,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", help="Override the original training dataset path")
    ap.add_argument("--cache-dir", help="Override the dataset cache directory")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 3], help="indices among validation episodes")
    ap.add_argument("--out", default=str(ROOT / "results" / "hommi_policy" / "policy_predictions.mp4"))
    ap.add_argument("--scale", type=int, default=2)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, policy = load_policy(a.ckpt, device)
    cfg.task.dataset_path = str(Path(a.dataset).resolve()) if a.dataset else str(
        ROOT / "data/hommi_processed/sessions/laundry30/replay_buffer_laundry30.zarr.zip")
    cfg.task.dataset.cache_dir = str(Path(a.cache_dir).resolve()) if a.cache_dir else str(ROOT / "data/hommi_cache")
    import hydra
    ds = hydra.utils.instantiate(cfg.task.dataset)
    val = ds.get_validation_dataset()
    idx = ds.action_indexing
    ep_of = np.array([e for (_, e, *_) in val.sampler.indices])
    eps = sorted(np.unique(ep_of))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    frames = []
    for k in a.episodes:
        sel = np.flatnonzero(ep_of == eps[k])
        for i in sel:
            s = val[int(i)]
            obs = {kk: v[None].to(device) for kk, v in s["obs"].items()}
            with torch.inference_mode():
                pred = policy.predict_action(obs)["action_pred"][0].float().cpu().numpy()
            gt = s["action"].numpy()
            pm = s["obs"]["camera_head_pointmap"][-1].numpy()
            head = s["obs"]["gripper_head_eef_pos"][-1].numpy()
            rgb = (s["obs"]["camera_head_main_rgb"][-1].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            img = Image.fromarray(rgb).convert("RGBA")
            over = Image.new("RGBA", img.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(over)
            for key, color in (("gripper_left_eef_pos", C_LEFT), ("gripper_right_eef_pos", C_RIGHT),
                               ("camera_head_lookatpoint", C_LOOK)):
                st, en = idx[key]
                draw_traj(dr, project(gt[:, st:en], pm, head), color, 90)
                draw_traj(dr, project(pred[:, st:en], pm, head), color, 255)
            img = Image.alpha_composite(img, over).convert("RGB")
            img = img.resize((img.width * a.scale, img.height * a.scale), Image.NEAREST)
            canvas = Image.new("RGB", (img.width, img.height + 44), (252, 252, 251))
            canvas.paste(img, (0, 44))
            d = ImageDraw.Draw(canvas)
            errs = {key: 100 * np.linalg.norm(pred[:, idx[key][0]:idx[key][1]] - gt[:, idx[key][0]:idx[key][1]], axis=1).mean()
                    for key in ("gripper_left_eef_pos", "gripper_right_eef_pos", "camera_head_lookatpoint")}
            d.text((6, 3), f"held-out demo {k}: faint = human future (1.6 s), solid = policy", fill=(11, 11, 11), font=font)
            d.text((6, 22), f"left {errs['gripper_left_eef_pos']:.1f} cm   right {errs['gripper_right_eef_pos']:.1f} cm   "
                   f"look-at {errs['camera_head_lookatpoint']:.1f} cm", fill=(82, 81, 78), font=font)
            frames.append(np.asarray(canvas))
    w = imageio.get_writer(a.out, fps=7.5, quality=8, macro_block_size=8)
    for f in frames:
        w.append_data(f)
    w.close()
    imageio.imwrite(Path(a.out).with_suffix(".png"), np.concatenate([frames[len(frames) // 4], frames[3 * len(frames) // 4]], 1))
    print("wrote", a.out, len(frames), "frames")


if __name__ == "__main__":
    main()
