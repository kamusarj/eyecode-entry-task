"""Offline evaluation of a trained HoMMI policy on held-out demonstrations.

Runs the official policy (EMA weights, 16 DDIM steps) on validation samples of the official dataset
and reports errors in physical units against the ground-truth human action chunks:
  * gripper position error (cm) and rotation error (deg), left / right
  * gripper width error (mm)
  * 3D look-at point error (cm)
at several horizons, plus a "hold still" baseline that repeats the current state.
Also saves qualitative 3D plots (head point cloud + predicted vs. GT trajectories).

  source scripts/hommi_env.sh
  python hommi_repro/eval_policy_offline.py --ckpt outputs/runs/.../checkpoints/latest.ckpt
"""
import argparse
import json
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

OmegaConf.register_new_resolver("eval", eval, replace=True)

ROOT = Path(__file__).resolve().parents[1]


def rot6d_to_mat(d6):
    """pytorch3d convention (rows) — matches the vendored rotation_6d_to_matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, -1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def rot_err_deg(d6a, d6b):
    Ra, Rb = rot6d_to_mat(d6a), rot6d_to_mat(d6b)
    tr = np.einsum("...ij,...ij->...", Ra, Rb)       # trace(Ra Rb^T)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def load_policy(ckpt, device):
    payload = torch.load(open(ckpt, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    ws = hydra.utils.get_class(cfg._target_)(cfg)
    ws.load_payload(payload, exclude_keys=["optimizer"], include_keys=None, strict=False)
    policy = ws.ema_model if cfg.training.use_ema else ws.model
    policy.num_inference_steps = 16
    policy.to(device).eval()
    return cfg, policy


def hold_still(obs, idx, horizon):
    """Baseline: every effector keeps its current pose / width / look-at point (left-gripper frame)."""
    B = obs["gripper_left_eef_pos"].shape[0]
    out = np.zeros((B, horizon, max(e for _, e in idx.values())), np.float32)
    for key, (s, e) in idx.items():
        out[:, :, s:e] = obs[key][:, -1:, :].repeat(horizon, axis=1)
    return out


def errors(pred, gt, idx):
    res = {}
    for side in ("left", "right"):
        p, r, w = (idx[f"gripper_{side}_eef_pos"], idx[f"gripper_{side}_eef_rot_axis_angle"],
                   idx[f"gripper_{side}_gripper_width"])
        res[f"{side}_pos_cm"] = 100 * np.linalg.norm(pred[..., p[0]:p[1]] - gt[..., p[0]:p[1]], axis=-1)
        res[f"{side}_rot_deg"] = rot_err_deg(pred[..., r[0]:r[1]], gt[..., r[0]:r[1]])
        res[f"{side}_width_mm"] = 1000 * np.abs(pred[..., w[0]:w[1]] - gt[..., w[0]:w[1]])[..., 0]
    la = idx["camera_head_lookatpoint"]
    res["lookat_cm"] = 100 * np.linalg.norm(pred[..., la[0]:la[1]] - gt[..., la[0]:la[1]], axis=-1)
    return res   # each (B, horizon)


def plot_sample(obs, gt, pred, idx, path, b=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    if "camera_head_pointmap" in obs:
        pm = obs["camera_head_pointmap"][b, -1].reshape(3, -1).T
        rgb = obs["camera_head_main_rgb"][b, -1].reshape(3, -1).T if "camera_head_main_rgb" in obs else None
        keep = np.isfinite(pm).all(1) & (np.abs(pm).sum(1) > 0) & (np.linalg.norm(pm, axis=1) < 2.5)
        sel = np.flatnonzero(keep)[::7]
        ax.scatter(*pm[sel].T, s=0.6, c=rgb[sel] if rgb is not None else "#b9b8b3", alpha=0.5, linewidths=0)
    styles = {"gripper_left": "#2a78d6", "gripper_right": "#eb6834"}
    for side, color in styles.items():
        s, e = idx[f"{side}_eef_pos"]
        ax.plot(*gt[b, :, s:e].T, color=color, lw=2, alpha=0.35)
        ax.plot(*pred[b, :, s:e].T, color=color, lw=2)
    s, e = idx["camera_head_lookatpoint"]
    ax.scatter(*gt[b, :, s:e].T, color="#1baf7a", s=8, alpha=0.35)
    ax.scatter(*pred[b, :, s:e].T, color="#1baf7a", s=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("left gripper frame - faint: human GT, solid: policy\n"
                 "blue: left gripper, orange: right gripper, green: look-at", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", help="Override the dataset path embedded in the training checkpoint")
    ap.add_argument("--cache-dir", help="Local dataset cache directory (default: submission data/hommi_cache)")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every k-th validation sample")
    ap.add_argument("--max-batches", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "results" / "hommi_policy"))
    ap.add_argument("--n-plots", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, policy = load_policy(args.ckpt, device)
    cfg.task.dataset_path = str(Path(args.dataset).resolve()) if args.dataset else str(
        ROOT / "data/hommi_processed/sessions/laundry30/replay_buffer_laundry30.zarr.zip")
    cfg.task.dataset.cache_dir = str(Path(args.cache_dir).resolve()) if args.cache_dir else str(ROOT / "data/hommi_cache")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val = dataset.get_validation_dataset()
    idx = dataset.action_indexing
    subset = torch.utils.data.Subset(val, list(range(0, len(val), args.stride)))
    loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=2)
    horizon_s = 1.0 / 20.0      # 60 Hz data, action down-sample 3 -> 20 Hz
    acc_pol, acc_base = {}, {}
    n_plotted = 0
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if bi >= args.max_batches:
                break
            obs_t = {k: v.to(device) for k, v in batch["obs"].items()}
            pred = policy.predict_action(obs_t)["action_pred"].float().cpu().numpy()
            gt = batch["action"].numpy()
            obs = {k: v.numpy() for k, v in batch["obs"].items()}
            base = hold_still(obs, idx, gt.shape[1])
            for acc, p in ((acc_pol, pred), (acc_base, base)):
                for k, v in errors(p, gt, idx).items():
                    acc.setdefault(k, []).append(v)
            while n_plotted < args.n_plots and n_plotted < (bi + 1) * len(gt):
                plot_sample(obs, gt, pred, idx, out / f"sample_{n_plotted}.png", b=n_plotted % len(gt))
                n_plotted += 1
                break
    summary = {"n_samples": int(sum(len(v) for v in acc_pol["lookat_cm"])), "checkpoint": str(args.ckpt)}
    for name, acc in (("policy", acc_pol), ("hold_still", acc_base)):
        for k, v in acc.items():
            arr = np.concatenate(v)                    # (N, horizon)
            summary[f"{name}/{k}/mean"] = float(arr.mean())
            for step in (4, 8, 16, 31):
                summary[f"{name}/{k}/t={step * horizon_s:.2f}s"] = float(arr[:, step].mean())
        np.savez(out / f"errors_{name}.npz", **{k: np.concatenate(v) for k, v in acc.items()})
    (out / "offline_metrics.json").write_text(json.dumps(summary, indent=1))
    for k in ("left_pos_cm", "right_pos_cm", "left_rot_deg", "right_rot_deg", "left_width_mm", "lookat_cm"):
        print(f"{k:14s} policy {summary[f'policy/{k}/mean']:7.2f}   hold-still {summary[f'hold_still/{k}/mean']:7.2f}")


if __name__ == "__main__":
    main()
