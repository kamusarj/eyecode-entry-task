"""A small learned "VLA" waypoint policy trained on FIXED-BASE demonstrations.

MoManipVLA fine-tunes OpenVLA-7B with LoRA on 200 fixed-base expert episodes (4x RTX 3090). That is
out of reach for a 6 GB laptop GPU, so we keep the recipe and shrink the model:
  * vision-language backbone: frozen DINOv3 ViT-S/16 patch tokens of the robot camera image
    (+ a learned embedding of the instruction "pick up the cube" / "put the cube in the bowl"),
  * policy head (trained): 2-layer transformer over [instruction, proprio, 196 patch tokens] that
    regresses the NEXT end-effector waypoint in the base frame (position, hand yaw, gripper).
Demonstrations are collected with the base parked at a reachable pose (fixed-base data), and the
policy is then deployed inside the mobile-manipulation episodes of tasks.py, exactly where the
paper plugs in the VLA.

  python momanipvla_repro/vla_policy.py gen   --episodes 300      # render demos (CPU/EGL)
  python momanipvla_repro/vla_policy.py feats                      # DINOv3 features (GPU)
  python momanipvla_repro/vla_policy.py train --epochs 80          # train the head (GPU)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kinematics import base_to_world, world_to_base  # noqa: E402
from tasks import GRIPPER_CLOSED, GRIPPER_OPEN, Episode, OracleVLA  # noqa: E402
from scene import ARM_HOME  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "vla_demos"
NAMES = OracleVLA.NAMES
INSTR = {n: (0 if n in ("pregrasp", "grasp", "lift") else 1) for n in NAMES}   # 0: pick, 1: place
IMG = 224


class CamRenderer:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.r = mujoco.Renderer(model, IMG, IMG)

    def render(self, idx, base, q, fingers, cube):
        d = self.d
        d.qpos[idx.base_q] = base
        d.qpos[idx.arm_q] = q
        d.qpos[idx.fingers_q] = fingers
        d.qpos[idx.cube_q: idx.cube_q + 3] = cube
        d.qpos[idx.cube_q + 3: idx.cube_q + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, d)
        self.r.update_scene(d, "base_cam")
        return self.r.render().copy()

    def close(self):
        self.r.close()


def proprio(kin, base, q, fingers):
    p, R = kin.fk(base, q)
    p_b, R_b = world_to_base(base, p, R)
    yaw = np.arctan2(R_b[1, 1], R_b[0, 1])            # direction of the finger axis (TCP y) in the base frame
    return np.array([*p_b, np.sin(2 * yaw), np.cos(2 * yaw), fingers / GRIPPER_OPEN], np.float32)


def target_vec(p_b, R_b, gripper):
    yaw = np.arctan2(R_b[1, 1], R_b[0, 1])
    # the parallel gripper is symmetric under 180 deg -> regress the doubled angle
    return np.array([*p_b, np.sin(2 * yaw), np.cos(2 * yaw), gripper / GRIPPER_OPEN], np.float32)


GRIP_AFTER = {"pregrasp": GRIPPER_OPEN, "grasp": GRIPPER_CLOSED, "lift": GRIPPER_CLOSED,
              "preplace": GRIPPER_CLOSED, "place": GRIPPER_OPEN, "retreat": GRIPPER_OPEN}


def generate(n_episodes, seed0=5000, n_views=3):
    """Fixed-base demos: park the base where every keypose of a phase is reachable, then step through
    the keyposes with IK, recording (image, instruction, proprio) -> next waypoint (base frame)."""
    DATA.mkdir(parents=True, exist_ok=True)
    imgs, instr, prop, tgt, meta = [], [], [], [], []
    oracle = OracleVLA(sigma_pos=0.0, sigma_yaw_deg=0.0, seed=0)
    for k in range(n_episodes):
        seed = seed0 + k
        ep = Episode(seed)
        rng = np.random.default_rng(seed)
        kin, idx = ep.kin, ep.kin.idx
        cam = CamRenderer(ep.model)
        state = dict(cube=ep.scene.cube_pos.copy(), cube_yaw=ep.scene.cube_yaw, bowl=ep.scene.bowl_pos.copy())
        q, fingers, attached = ARM_HOME.copy(), GRIPPER_OPEN, None
        ok = True
        for phase, names in (("pick", ("pregrasp", "grasp", "lift")), ("place", ("preplace", "place", "retreat"))):
            target_xy = state["cube"][:2] if phase == "pick" else state["bowl"][:2]
            base = None
            for _ in range(60):                         # fixed-base data collection pose
                d = rng.uniform(0.50, 0.72)
                v = (ep.nav_pick if phase == "pick" else ep.nav_place)[:2] - target_xy   # approach side
                ang = np.arctan2(v[1], v[0]) + rng.normal(0, 0.35)
                xy = target_xy + d * np.array([np.cos(ang), np.sin(ang)])
                yaw = np.arctan2(target_xy[1] - xy[1], target_xy[0] - xy[0]) + rng.normal(0, 0.2)
                cand = np.array([xy[0], xy[1], yaw])
                kin.set(cand, q)
                if ep.esdf.query(kin.query_points()).min() < 0.03:
                    continue
                q_try, good = q.copy(), True
                for n in names:
                    p_b, R_b = oracle.predict(n, cand, state)
                    p_w, R_w = base_to_world(cand, p_b, R_b)
                    q_try, _, conv, _, _ = kin.ik(cand, p_w, R_w, q_init=q_try)
                    if not conv or ep.esdf.query(kin.query_points()).min() < 0.0:
                        good = False
                        break
                if good:
                    base = cand
                    break
            if base is None:
                ok = False
                break
            for n in names:
                p_b, R_b = oracle.predict(n, base, state)
                cube_now = state["cube"] if attached is None else None
                if attached is not None:
                    p, R = kin.fk(base, q)
                    cube_now = p + R @ attached
                for v in range(n_views):                  # extra views: small base perturbations, same world target
                    b = base if v == 0 else base + np.array([*rng.normal(0, 0.06, 2), rng.normal(0, 0.08)])
                    imgs.append(cam.render(idx, b, q, fingers, cube_now))
                    instr.append(INSTR[n])
                    prop.append(proprio(kin, b, q, fingers))
                    p_w, R_w = base_to_world(base, p_b, R_b)
                    tb, Rbb = world_to_base(b, p_w, R_w)
                    tgt.append(target_vec(tb, Rbb, GRIP_AFTER[n]))
                    meta.append((seed, NAMES.index(n), v))
                p_w, R_w = base_to_world(base, p_b, R_b)
                q, _, _, _, _ = kin.ik(base, p_w, R_w, q_init=q)
                if n == "grasp":
                    p, R = kin.fk(base, q)
                    attached = R.T @ (state["cube"] - p)
                if n == "place":
                    p, R = kin.fk(base, q)
                    state["cube"] = p + R @ attached
                    attached = None
                fingers = GRIP_AFTER[n]
        cam.close()
        if k % 20 == 0:
            print(f"[{k}/{n_episodes}] samples={len(imgs)} ok={ok}", flush=True)
    np.savez_compressed(DATA / "demos.npz", imgs=np.stack(imgs), instr=np.array(instr), prop=np.stack(prop),
                        tgt=np.stack(tgt), meta=np.array(meta))
    print("saved", len(imgs), "samples")


# ----------------------------------------------------------------------------------------- model
def load_backbone(device):
    import torch
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def encode(imgs_uint8):
        x = torch.as_tensor(imgs_uint8, device=device).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = m(pixel_values=(x - mean) / std).last_hidden_state
        return out[:, 1 + m.config.num_register_tokens:].float()      # (B, 196, 384) patch tokens
    return encode


def extract_features():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(DATA / "demos.npz")
    enc = load_backbone(device)
    feats = [enc(d["imgs"][i:i + 32]).half().cpu().numpy() for i in range(0, len(d["imgs"]), 32)]
    np.save(DATA / "feats.npy", np.concatenate(feats))
    print("features", np.concatenate(feats).shape)


def build_head():
    import torch
    import torch.nn as nn

    class WaypointHead(nn.Module):
        def __init__(self, d=192, n_patch=196):
            super().__init__()
            self.patch = nn.Linear(384, d)
            self.prop = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
            self.instr = nn.Embedding(2, d)
            self.pos = nn.Parameter(torch.zeros(1, n_patch + 2, d))
            layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 3)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 6))

        def forward(self, patches, instr, prop):
            tok = torch.cat([self.instr(instr)[:, None], self.prop(prop)[:, None], self.patch(patches)], 1)
            h = self.enc(tok + self.pos)
            return self.out(h[:, 0])

    return WaypointHead()


def train(epochs=80, bs=64, lr=3e-4):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(DATA / "demos.npz")
    feats = np.load(DATA / "feats.npy")
    seeds = d["meta"][:, 0]
    val_mask = seeds >= np.quantile(np.unique(seeds), 0.85)          # hold out the last 15% of episodes
    stats = dict(p_mean=d["prop"][~val_mask].mean(0), p_std=d["prop"][~val_mask].std(0) + 1e-3,
                 t_mean=d["tgt"][~val_mask].mean(0), t_std=d["tgt"][~val_mask].std(0) + 1e-3)
    T = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    X = T(feats)                                                     # fp16 on the GPU (shared with HoMMI run)
    I = T(d["instr"]).long()
    P = T((d["prop"] - stats["p_mean"]) / stats["p_std"]).float()
    Y = T((d["tgt"] - stats["t_mean"]) / stats["t_std"]).float()
    tr, va = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    head = build_head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * int(np.ceil(len(tr) / bs)))
    w = torch.tensor([1, 1, 1, 0.3, 0.3, 0.3], device=device)
    log = []
    for e in range(epochs):
        head.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            Xb = X[b].float() + 0.05 * torch.randn(len(b), *X.shape[1:], device=device)   # feature noise aug.
            loss = ((head(Xb, I[b], P[b]) - Y[b]) ** 2 * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        head.eval()
        with torch.no_grad():
            pv = np.concatenate([head(X[c].float(), I[c], P[c]).cpu().numpy() for c in np.array_split(va, 8)]) * stats["t_std"] + stats["t_mean"]
        err = np.linalg.norm(pv[:, :3] - d["tgt"][va][:, :3], axis=1)
        log.append(dict(epoch=e, train_loss=tot / len(tr), val_pos_err_cm=100 * float(err.mean()),
                        val_pos_err_p90_cm=100 * float(np.percentile(err, 90))))
        if e % 10 == 0 or e == epochs - 1:
            print(log[-1], flush=True)
    torch.save(dict(head=head.state_dict(), stats=stats, log=log), DATA / "waypoint_head.pt")
    import pandas as pd
    pd.DataFrame(log).to_csv(ROOT / "results" / "momanipvla" / "vla_training_log.csv", index=False)
    # per-waypoint validation error (view 0 only)
    names = d["meta"][va][:, 1]
    per = {NAMES[k]: 100 * float(err[names == k].mean()) for k in np.unique(names)}
    print("val position error per waypoint (cm):", per)


class LearnedVLA:
    """Drop-in replacement for OracleVLA inside tasks.Episode.run (needs the current robot state)."""

    needs_state = True

    def __init__(self, episode, device=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(DATA / "waypoint_head.pt", map_location=self.device, weights_only=False)
        self.head = build_head().to(self.device).eval()
        self.head.load_state_dict(ck["head"])
        self.stats = ck["stats"]
        self.encode = load_backbone(self.device)
        self.ep = episode
        self.cam = CamRenderer(episode.model)

    def predict(self, name, base, state):
        torch = self.torch
        q, fingers, cube = state["q"], state["fingers"], state["cube_now"]
        img = self.cam.render(self.ep.kin.idx, base, q, fingers, cube)
        pr = (proprio(self.ep.kin, base, q, fingers) - self.stats["p_mean"]) / self.stats["p_std"]
        with torch.no_grad():
            f = self.encode(img[None])
            y = self.head(f, torch.tensor([INSTR[name]], device=self.device),
                          torch.as_tensor(pr[None], device=self.device).float())[0].cpu().numpy()
        y = y * self.stats["t_std"] + self.stats["t_mean"]
        yaw = 0.5 * np.arctan2(y[3], y[4])
        z = np.array([0.0, 0.0, -1.0])
        yb = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        R_b = np.stack([np.cross(yb, z), yb, z], 1)
        return y[:3].astype(float), R_b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "feats", "train"])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    if a.cmd == "gen":
        generate(a.episodes)
    elif a.cmd == "feats":
        extract_features()
    else:
        train(a.epochs)
