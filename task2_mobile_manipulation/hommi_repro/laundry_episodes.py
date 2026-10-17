"""Load processed HoMMI Laundry-v3 demonstrations for the whole-body-controller replay.

Poses come from the official `dataset_plan.pkl` (same format as rby1-wbc/demo/dataset_plan.pkl).
The look-at point is the one the official pipeline stores in the replay buffer
(`camera_head_lookatpoint` = [0, 0, median depth of the central rays] in the head-camera frame,
`hommi/demonstration_processing/utils/lookat_util.py`), mapped to the world with the head pose.
"""
from pathlib import Path

import numpy as np
import zarr

from wbc_sim import load_plan_episodes


def load_laundry_episodes(session_dir):
    session_dir = Path(session_dir)
    zips = sorted(session_dir.glob("replay_buffer_*.zarr.zip"))
    if not zips:
        raise FileNotFoundError(f"Required replay_buffer_*.zarr.zip with recorded look-at depth missing in {session_dir}")
    eps = load_plan_episodes(session_dir / "dataset_plan.pkl")
    with zarr.ZipStore(str(zips[0]), mode="r") as store:
        root = zarr.open(store, mode="r")
        look = np.asarray(root["data/camera_head_lookatpoint"][:], dtype=np.float64)
        ends = np.asarray(root["meta/episode_ends"][:])
    starts = np.r_[0, ends[:-1]]
    for ep, s, e in zip(eps, starts, ends):
        z = look[s:e, 2]
        n = min(len(z), len(ep.head))
        valid = z[:n] > 0.05
        # frames without valid depth on the central rays: hold the last valid depth
        z_f = np.where(valid, z[:n], np.nan)
        for i in range(1, n):
            if np.isnan(z_f[i]):
                z_f[i] = z_f[i - 1]
        z_f = np.nan_to_num(z_f, nan=1.0)
        H = ep.head[:n]
        ep.left, ep.right, ep.head = ep.left[:n], ep.right[:n], H
        ep.lookat = H[:, :3, 3] + z_f[:, None] * H[:, :3, 2]
    return eps
