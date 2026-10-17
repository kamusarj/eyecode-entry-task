"""Extract the demonstrator's egocentric head RGB of one episode from the official replay buffer (for videos).

  source scripts/hommi_env.sh && python hommi_repro/extract_head_frames.py --episode 0
"""
import argparse
from pathlib import Path

import numpy as np
import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=str(ROOT / "data/hommi_processed/sessions/laundry30"))
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--stride", type=int, default=6, help="60 Hz -> 10 Hz")
    a = ap.parse_args()
    zp = sorted(Path(a.session).glob("replay_buffer_*.zarr.zip"))[0]
    root = zarr.open(zarr.ZipStore(str(zp), mode="r"), mode="r")
    ends = root["meta/episode_ends"][:]
    s, e = (0 if a.episode == 0 else ends[a.episode - 1]), ends[a.episode]
    frames = root["data/camera_head_main_rgb"][s:e:a.stride]
    left = root["data/camera_left_main_rgb"][s:e:a.stride]
    right = root["data/camera_right_main_rgb"][s:e:a.stride]
    out = ROOT / "data" / f"head_frames_ep{a.episode}.npz"
    np.savez_compressed(out, head=frames, left=left, right=right, stride=a.stride)
    print(out, frames.shape)


if __name__ == "__main__":
    main()
