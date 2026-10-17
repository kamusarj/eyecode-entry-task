"""Export six held-out HoMMI samples and the depth-only WBC replay data.

Run in the HoMMI environment; inputs are the original trained checkpoint/dataset.
The output is review data, not a replacement training set.
"""
import argparse
import json
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cfg.task.dataset_path = str(Path(args.dataset).resolve())
    cfg.task.dataset.cache_dir = str(Path(args.cache_dir).resolve())
    ds = hydra.utils.instantiate(cfg.task.dataset)
    val = ds.get_validation_dataset()
    ep_ids = np.array([e for (_, e, *_) in val.sampler.indices])
    selected = [int(np.flatnonzero(ep_ids == e)[len(np.flatnonzero(ep_ids == e)) // 2])
                for e in sorted(np.unique(ep_ids))]
    samples = [val[i] for i in selected]
    arrays = {f"obs/{k}": np.stack([s["obs"][k].numpy() for s in samples])
              for k in samples[0]["obs"]}
    arrays["action"] = np.stack([s["action"].numpy() for s in samples])
    out = ROOT / "data" / "review_samples"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "hommi_validation_6.npz", **arrays)
    metadata = dict(action_indexing=ds.action_indexing, validation_indices=selected,
                    episode_ids=[int(ep_ids[i]) for i in selected],
                    n_validation_samples=len(val), n_validation_episodes=len(selected),
                    selection="middle sample of each held-out episode; no success filtering",
                    source="real-stanford/hommi Laundry-v3; official processed laundry30",
                    action_step_seconds=0.05)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with zarr.ZipStore(args.dataset, mode="r") as src, zarr.ZipStore(
            str(ROOT / "data/wbc_laundry30/replay_buffer_lookat_only.zarr.zip"), mode="w") as dst:
        original, compact = zarr.open(src, mode="r"), zarr.group(dst)
        for name in ["data/camera_head_lookatpoint", "meta/episode_ends"]:
            compact.create_dataset(name, data=original[name][:])
    print(f"Exported {len(samples)} held-out samples and depth-only WBC replay data")


if __name__ == "__main__":
    main()
