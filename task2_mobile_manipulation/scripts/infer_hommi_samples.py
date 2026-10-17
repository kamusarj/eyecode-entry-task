"""Run the trained official HoMMI policy on the bundled six held-out samples."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hommi_repro"))
from eval_policy_offline import errors, hold_still, load_policy, plot_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/hommi_latest.ckpt"))
    ap.add_argument("--out", default=str(ROOT / "results/verification/hommi_samples"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    sample_dir = ROOT / "data/review_samples"
    meta = json.loads((sample_dir / "metadata.json").read_text())
    idx = meta["action_indexing"]
    with np.load(sample_dir / "hommi_validation_6.npz", allow_pickle=False) as data:
        obs = {k[4:]: data[k] for k in data.files if k.startswith("obs/")}
        gt = data["action"]
    _, policy = load_policy(args.checkpoint, args.device)
    with torch.inference_mode():
        pred = policy.predict_action({k: torch.from_numpy(v).to(args.device)
                                      for k, v in obs.items()})["action_pred"].float().cpu().numpy()
    baseline = hold_still(obs, idx, gt.shape[1])
    summary = dict(n_samples=len(gt), seed=42, device=args.device,
                   scope="six review samples; separate from the historical 395-sample evaluation")
    for name, values in [("policy", pred), ("hold_still", baseline)]:
        for key, value in errors(values, gt, idx).items():
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite {name}/{key}")
            summary[f"{name}/{key}/mean"] = float(value.mean())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", predictions=pred, ground_truth=gt)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_sample(obs, gt, pred, idx, out / "sample.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
