"""Batch experiment H1: head action representation vs. human-robot embodiment gap.

For every (episode, alignment, head-height offset, head mode) the demo is replayed through the
official whole-body IK and summarized. Results go to results/hommi_wbc/*.csv.

  python hommi_repro/run_wbc_ablation.py --num-episodes 30 --workers 6
"""
import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wbc_sim import HEAD_MODES, WBCReplay, load_plan_episodes, summarize  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "hommi_wbc"
_sim, _eps = None, None


def _init(episode_source):
    global _sim, _eps
    _sim = WBCReplay()
    _eps = load_episodes(episode_source)


def load_episodes(source):
    if source == "cup1":
        return load_plan_episodes()
    from laundry_episodes import load_laundry_episodes  # processed official Laundry-v3 data
    return load_laundry_episodes(source)


def _job(args):
    ep_idx, align, dh, mode, max_s = args
    ep = _eps[ep_idx]
    t0 = time.time()
    log = _sim.run(ep, mode, align=align, head_height_offset=dh, max_s=max_s)
    row = summarize(log)
    p95 = np.percentile(np.r_[log["pos_err_l"], log["pos_err_r"]], 95)
    row.update(episode=ep.name, align=align, head_offset_m=dh, mode=mode,
               trackable=bool(p95 < 0.02 and row["gaze_err_deg"] < 10.0),
               runtime_s=time.time() - t0, sim_s=float(log["t"][-1] - log["t"][0]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cup1", help="'cup1' (rby1-wbc demo plan) or a processed dataset_plan.pkl")
    ap.add_argument("--num-episodes", type=int, default=30)
    ap.add_argument("--offsets", type=float, nargs="+", default=[-0.15, 0.0, 0.15, 0.30])
    ap.add_argument("--max-s", type=float, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="cup1")
    args = ap.parse_args()

    n_eps = min(args.num_episodes, len(load_episodes(args.source)))
    jobs = [(e, "relative", 0.0, m, args.max_s) for e, m in itertools.product(range(n_eps), HEAD_MODES)]
    jobs += [(e, "gripper_centric", dh, m, args.max_s)
             for e, dh, m in itertools.product(range(n_eps), args.offsets, HEAD_MODES)]
    print(f"{len(jobs)} runs on {n_eps} episodes")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(args.workers, initializer=_init, initargs=(args.source,)) as pool:
        for i, row in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(row)
            if i % 20 == 0:
                print(f"[{i + 1}/{len(jobs)}] {row['episode'][:30]} {row['align']} {row['head_offset_m']:+.2f} "
                      f"{row['mode']}: ee={row['ee_pos_err_mm']:.1f}mm gaze={row['gaze_err_deg']:.1f}deg", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"per_episode_{args.tag}.csv", index=False)
    agg = (df.groupby(["align", "head_offset_m", "mode"])
             .agg(ee_pos_err_mm=("ee_pos_err_mm", "mean"), ee_pos_err_p95_mm=("ee_pos_err_p95_mm", "mean"),
                  ee_rot_err_deg=("ee_rot_err_deg", "mean"), gaze_err_deg=("gaze_err_deg", "mean"),
                  gaze_in_10deg=("gaze_in_10deg", "mean"), head_pos_err_mm=("head_pos_err_mm", "mean"),
                  near_limit_frac=("near_limit_frac", "mean"), qp_fail_frac=("qp_fail_frac", "mean"),
                  trackable_rate=("trackable", "mean"), n=("episode", "count"))
             .reset_index())
    agg.to_csv(OUT / f"summary_{args.tag}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(agg.round(3))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
