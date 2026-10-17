"""Experiment M1: MoManipVLA vs. baselines and cost-term ablations (paper Tables 1-2, scaled down).

  python momanipvla_repro/run_eval.py --episodes 100 --workers 4
"""
import argparse
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planner import CostWeights, PlannerConfig  # noqa: E402
from tasks import Episode  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "momanipvla"

VARIANTS = {
    # name: (planner method, cost weights, feasibility-first selection)
    "fixed_base_vla": ("fixed", CostWeights(), False),
    "heuristic_base": ("heuristic", CostWeights(), False),
    "momanip_bilevel": ("bilevel", CostWeights(), False),
    "momanip_direct": ("direct", CostWeights(), False),
    "bilevel_wo_reach": ("bilevel", CostWeights(reach=0.0), False),
    "bilevel_wo_smooth": ("bilevel", CostWeights(smooth=0.0), False),
    "bilevel_wo_collision": ("bilevel", CostWeights(collision=0.0), False),
    "bilevel_feasible_select": ("bilevel", CostWeights(), True),     # our fix (Sec. limitation)
    "direct_feasible_select": ("direct", CostWeights(), True),
}


def job(args):
    seed, variant, vla_kind = args
    method, w, feas = VARIANTS[variant]
    from tasks import OracleVLA
    cfg = replace(PlannerConfig(), weights=w, select_feasible=feas)
    if vla_kind == "learned":
        from vla_policy import LearnedVLA                      # fixed-base-trained waypoint policy
        ep = Episode(seed)
        vla = LearnedVLA(ep, device="cpu")
    else:                                                       # "oracle:<sigma>"
        sigma = float(vla_kind.split(":")[1])
        ep = Episode(seed, vla_factory=lambda r: OracleVLA(sigma_pos=sigma, seed=seed))
        vla = None
    t0 = time.time()
    r = ep.run(method, cfg=cfg, vla=vla)
    return dict(seed=seed, variant=variant, vla=vla_kind, pick=r.pick, place=r.place, success=r.success,
                steps=r.steps, latency_ms=float(np.mean(r.latency_ms)) if r.latency_ms else 0.0,
                fail_reason=r.fail_reason, base_travel=r.base_travel, min_clearance=r.min_clearance,
                smoothness=r.smooth, wall_s=time.time() - t0)


def summarize(df):
    g = df.groupby(["vla", "variant"])
    out = g.agg(pick_sr=("pick", "mean"), place_sr=("place", "mean"), overall_sr=("success", "mean"),
                steps=("steps", "mean"), latency_ms=("latency_ms", "mean"), n=("seed", "count")).reset_index()
    out["partial_sr"] = (out.pick_sr + out.place_sr) / 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--start-seed", type=int, default=1000)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--vla", nargs="+", default=["oracle:0.008"], help="oracle:<sigma_m> and/or learned")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, v, k) for k in args.vla for s in range(args.start_seed, args.start_seed + args.episodes)
            for v in args.variants]
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(job, jobs)):
            rows.append(row)
            if i % 25 == 0:
                print(f"[{i + 1}/{len(jobs)}] seed={row['seed']} {row['variant']}: success={row['success']} "
                      f"{row['fail_reason']}", flush=True)
            if i % 100 == 0:
                pd.DataFrame(rows).to_csv(OUT / f"episodes_{args.tag}.csv", index=False)
    df = pd.DataFrame(rows).sort_values(["vla", "variant", "seed"])
    df.to_csv(OUT / f"episodes_{args.tag}.csv", index=False)
    summ = summarize(df)
    summ.to_csv(OUT / f"summary_{args.tag}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.round(3))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
