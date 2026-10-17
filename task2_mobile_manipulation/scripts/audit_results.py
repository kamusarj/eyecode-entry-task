"""Independently recompute report metrics from the bundled raw results (CPU only)."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def wilson(k, n):
    z = 1.96
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [max(0, center-half), min(1, center+half)]


def main():
    rows = list(csv.DictReader((RES / "momanipvla/episodes_main.csv").open()))
    groups = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row)
    recorded = {r["variant"]: r for r in csv.DictReader((RES / "momanipvla/summary_main.csv").open())}
    report = {"momanipvla": {}}
    for name, records in groups.items():
        seeds = [int(r["seed"]) for r in records]
        assert len(seeds) == len(set(seeds)) == 60, (name, "missing/duplicate episode")
        assert sorted(seeds) == list(range(1000, 1060))
        k = sum(r["success"] == "True" for r in records)
        rate = k / len(records)
        assert math.isclose(rate, float(recorded[name]["overall_sr"]), abs_tol=1e-10)
        report["momanipvla"][name] = dict(successes=k, n=len(records), success_rate=rate,
                                          wilson_95=wilson(k, len(records)))
    base = {int(r["seed"]): r["success"] == "True" for r in groups["momanip_bilevel"]}
    fixed = {int(r["seed"]): r["success"] == "True" for r in groups["bilevel_feasible_select"]}
    rescued = sum(fixed[s] and not base[s] for s in base)
    broken = sum(base[s] and not fixed[s] for s in base)
    n = rescued + broken
    p = min(1, 2*sum(math.comb(n, i) for i in range(min(rescued, broken)+1))/2**n) if n else 1
    assert [rescued, broken] == [8, 0] and p == 0.0078125
    report["paired_improvement"] = dict(rescued=rescued, broken=broken, exact_mcnemar_p=p)
    saved = json.loads((RES / "hommi_policy/offline_metrics.json").read_text())
    report["hommi_offline"] = {}
    for mode in ["policy", "hold_still"]:
        with np.load(RES / f"hommi_policy/errors_{mode}.npz", allow_pickle=False) as data:
            for key in data.files:
                a = data[key]
                assert a.shape == (395, 32) and np.isfinite(a).all()
                mean = float(a.mean())
                assert math.isclose(mean, saved[f"{mode}/{key}/mean"], rel_tol=1e-6)
                report["hommi_offline"][f"{mode}/{key}/mean"] = mean
    report["wbc_zero_offset"] = {}
    for tag in ["cup1", "laundry"]:
        raw = list(csv.DictReader((RES / f"hommi_wbc/per_episode_{tag}.csv").open()))
        summaries = list(csv.DictReader((RES / f"hommi_wbc/summary_{tag}.csv").open()))
        for s in summaries:
            selected = [r for r in raw if r["align"] == s["align"] and r["mode"] == s["mode"]
                        and float(r["head_offset_m"]) == float(s["head_offset_m"])]
            assert len(selected) == int(s["n"])
            for key in ["ee_pos_err_mm", "ee_pos_err_p95_mm", "gaze_err_deg"]:
                mean = sum(float(r[key]) for r in selected)/len(selected)
                assert math.isclose(mean, float(s[key]), rel_tol=1e-6, abs_tol=1e-8)
            if s["align"] == "gripper_centric" and float(s["head_offset_m"]) == 0:
                report["wbc_zero_offset"][f"{tag}/{s['mode']}"] = {
                    k: float(s[k]) for k in ["ee_pos_err_mm", "ee_pos_err_p95_mm", "gaze_err_deg"]}
    out = RES / "verification"
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifact_metrics_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: 540 planning rows, paired McNemar test, 395x32 offline errors per metric, WBC aggregates")


if __name__ == "__main__":
    main()
