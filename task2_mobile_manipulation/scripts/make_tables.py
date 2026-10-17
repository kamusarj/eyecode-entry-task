"""Build the report tables (LaTeX snippets in report/tables/) from the result CSV/JSON files.

  python scripts/make_tables.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = ROOT.parent / "report" / "tables" / "part2"

MODE = {"direct_lookat": "HoMMI look-at point", "ik": "6-DoF head pose in IK", "hold_init": "No active neck"}
VAR = {
    "fixed_base_vla": "Fixed-base VLA (no base motion)", "heuristic_base": "Heuristic base placement",
    "momanip_direct": "MoManipVLA, direct search", "momanip_bilevel": "MoManipVLA, bi-level",
    "bilevel_wo_reach": "\\quad w/o reachability $F_r$", "bilevel_wo_smooth": "\\quad w/o smoothness $F_s$",
    "bilevel_wo_collision": "\\quad w/o collision $F_c$",
    "bilevel_feasible_select": "Bi-level + feasibility-first (ours)",
    "direct_feasible_select": "Direct + feasibility-first (ours)",
}


def wilson(k, n, z=1.96):
    """95% Wilson interval for a success rate (small-sample friendly)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def wbc_table():
    rows = []
    for tag, label in (("cup1", "cup1 (tabletop)"), ("laundry", "Laundry (HoMMI)")):
        p = RES / "hommi_wbc" / f"summary_{tag}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[(df["align"] == "gripper_centric") & (df["head_offset_m"].abs() < 1e-6)]
        for mode in MODE:
            r = df[df["mode"] == mode].iloc[0]
            rows.append((label, MODE[mode], r.ee_pos_err_mm, r.ee_pos_err_p95_mm, r.ee_rot_err_deg, r.gaze_err_deg,
                         100 * r.gaze_in_10deg, 100 * r.trackable_rate, int(r.n)))
    if not rows:
        return
    lines = ["\\begin{tabular}{llrrrrrr}\\toprule",
             "Demos & Head action & \\multicolumn{2}{c}{Hand pos.\\ err.\\ (mm)} & Hand rot. & Gaze err. & Gaze$<$10$^\\circ$ & Trackable \\\\",
             " & & mean & p95 & err.\\ ($^\\circ$) & ($^\\circ$) & (\\% time) & (\\% eps) \\\\\\midrule"]
    last = None
    for r in rows:
        demo = r[0] + f" ($n$={r[8]})" if r[0] != last else ""
        if last is not None and r[0] != last:
            lines.append("\\midrule")
        last = r[0]
        lines.append(f"{demo} & {r[1]} & {r[2]:.2f} & {r[3]:.1f} & {r[4]:.2f} & {r[5]:.1f} & {r[6]:.0f} & {r[7]:.0f} \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    (TAB / "wbc.tex").write_text("\n".join(lines))


def momanip_table(tag="main", key="vla_sigma"):
    p = RES / "momanipvla" / f"episodes_{tag}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "vla" in df.columns:
        key = "vla"
    out = []
    for v in VAR:
        d = df[df.variant == v]
        if len(d) == 0:
            continue
        n, k = len(d), int(d.success.sum())
        lo, hi = wilson(k, n)
        out.append(dict(variant=v, n=n, pick=100 * d.pick.mean(), place_given_pick=100 * d.place[d.pick].mean() if d.pick.any() else 0,
                        overall=100 * d.success.mean(), lo=100 * lo, hi=100 * hi,
                        partial=50 * (d.pick.mean() + d.place.mean()),
                        steps=d.steps[d.success].mean() if d.success.any() else np.nan,
                        latency=d.latency_ms.mean(), travel=d.base_travel.mean()))
    t = pd.DataFrame(out)
    lines = ["\\begin{tabular}{lrrrrrr}\\toprule",
             "Method & Pick SR & Overall SR (95\\% CI) & Partial SR & Steps$^\\dagger$ & Latency (ms) \\\\\\midrule"]
    for _, r in t.iterrows():
        name = VAR[r.variant]
        if r.variant in ("bilevel_wo_reach", "bilevel_feasible_select"):
            lines.append("\\midrule")
        steps = "--" if np.isnan(r.steps) else f"{r.steps:.0f}"
        lines.append(f"{name} & {r.pick:.0f} & {r.overall:.0f} ({r.lo:.0f}--{r.hi:.0f}) & {r.partial:.0f} & "
                     f"{steps} & {r.latency:.0f} \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    (TAB / f"momanip_{tag}.tex").write_text("\n".join(lines))
    t.to_csv(RES / "momanipvla" / f"table_{tag}.csv", index=False)
    return df


def mcnemar(df, a, b):
    """Exact McNemar test on paired episodes (same seeds): returns (b_only, a_only, p)."""
    from math import comb
    p = df.pivot_table(index="seed", columns="variant", values="success", aggfunc="first")
    if a not in p or b not in p:
        return None
    n10 = int(((p[a] == True) & (p[b] == False)).sum())   # noqa: E712
    n01 = int(((p[a] == False) & (p[b] == True)).sum())   # noqa: E712
    n = n10 + n01
    k = min(n10, n01)
    pval = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return n10, n01, pval


def failure_table(df, tag="main"):
    if df is None:
        return
    d = df[~df.success].copy()
    d["stage"] = d.fail_reason.str.split(":").str[0]
    d["kind"] = d.fail_reason.str.split(":").str[1].str.split("(").str[0]
    d["base_hit"] = d.fail_reason.str.contains(r"\(base")
    cats = {"unreachable": "IK unreachable", "collision": "collision", "misaligned": "grasp misaligned (VLA error)",
            "missed_bowl": "place missed bowl"}
    keep = ["fixed_base_vla", "heuristic_base", "momanip_bilevel", "bilevel_feasible_select"]
    lines = ["\\begin{tabular}{l" + "r" * len(keep) + "}\\toprule",
             "Failure cause (\\% of episodes) & Fixed base & Heuristic & Bi-level (paper) & Bi-level + FF (ours) \\\\\\midrule"]
    n_by = df.groupby("variant").size()
    for kind, label in cats.items():
        vals = [100 * ((d.variant == k) & (d.kind == kind)).sum() / n_by.get(k, 1) for k in keep]
        lines.append(f"{label} & " + " & ".join(f"{v:.0f}" for v in vals) + " \\\\")
    vals = [100 * ((d.variant == k) & d.base_hit).sum() / n_by.get(k, 1) for k in keep]
    lines.append("\\quad of which base--furniture collision & " + " & ".join(f"{v:.0f}" for v in vals) + " \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    (TAB / f"failures_{tag}.tex").write_text("\n".join(lines))


def policy_table():
    p = RES / "hommi_policy" / "offline_metrics.json"
    if not p.exists():
        return
    m = json.loads(p.read_text())
    rows = [("Left gripper position (cm)", "left_pos_cm"), ("Right gripper position (cm)", "right_pos_cm"),
            ("Left gripper rotation (deg)", "left_rot_deg"), ("Right gripper rotation (deg)", "right_rot_deg"),
            ("Gripper width, left (mm)", "left_width_mm"), ("Look-at point (cm)", "lookat_cm")]
    hs = [h for h in (k.split("t=")[1] for k in m if k.startswith("policy/lookat_cm/t=")) if h in ("0.40s", "1.55s")]
    lines = ["\\begin{tabular}{l" + "rr" * (len(hs) + 1) + "}\\toprule",
             "Error & \\multicolumn{2}{c}{chunk mean} & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$t$={h}}}" for h in hs) + " \\\\",
             " & policy & hold & " + " & ".join("policy & hold" for _ in hs) + " \\\\\\midrule"]
    for label, k in rows:
        vals = [m[f"policy/{k}/mean"], m[f"hold_still/{k}/mean"]]
        for h in hs:
            vals += [m[f"policy/{k}/t={h}"], m[f"hold_still/{k}/t={h}"]]
        lines.append(label + " & " + " & ".join(f"{v:.1f}" for v in vals) + " \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    (TAB / "policy.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    wbc_table()
    df = momanip_table("main")
    failure_table(df, "main")
    if df is not None:
        tests = {f"{a} vs {b}": mcnemar(df, a, b) for a, b in (
            ("bilevel_feasible_select", "momanip_bilevel"), ("momanip_direct", "momanip_bilevel"),
            ("momanip_bilevel", "heuristic_base"), ("momanip_bilevel", "bilevel_wo_smooth"))}
        (RES / "momanipvla" / "mcnemar_main.json").write_text(json.dumps(tests, indent=1))
        print(tests)
    dl = momanip_table("learned")
    failure_table(dl, "learned")
    policy_table()
    for f in sorted(TAB.glob("*.tex")):
        print("==", f.name)
        print(f.read_text())
