"""Report figures from the CSV results (matplotlib, static PDF/PNG for the LaTeX report).

  python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT.parent / "report" / "figures" / "part2"

# validated categorical slots 1-3 (dataviz reference palette, light mode) + ink / grid tokens
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
MUTED = "#b9b8b3"
MODE_STYLE = {"direct_lookat": ("HoMMI look-at point", C1), "ik": ("6-DoF head pose in IK", C2),
              "hold_init": ("No active neck", C3)}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "legend.frameon": False, "axes.axisbelow": True,
})


def wbc_offset_figure(tag, title):
    path = RES / "hommi_wbc" / f"summary_{tag}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    g = df[df["align"] == "gripper_centric"].sort_values("head_offset_m")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), layout="constrained")
    panels = [("ee_pos_err_p95_mm", "Hand tracking error, p95 (mm, log)"), ("gaze_err_deg", "Gaze error, mean (deg)")]
    for ax, (col, ylab) in zip(axes, panels):
        for mode in ("hold_init", "ik", "direct_lookat"):           # look-at drawn last (on top)
            label, color = MODE_STYLE[mode]
            s = g[g["mode"] == mode]
            x = 100 * s.head_offset_m
            ax.plot(x, s[col], color=color, lw=2, solid_capstyle="round", label=label, zorder=3)
            ax.scatter(x, s[col], s=36, color=color, edgecolor=SURF, linewidth=1.5, zorder=4)
        ax.set_xlabel("Demonstrator head height offset (cm)")
        ax.set_ylabel(ylab)
        ax.set_xticks(sorted((100 * g.head_offset_m).unique()))
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.05, 500)
    lo = g[g["mode"] == "direct_lookat"]["ee_pos_err_p95_mm"].max()
    axes[0].annotate("look-at = no-neck\n(head not in the IK)", xy=(0, lo), xytext=(2, lo * 12), fontsize=7,
                     color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[1].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(MODE_STYLE[m][0]) for m in MODE_STYLE]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside upper center", ncol=3,
               fontsize=8, handlelength=1.6, labelcolor=INK, title=title, title_fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"wbc_offsets_{tag}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


VARIANT_LABEL = {
    "fixed_base_vla": "Fixed-base VLA", "heuristic_base": "Heuristic base placement",
    "momanip_direct": "MoManipVLA, direct search", "momanip_bilevel": "MoManipVLA, bi-level (paper)",
    "bilevel_wo_reach": "Bi-level w/o reachability", "bilevel_wo_smooth": "Bi-level w/o smoothness",
    "bilevel_wo_collision": "Bi-level w/o collision", "bilevel_feasible_select": "Bi-level + feasibility-first (ours)",
    "direct_feasible_select": "Direct + feasibility-first (ours)",
}


def momanip_figure(tag="main"):
    path = RES / "momanipvla" / f"summary_{tag}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    order = [v for v in VARIANT_LABEL if v in set(df.variant)]
    df = df.set_index("variant").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    y = range(len(df))[::-1]
    highlight = {"momanip_bilevel": C1, "bilevel_feasible_select": C1}
    colors = [highlight.get(v, MUTED) for v in df.variant]
    ax = axes[0]
    ax.barh(list(y), 100 * df.overall_sr, height=0.55, color=colors, zorder=3)
    ci_path = RES / "momanipvla" / f"table_{tag}.csv"
    ci = pd.read_csv(ci_path).set_index("variant").loc[df.variant] if ci_path.exists() else None
    for i, (yi, v) in enumerate(zip(y, 100 * df.overall_sr)):
        x_text = v + 1
        if ci is not None:
            lo, hi = ci.lo.iloc[i], ci.hi.iloc[i]
            ax.plot([lo, hi], [yi, yi], color=INK2, lw=0.9, zorder=4)
            x_text = hi + 1
        ax.text(x_text, yi, f"{v:.0f}%", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels([VARIANT_LABEL[v] for v in df.variant], color=INK)
    ax.set_xlabel("Overall success rate (%), 95% CI")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ax.barh(list(y), df.latency_ms / 1000, height=0.55, color=colors, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([])
    ax.set_xlabel("Latency / waypoint (s)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"momanip_{tag}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def training_figure():
    import glob
    import json
    runs = sorted(glob.glob(str(ROOT / "outputs" / "runs" / "*" / "*small_vits")))
    src = Path(runs[-1]) / "logs.json.txt" if runs else RES / "hommi_policy" / "train_log.json.txt"
    if not src.exists():
        return
    rows = [json.loads(line) for line in open(src)]
    df = pd.DataFrame(rows)
    (RES / "hommi_policy").mkdir(parents=True, exist_ok=True)
    if runs:   # keep a copy of the log next to the results (outputs/ is not versioned)
        (RES / "hommi_policy" / "train_log.json.txt").write_text(open(src).read())
    ep = df.groupby("epoch").agg(train_loss=("train_loss", "mean")).reset_index()
    val = df.dropna(subset=["val/action_mse_error"]) if "val/action_mse_error" in df else None
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), layout="constrained")
    axes[0].plot(ep.epoch, ep.train_loss, color=C1, lw=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Diffusion loss (train, log)")
    if val is not None and len(val):
        for col, label, color in (("val/action_mse_error", "all actions", C1),
                                  ("val/camera_head_lookatpoint", "look-at point", C3)):
            if col in val:
                axes[1].plot(val.epoch, val[col], color=color, lw=2, label=label)
                axes[1].scatter(val.epoch, val[col], s=30, color=color, edgecolor=SURF, linewidth=1.5, zorder=4)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val action MSE (1 batch, raw units, log)")
        axes[1].legend(fontsize=7.5, labelcolor=INK)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"hommi_training.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def noise_figure():
    pn, pm = RES / "momanipvla" / "episodes_noise.csv", RES / "momanipvla" / "episodes_main.csv"
    if not (pn.exists() and pm.exists()):
        return
    dn = pd.read_csv(pn)
    dm = pd.read_csv(pm)
    dm = dm[dm.seed.isin(dn.seed.unique())].copy()
    dm["vla"] = "oracle:0.008"
    dm = dm.drop(columns=[c for c in ("vla_sigma",) if c in dm])
    d = pd.concat([dn, dm[dm.variant.isin(dn.variant.unique())]])
    d["sigma_mm"] = d.vla.str.split(":").str[1].astype(float) * 1000
    g = d.groupby(["variant", "sigma_mm"]).success.mean().reset_index()
    fig, ax = plt.subplots(figsize=(3.4, 2.3), layout="constrained")
    styles = {"heuristic_base": ("Heuristic base", MUTED), "momanip_bilevel": ("Bi-level (paper)", C2),
              "bilevel_feasible_select": ("Bi-level + FF (ours)", C1)}
    for v, (label, color) in styles.items():
        s = g[g.variant == v].sort_values("sigma_mm")
        ax.plot(s.sigma_mm, 100 * s.success, color=color, lw=2, label=label)
        ax.scatter(s.sigma_mm, 100 * s.success, s=30, color=color, edgecolor=SURF, linewidth=1.5, zorder=4)
    ax.set_xlabel("VLA waypoint error $\\sigma$ per axis (mm)")
    ax.set_ylabel("Overall success (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, labelcolor=INK)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"momanip_noise.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    noise_figure()
    training_figure()
    wbc_offset_figure("cup1", "Tabletop demos (rby1-wbc cup1, 25 episodes)")
    wbc_offset_figure("laundry", "HoMMI Laundry demos (30 episodes, look-at from LiDAR depth)")
    momanip_figure("main")
    print("figures in", FIG)
