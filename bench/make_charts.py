"""Summarise results/timings.csv and render the charts used in the README and notebook.

Outputs (results/):
    summary.csv        one row per (runtime, config): mean wall time, geo-mean speedup, cores used, keypoints
    speedup.png        headline: speedup over whole-image serial, by method and tile count
    cores_used.png     diagnosis: how many CPU cores were actually busy
    time_per_image.png per-image wall time, 16 tiles
    vs_2023.png        2023 threaded times (slides) vs today's rerun on the same CPU

Usage:  python bench/make_charts.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# Validated categorical slots (blue, orange, aqua) + a neutral for the serial baselines.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
NEUTRAL = "#a3a29c"
METHODS = {  # key -> (label, color)
    "serial_tiles": ("Serial (tiles, one at a time)", NEUTRAL),
    "threads": ("Threads (GIL)", "#eb6834"),
    "procs": ("Processes", "#2a78d6"),
    "threads_nogil": ("Threads (no-GIL 3.14t)", "#1baf7a"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def load() -> pd.DataFrame:
    df = pd.read_csv(RES / "timings.csv")
    df["nogil"] = df["runtime"].str.contains("nogil")
    df["method"] = df["config"].str.replace(r"\d+$", "", regex=True)
    df.loc[df["nogil"] & (df["method"] == "threads"), "method"] = "threads_nogil"
    df.loc[df["nogil"] & (df["method"] == "serial_whole"), "method"] = "serial_whole_nogil"
    return df


def check_consistency(df: pd.DataFrame) -> None:
    """Parallel execution must not change the output: same keypoints for every N-tile config in a runtime."""
    tiled = df[df["n_tiles"] > 1]
    stalled = df[df["wall_s"] > 1.5 * df["work_s"] + 5]
    if len(stalled):
        raise SystemExit(f"stalled rows (machine paused mid-run), remove and re-run:\n{stalled}")
    bad = tiled.groupby(["runtime", "image", "n_tiles"])["kp_hash"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise SystemExit(f"keypoint mismatch across configs:\n{bad}")


def add_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Speedup of each row vs serial_whole for the same image in the same runtime."""
    base = df[df["config"] == "serial_whole"].set_index(["runtime", "image"])["wall_s"]
    df = df.join(base.rename("serial_s"), on=["runtime", "image"])
    df["speedup"] = df["serial_s"] / df["wall_s"]
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    whole_kp = df[df["config"] == "serial_whole"].set_index(["runtime", "image"])["keypoints"]
    df = df.join(whole_kp.rename("whole_kp"), on=["runtime", "image"])
    g = df.groupby(["runtime", "method", "config", "n_tiles"])
    out = g.agg(
        images=("image", "nunique"),
        mean_wall_s=("wall_s", "mean"),
        total_wall_min=("wall_s", lambda s: s.sum() / 60),
        geomean_speedup=("speedup", lambda s: float(np.exp(np.log(s).mean()))),
        min_speedup=("speedup", "min"),
        max_speedup=("speedup", "max"),
        mean_cores_used=("cores_used", "mean"),
        mean_keypoints=("keypoints", "mean"),
        keypoints_vs_whole=("keypoints", "sum"),
    ).reset_index()
    out["keypoints_vs_whole"] = out["keypoints_vs_whole"] / g["whole_kp"].sum().values
    return out.round(4)


def _bar_label(ax, bars, fmt):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color=INK)


def grouped_bars(summary: pd.DataFrame, value: str, fmt, title: str, subtitle: str, ylabel: str, refs, fname: str):
    methods = [m for m in METHODS if m in set(summary["method"])]
    tiles = [4, 16]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    width = 0.8 / len(methods)
    x = np.arange(len(tiles))
    for k, m in enumerate(methods):
        label, color = METHODS[m]
        vals = [summary.loc[(summary["method"] == m) & (summary["n_tiles"] == t), value].squeeze() for t in tiles]
        vals = [v if np.isscalar(v) else np.nan for v in vals]
        bars = ax.bar(x - 0.4 + width * (k + 0.5), vals, width - 0.03, color=color, label=label, zorder=2)
        _bar_label(ax, bars, fmt)
    for y, text, style in refs:
        ax.axhline(y, color=INK_2, lw=1, ls=style, zorder=1)
        ax.annotate(text, (1.02, y), xycoords=("axes fraction", "data"), va="center", fontsize=9, color=INK_2,
                    annotation_clip=False)
    ax.set_xticks(x, [f"{t} tiles" for t in tiles])
    ax.tick_params(axis="x", length=0, labelsize=12, labelcolor=INK)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(summary[value].max() * 1.18, max(r[0] for r in refs) * 1.08))
    fig.suptitle(title, x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.915, subtitle, ha="left", fontsize=10, color=INK_2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.08), ncol=len(methods), fontsize=9.5, handlelength=1, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RES / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)


def time_per_image(df: pd.DataFrame) -> None:
    series = [
        ("serial_whole", False, "Serial (whole image)", NEUTRAL),
        ("threads16", False, "Threads, 16 tiles (GIL)", METHODS["threads"][1]),
        ("procs16", False, "Processes, 16 tiles", METHODS["procs"][1]),
        ("threads16", True, "Threads, 16 tiles (no-GIL)", METHODS["threads_nogil"][1]),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for config, nogil, label, color in series:
        d = df[(df["config"] == config) & (df["nogil"] == nogil)].sort_values("image")
        if d.empty:
            continue
        ax.plot(d["image"], d["wall_s"], color=color, lw=2, marker="o", ms=4.5, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
    ax.set_xlabel("Image (Oxford 102 Flowers, image_00001 to image_00050)")
    ax.set_ylabel("Wall-clock seconds per image")
    ax.set_xlim(0.5, df["image"].max() + 0.5)
    ax.set_ylim(0, None)
    fig.suptitle("Every image, same story: GIL threads track serial, processes don't", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=4, fontsize=9.5, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(RES / "time_per_image.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def vs_2023(df: pd.DataFrame) -> None:
    old = pd.read_csv(RES / "original_2023.csv")
    new = df[~df["nogil"]].pivot_table(index="image", columns="config", values="wall_s")
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    pairs = [("threads4_s", "threads4", "4 tiles", METHODS["threads"][1], "o"),
             ("threads16_s", "threads16", "16 tiles", INK_2, "s")]
    hi = 0
    for old_col, new_col, label, color, marker in pairs:
        if new_col not in new:
            continue
        m = old[["image", old_col]].dropna().merge(new[new_col], left_on="image", right_index=True)
        ax.scatter(m[old_col], m[new_col], s=36, color=color, marker=marker, label=f"Threads, {label}",
                   edgecolor=SURFACE, linewidth=1, zorder=3)
        hi = max(hi, m[old_col].max(), m[new_col].max())
    ax.plot([0, hi * 1.05], [0, hi * 1.05], color=INK_2, lw=1, ls="--", zorder=1)
    ax.annotate("same time", (hi * 0.98, hi * 1.0), ha="right", va="bottom", fontsize=9, color=INK_2, rotation=40)
    ax.set_xlim(0, hi * 1.05)
    ax.set_ylim(0, hi * 1.05)
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("2023 run (Python 3.9), seconds")
    ax.set_ylabel("2026 rerun (Python 3.14), seconds")
    fig.suptitle("Threaded runs: 2023 vs today, same CPU", x=0.02, ha="left", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(RES / "vs_2023.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = add_speedup(load())
    check_consistency(df)
    summary = summarise(df)
    summary.to_csv(RES / "summary.csv", index=False)
    print(summary.to_string(index=False))

    # Speedup for the no-GIL arm is vs serial_whole on the same 3.14t interpreter.
    cores = 8
    grouped_bars(
        summary[summary["method"].isin(METHODS)], "geomean_speedup", lambda v: f"{v:.1f}×",
        "Under the GIL, threads gave no parallel speedup. Processes did.",
        "NMS-SIFT on 50 Oxford Flowers images, Ryzen 7 3700X (8 cores / 16 threads). "
        "Geometric-mean speedup over whole-image serial.",
        "Speedup vs serial (×)", [(1, "serial = 1×", "--"), (cores, f"{cores} physical cores", ":")], "speedup.png")
    grouped_bars(
        summary[summary["method"].isin(METHODS)], "mean_cores_used", lambda v: f"{v:.1f}",
        "With the GIL, every thread shares one core",
        "Mean CPU cores busy during an image (summed per-tile CPU time ÷ wall time). "
        "Above 8 means SMT siblings are in use.",
        "Cores busy (avg)", [(1, "1 core", "--"), (cores, f"{cores} physical cores", ":")], "cores_used.png")
    time_per_image(df)
    vs_2023(df)


if __name__ == "__main__":
    main()
