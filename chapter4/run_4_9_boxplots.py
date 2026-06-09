#!/usr/bin/env python3
"""Generate Chapter 4 simulation box plots with the added baselines.

The script combines the completed 4-2 centralized raw results and the 4-3
distributed raw results, then regenerates the manuscript figure files using
the existing naming convention, for example ``fig/ex230.5t3mae.png``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ROOT


SCENARIOS = ("ex21", "ex22", "ex23")
DISTS = ("N01", "t3", "chi2_2")
TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)
METRICS = ("mae", "rmse")

METHOD_ORDER = (
    "csqrnn",
    "dcsqrnn_1pct",
    "dcsqrnn_5pct",
    "dcsqrnn_10pct",
    "qrnn",
    "fedavg",
    "catboost",
    "elm",
    "rvfl",
    "qr",
)

METHOD_LABELS = {
    "csqrnn": "CS-\nQRNN",
    "dcsqrnn_1pct": "DCS-\nQRNN\n1%",
    "dcsqrnn_5pct": "DCS-\nQRNN\n5%",
    "dcsqrnn_10pct": "DCS-\nQRNN\n10%",
    "qrnn": "QRNN",
    "fedavg": "FedAvg\nQRNN",
    "catboost": "CatBoost",
    "elm": "ELM-QR",
    "rvfl": "RVFL-QR",
    "qr": "QR",
}

METHOD_COLORS = {
    "csqrnn": "#4C78A8",
    "dcsqrnn_1pct": "#F58518",
    "dcsqrnn_5pct": "#54A24B",
    "dcsqrnn_10pct": "#B279A2",
    "qrnn": "#72B7B2",
    "fedavg": "#E45756",
    "catboost": "#FF9DA6",
    "elm": "#9D755D",
    "rvfl": "#BAB0AC",
    "qr": "#D67195",
}


def _parse_list(value: str, choices: tuple[str, ...]) -> list[str]:
    if value == "all":
        return list(choices)
    out = [item.strip() for item in value.split(",") if item.strip()]
    bad = sorted(set(out) - set(choices))
    if bad:
        raise ValueError(f"Unsupported values {bad}; choices are {choices} or all")
    return out


def _parse_taus(args) -> list[float]:
    if args.all_taus:
        return list(TAUS)
    if args.taus:
        return [float(item.strip()) for item in args.taus.split(",") if item.strip()]
    return [float(args.tau)]


def _tau_key(tau: float) -> str:
    return f"{float(tau):g}"


def _normalise_centralized(df: pd.DataFrame) -> pd.DataFrame:
    keep = {"csqrnn", "qrnn", "catboost", "elm", "rvfl", "qr"}
    out = df[df["method"].isin(keep)].copy()
    out["plot_method"] = out["method"]
    out["pilot_ratio"] = np.nan
    return out


def _normalise_distributed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        method = str(row["method"])
        if method == "fedavg":
            plot_method = "fedavg"
        elif method == "dcsqrnn":
            pr = float(row["pilot_ratio"])
            if np.isclose(pr, 0.01):
                plot_method = "dcsqrnn_1pct"
            elif np.isclose(pr, 0.05):
                plot_method = "dcsqrnn_5pct"
            elif np.isclose(pr, 0.10):
                plot_method = "dcsqrnn_10pct"
            else:
                continue
        else:
            continue
        new_row = row.copy()
        new_row["plot_method"] = plot_method
        rows.append(new_row)
    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["plot_method"])
    return pd.DataFrame(rows)


def load_plot_data(centralized_raw: Path, distributed_raw: Path) -> pd.DataFrame:
    if not centralized_raw.exists():
        raise FileNotFoundError(f"Centralized raw result not found: {centralized_raw}")
    if not distributed_raw.exists():
        raise FileNotFoundError(f"Distributed raw result not found: {distributed_raw}")
    centered = _normalise_centralized(pd.read_csv(centralized_raw))
    distributed = _normalise_distributed(pd.read_csv(distributed_raw))
    needed = ["scenario", "dist", "tau", "rep", "plot_method", "mae", "rmse"]
    return pd.concat([centered[needed], distributed[needed]], ignore_index=True)


def _subset_values(df: pd.DataFrame, scenario: str, dist: str, tau: float, metric: str,
                   methods: list[str]) -> tuple[list[str], list[np.ndarray]]:
    sub = df[
        (df["scenario"] == scenario)
        & (df["dist"] == dist)
        & np.isclose(df["tau"].astype(float), float(tau))
    ]
    labels = []
    values = []
    for method in methods:
        vals = sub.loc[sub["plot_method"] == method, metric].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            continue
        labels.append(method)
        values.append(vals)
    return labels, values


def plot_one(df: pd.DataFrame, scenario: str, dist: str, tau: float, metric: str,
             methods: list[str], out_dir: Path, dpi: int, fig_width: float,
             fig_height: float, label_size: int, tick_size: int,
             axis_label_size: int, box_spacing: float, bottom_margin: float) -> Path | None:
    method_keys, values = _subset_values(df, scenario, dist, tau, metric, methods)
    if not values:
        print(f"Skip empty setting: {scenario}, {dist}, tau={tau}, {metric}")
        return None

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    positions = np.arange(1, len(values) + 1) * box_spacing
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 2.0},
        boxprops={"linewidth": 1.6, "color": "black"},
        whiskerprops={"linewidth": 1.5, "color": "black"},
        capprops={"linewidth": 1.5, "color": "black"},
        flierprops={
            "marker": "o",
            "markersize": 4.5,
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "alpha": 0.85,
        },
    )

    for patch, method in zip(box["boxes"], method_keys):
        patch.set_facecolor(METHOD_COLORS.get(method, "#CCCCCC"))
        patch.set_alpha(0.85)

    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in method_keys], fontsize=label_size, linespacing=1.1)
    ax.tick_params(axis="x", pad=7)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.set_ylabel(metric.upper(), fontsize=axis_label_size, labelpad=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(positions[0] - 0.85 * box_spacing, positions[-1] + 0.85 * box_spacing)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.98, bottom=bottom_margin)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{scenario}{_tau_key(tau)}{dist}{metric}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--centralized-raw",
        default=str(ROOT / "results" / "chapter4_centralized" / "centralized_raw.csv"),
        help="Raw 4-2 centralized result CSV.",
    )
    parser.add_argument(
        "--distributed-raw",
        default=str(ROOT / "results" / "chapter4_distributed" / "distributed_raw.csv"),
        help="Raw 4-3 distributed result CSV.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT.parent / "DCSQRNN" / "fig"),
        help="Output figure directory. Defaults to the manuscript fig folder.",
    )
    parser.add_argument("--scenario", default="all", help="ex21, ex22, ex23, comma list, or all.")
    parser.add_argument("--dist", default="all", help="N01, t3, chi2_2, comma list, or all.")
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--taus", default=None, help="Comma-separated tau list. Overrides --tau.")
    parser.add_argument("--all-taus", action="store_true")
    parser.add_argument(
        "--methods",
        default=",".join(METHOD_ORDER),
        help="Comma-separated method keys to include in the box plots.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=12.0)
    parser.add_argument("--fig-height", type=float, default=5.3)
    parser.add_argument("--label-size", type=int, default=13, help="X-axis method-label font size.")
    parser.add_argument("--tick-size", type=int, default=12, help="Y-axis tick-label font size.")
    parser.add_argument("--axis-label-size", type=int, default=14, help="Axis-title font size.")
    parser.add_argument("--box-spacing", type=float, default=1.28, help="Horizontal spacing between boxes.")
    parser.add_argument("--bottom-margin", type=float, default=0.32, help="Bottom margin for method labels.")
    args = parser.parse_args()

    scenarios = _parse_list(args.scenario, SCENARIOS)
    dists = _parse_list(args.dist, DISTS)
    taus = _parse_taus(args)
    methods = _parse_list(args.methods, METHOD_ORDER)
    df = load_plot_data(Path(args.centralized_raw), Path(args.distributed_raw))
    out_dir = Path(args.out)

    saved = []
    for scenario in scenarios:
        for dist in dists:
            for tau in taus:
                for metric in METRICS:
                    out = plot_one(
                        df,
                        scenario,
                        dist,
                        tau,
                        metric,
                        methods,
                        out_dir,
                        args.dpi,
                        args.fig_width,
                        args.fig_height,
                        args.label_size,
                        args.tick_size,
                        args.axis_label_size,
                        args.box_spacing,
                        args.bottom_margin,
                    )
                    if out is not None:
                        saved.append(out)
                        print(f"Saved {out}")
    print(f"Generated {len(saved)} box-plot files in {out_dir}")


if __name__ == "__main__":
    main()
