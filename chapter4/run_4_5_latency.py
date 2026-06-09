#!/usr/bin/env python3
"""Chapter 4: simulated wall-clock time under communication latency."""

from __future__ import annotations

import argparse

from _common import ROOT
from common.storage import load_csv, save_csv, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distributed-summary", required=True,
                        help="Path to chapter4 distributed_summary.csv.")
    parser.add_argument("--latency-grid", default="0,0.01,0.05,0.10,0.50,1.00")
    parser.add_argument("--dcs-rounds", type=int, default=1)
    parser.add_argument("--fedavg-rounds", type=int, default=10)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter4_latency"))
    args = parser.parse_args()

    latencies = [float(v) for v in args.latency_grid.split(",") if v.strip()]
    rows_in = load_csv(args.distributed_summary)
    rows = []
    for row in rows_in:
        method = row["method"]
        if method not in {"dcsqrnn", "fedavg"}:
            continue
        comp_time = float(row["time_mean"])
        rounds = args.dcs_rounds if method == "dcsqrnn" else args.fedavg_rounds
        for latency in latencies:
            rows.append({
                "scenario": row.get("scenario", ""),
                "dist": row.get("dist", ""),
                "tau": row.get("tau", ""),
                "method": method,
                "pilot_ratio": row.get("pilot_ratio", ""),
                "computation_time": comp_time,
                "communication_rounds": rounds,
                "latency": latency,
                "wall_clock_time": comp_time + rounds * latency,
            })
    save_csv(f"{args.out}/latency_wall_clock.csv", rows)
    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()

