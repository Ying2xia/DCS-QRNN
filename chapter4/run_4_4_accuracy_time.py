#!/usr/bin/env python3
"""Chapter 4: CPU timing tables merged with main-experiment accuracy.

The main 4-2/4-3 experiments may be run on GPU to obtain MAE/RMSE quickly.
For the paper's computational-efficiency tables, however, CPU running times
are more comparable to the original RACT results. This script therefore:

1. reads MAE/RMSE from the completed 4-2 and 4-3 summary CSV files;
2. runs only CPU timing replications for selected representative settings;
3. writes merged table-ready CSV files using main-experiment accuracy and
   CPU-only running time.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np

from _common import ROOT, parse_methods
from common.config import (
    DEFAULT_H,
    DEFAULT_J,
    DEFAULT_K,
    DEFAULT_LAMBDA,
    DEFAULT_SIM_N,
    DEFAULT_SIM_N_TEST,
    DEFAULT_T,
)
from common.data import GEN
from common.experiments import (
    run_catboost,
    run_csqrnn,
    run_dcsqrnn,
    run_elm,
    run_fedavg,
    run_linear_qr,
    run_qrnn,
    run_rvfl,
)
from common.metrics import summarise
from common.storage import load_csv, save_csv, save_json
from common.training import load_hyperparameter_map, parameter_count, resolve_hyperparams


CENTRALIZED_METHODS = {"csqrnn", "qrnn", "catboost", "elm", "rvfl", "qr"}
DISTRIBUTED_METHODS = {"dcsqrnn", "fedavg"}


def _float_key(value) -> float:
    return round(float(value), 10)


def _pilot_key(value) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.10g}"


def _load_accuracy_lookup(centralized_summary: str | None,
                          distributed_summary: str | None) -> dict[tuple, dict]:
    lookup: dict[tuple, dict] = {}

    if centralized_summary:
        for row in load_csv(centralized_summary):
            key = (
                row["scenario"],
                row["dist"],
                _float_key(row["tau"]),
                row["method"],
                "",
            )
            lookup[key] = row

    if distributed_summary:
        for row in load_csv(distributed_summary):
            key = (
                row["scenario"],
                row["dist"],
                _float_key(row["tau"]),
                row["method"],
                _pilot_key(row.get("pilot_ratio", "")),
            )
            lookup[key] = row

    return lookup


def _accuracy_values(lookup: dict[tuple, dict], scenario: str, dist: str,
                     tau: float, method: str, pilot_ratio="") -> dict:
    key = (scenario, dist, _float_key(tau), method, _pilot_key(pilot_ratio))
    row = lookup.get(key)
    if row is None and method == "csqrnn":
        row = lookup.get((scenario, dist, _float_key(tau), method, ""))
    if row is None:
        return {
            "mae_mean": "",
            "mae_std": "",
            "rmse_mean": "",
            "rmse_std": "",
            "accuracy_source": "missing",
        }
    return {
        "mae_mean": row.get("mae_mean", ""),
        "mae_std": row.get("mae_std", ""),
        "rmse_mean": row.get("rmse_mean", ""),
        "rmse_std": row.get("rmse_std", ""),
        "accuracy_source": "main_summary",
    }


def _run_one_timing(method: str, X_tr, y_tr, X_te, Q_te,
                    tau: float, J: int, lam: float, h: float,
                    K: int, pilot_ratio: float, T: int, seed: int,
                    maxiter: int, cfg: dict):
    if method == "csqrnn":
        res, _ = run_csqrnn(
            X_tr, y_tr, X_te, Q_te, tau, J, lam, h, seed,
            maxiter=maxiter, backend="numpy", device="cpu",
        )
    elif method == "dcsqrnn":
        res = run_dcsqrnn(
            X_tr, y_tr, X_te, Q_te, tau, J, lam, h, K, pilot_ratio, T, seed,
            maxiter_ref=maxiter, backend="numpy", device="cpu",
        )
    elif method == "fedavg":
        res = run_fedavg(X_tr, y_tr, X_te, Q_te, tau, J, lam, K, seed, cfg)
    elif method == "qrnn":
        res = run_qrnn(X_tr, y_tr, X_te, Q_te, tau, J, lam, seed)
    elif method == "catboost":
        res = run_catboost(X_tr, y_tr, X_te, Q_te, tau, seed, cfg)
    elif method == "elm":
        res = run_elm(X_tr, y_tr, X_te, Q_te, tau, seed, cfg)
    elif method == "rvfl":
        res = run_rvfl(X_tr, y_tr, X_te, Q_te, tau, seed, cfg)
    elif method == "qr":
        res = run_linear_qr(X_tr, y_tr, X_te, Q_te, tau)
    else:
        raise ValueError(method)
    return res


def _timing_rows_for_setting(methods: Iterable[str], scenario: str, dist: str,
                             tau: float, reps: int, args, cfg: dict,
                             J_map, lam_map, fallback_J, fallback_lam,
                             pilot_ratio: float | None = None,
                             pilots: Iterable[float] | None = None,
                             K_override: int | None = None):
    gen_fn, _ = GEN[scenario]
    rows = []
    pilots = list(pilots or [])

    for rep in range(int(reps)):
        seed = (
            int(args.seed)
            + 100000 * ["ex21", "ex22", "ex23"].index(scenario)
            + 10000 * ["N01", "t3", "chi2_2"].index(dist)
            + 100 * int(100 * tau)
            + rep
        )
        print(f"[timing] {scenario}, {dist}, tau={tau}, rep={rep + 1}/{reps}")
        X_tr, y_tr, _ = gen_fn(args.N, tau, dist, np.random.default_rng(seed))
        X_te, _, Q_te = gen_fn(args.N_test, tau, dist, np.random.default_rng(seed + 1_000_000))
        J, lam = resolve_hyperparams(J_map, lam_map, fallback_J, fallback_lam, tau, scenario, dist)
        K_value = int(K_override or args.K)

        for method in methods:
            method_pilots = pilots if method == "dcsqrnn" and pilots else [pilot_ratio]
            for pr in method_pilots:
                if method != "dcsqrnn":
                    pr = ""
                try:
                    res = _run_one_timing(
                        method, X_tr, y_tr, X_te, Q_te, tau, J, lam, args.h,
                        K_value, float(pr or args.pilot_ratio), args.T, seed,
                        args.maxiter, cfg,
                    )
                    rows.append({
                        "scenario": scenario,
                        "dist": dist,
                        "tau": tau,
                        "rep": rep + 1,
                        "method": method,
                        "pilot_ratio": pr,
                        "K": K_value,
                        "J": J,
                        "lambda": lam,
                        "cpu_time": res["time"],
                        "timing_mae": res.get("mae", ""),
                        "timing_rmse": res.get("rmse", ""),
                    })
                except ImportError as exc:
                    print(f"  skipped {method}: {exc}")
    return rows


def _merge_timing_with_accuracy(timing_summary: list[dict],
                                accuracy_lookup: dict[tuple, dict]) -> list[dict]:
    rows = []
    for row in timing_summary:
        scenario = row["scenario"]
        dist = row["dist"]
        tau = float(row["tau"])
        method = row["method"]
        pilot = row.get("pilot_ratio", "")
        merged = {
            "scenario": scenario,
            "dist": dist,
            "tau": tau,
            "method": method,
            "pilot_ratio": pilot,
            "K": row.get("K", ""),
            **_accuracy_values(accuracy_lookup, scenario, dist, tau, method, pilot),
            "cpu_time_mean": row.get("cpu_time_mean", ""),
            "cpu_time_std": row.get("cpu_time_std", ""),
            "timing_reps": row.get("n", ""),
        }
        rows.append(merged)
    return rows


def _add_cpu_ract(rows: list[dict]) -> list[dict]:
    by_setting = {}
    for row in rows:
        if row["method"] == "csqrnn":
            by_setting[(row["scenario"], row["dist"], row["tau"])] = float(row["cpu_time_mean"])
    out = []
    for row in rows:
        item = dict(row)
        if row["method"] == "dcsqrnn":
            full_time = by_setting.get((row["scenario"], row["dist"], row["tau"]))
            if full_time is not None and row.get("cpu_time_mean") not in ("", None):
                item["cpu_ract"] = full_time / max(float(row["cpu_time_mean"]), 1e-12)
            else:
                item["cpu_ract"] = ""
        else:
            item["cpu_ract"] = ""
        out.append(item)
    return out


def _communication_volume_mb(K: int, p: int, J: int, rounds: int) -> float:
    d = parameter_count(p, J)
    # One broadcast plus one upload per worker per communication round.
    return float(2 * int(K) * int(d) * 8 * int(rounds) / (1024 ** 2))


def _latency_wall_clock_rows(timing_summary: list[dict], scenario: str, dist: str,
                             tau: float, K: int, J: int, p: int, args) -> list[dict]:
    latency_ms = [float(v) for v in args.latency_grid_ms.split(",") if v.strip()]
    bandwidth_mbps = float(args.bandwidth_mbps)
    rows = []
    for row in timing_summary:
        method = row["method"]
        if method not in {"dcsqrnn", "fedavg"}:
            continue
        rounds = int(args.T if method == "dcsqrnn" else args.fedavg_rounds)
        comp_time = float(row["cpu_time_mean"])
        volume_mb = _communication_volume_mb(K, p, J, rounds)
        transfer_time = volume_mb / max(bandwidth_mbps, 1e-12)
        item = {
            "scenario": scenario,
            "dist": dist,
            "tau": tau,
            "method": method,
            "pilot_ratio": row.get("pilot_ratio", ""),
            "K": K,
            "communication_rounds": rounds,
            "communication_volume_mb": volume_mb,
            "computation_time": comp_time,
            "bandwidth_mbps": bandwidth_mbps,
        }
        for latency in latency_ms:
            latency_s = latency / 1000.0
            item[f"wall_clock_L{int(latency)}ms"] = comp_time + rounds * latency_s + transfer_time
        rows.append(item)
    return rows


def _worker_scalability_table(worker_summary: list[dict], rounds: int) -> list[dict]:
    rows = []
    for row in worker_summary:
        rows.append({
            "K": int(row["K"]),
            "scenario": row["scenario"],
            "dist": row["dist"],
            "tau": row["tau"],
            "pilot_ratio": row.get("pilot_ratio", ""),
            "mae_mean": row.get("timing_mae_mean", ""),
            "mae_std": row.get("timing_mae_std", ""),
            "rmse_mean": row.get("timing_rmse_mean", ""),
            "rmse_std": row.get("timing_rmse_std", ""),
            "cpu_time_mean": row.get("cpu_time_mean", ""),
            "cpu_time_std": row.get("cpu_time_std", ""),
            "communication_rounds": int(rounds),
            "timing_reps": row.get("n", ""),
        })
    return sorted(rows, key=lambda r: r["K"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["ex21", "ex22", "ex23"], default="ex23")
    parser.add_argument("--dist", choices=["N01", "t3", "chi2_2"], default="t3")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--N", type=int, default=DEFAULT_SIM_N)
    parser.add_argument("--N-test", type=int, default=DEFAULT_SIM_N_TEST)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--J", type=int, default=DEFAULT_J)
    parser.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--pilot-ratio", type=float, default=0.05,
                        help="DCS pilot ratio for the accuracy-time baseline table.")
    parser.add_argument("--absolute-pilot-ratios", default="0.01,0.05,0.10",
                        help="DCS pilot ratios for the absolute-time table.")
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--methods", default="csqrnn,dcsqrnn,qrnn,catboost,elm,rvfl,qr")
    parser.add_argument("--hyperparams", default=None)
    parser.add_argument("--centralized-summary",
                        default=str(ROOT / "results" / "chapter4_centralized" / "centralized_summary.csv"))
    parser.add_argument("--distributed-summary",
                        default=str(ROOT / "results" / "chapter4_distributed" / "distributed_summary.csv"))
    parser.add_argument("--run-absolute-time", action="store_true",
                        help="Also run CPU timing for Table absolute_time: ex21/ex22/ex23, N01, tau=0.5.")
    parser.add_argument("--run-latency-wall-clock", action="store_true",
                        help="Also run CPU timing and communication simulation for Table latency_wall_clock.")
    parser.add_argument("--run-worker-scalability", action="store_true",
                        help="Also run DCS-QRNN CPU timing for the worker-scalability table.")
    parser.add_argument("--run-all-tables", action="store_true",
                        help="Run all four timing-table outputs in Section 4.4.")
    parser.add_argument("--absolute-dist", choices=["N01", "t3", "chi2_2"], default="N01")
    parser.add_argument("--absolute-tau", type=float, default=0.5)
    parser.add_argument("--latency-grid-ms", default="10,50,100,200")
    parser.add_argument("--bandwidth-mbps", type=float, default=100.0)
    parser.add_argument("--worker-grid", default="5,10,20,50")
    parser.add_argument("--worker-pilot-ratio", type=float, default=None)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--rf-hidden", type=int, default=50)
    parser.add_argument("--rf-lam", type=float, default=1e-4)
    parser.add_argument("--rf-eps", type=float, default=0.01)
    parser.add_argument("--rf-maxiter", type=int, default=1000)
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-lr", type=float, default=0.05)
    parser.add_argument("--catboost-l2", type=float, default=3.0)
    parser.add_argument("--catboost-task-type", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--catboost-devices", default="0")
    parser.add_argument("--fedavg-rounds", type=int, default=10)
    parser.add_argument("--fedavg-local-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter4_accuracy_time"))
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    cfg = vars(args)
    J_map, lam_map, fallback_J, fallback_lam = load_hyperparameter_map(args.hyperparams, args.J, args.lam)
    accuracy_lookup = _load_accuracy_lookup(args.centralized_summary, args.distributed_summary)

    timing_rows = _timing_rows_for_setting(
        methods, args.scenario, args.dist, args.tau, args.reps, args, cfg,
        J_map, lam_map, fallback_J, fallback_lam,
        pilot_ratio=args.pilot_ratio,
    )
    timing_summary = summarise(
        timing_rows,
        ["scenario", "dist", "tau", "method", "pilot_ratio"],
        ["cpu_time", "timing_mae", "timing_rmse"],
    )
    accuracy_time = _add_cpu_ract(_merge_timing_with_accuracy(timing_summary, accuracy_lookup))

    save_csv(f"{args.out}/cpu_timing_raw.csv", timing_rows)
    save_csv(f"{args.out}/cpu_timing_summary.csv", timing_summary)
    save_csv(f"{args.out}/accuracy_time_summary.csv", accuracy_time)

    if args.run_absolute_time or args.run_all_tables:
        pilots = [float(v) for v in args.absolute_pilot_ratios.split(",") if v.strip()]
        absolute_rows = []
        for scenario in ("ex21", "ex22", "ex23"):
            absolute_rows.extend(
                _timing_rows_for_setting(
                    ["csqrnn", "dcsqrnn"], scenario, args.absolute_dist,
                    args.absolute_tau, args.reps, args, cfg,
                    J_map, lam_map, fallback_J, fallback_lam,
                    pilots=pilots,
                )
            )
        absolute_timing_summary = summarise(
            absolute_rows,
            ["scenario", "dist", "tau", "method", "pilot_ratio"],
            ["cpu_time", "timing_mae", "timing_rmse"],
        )
        absolute_time = _add_cpu_ract(_merge_timing_with_accuracy(absolute_timing_summary, accuracy_lookup))
        save_csv(f"{args.out}/absolute_time_raw.csv", absolute_rows)
        save_csv(f"{args.out}/absolute_time_summary.csv", absolute_time)

    if args.run_latency_wall_clock or args.run_all_tables:
        latency_rows = _timing_rows_for_setting(
            ["dcsqrnn", "fedavg"], args.scenario, args.dist, args.tau,
            args.reps, args, cfg, J_map, lam_map, fallback_J, fallback_lam,
            pilot_ratio=args.pilot_ratio,
        )
        latency_timing_summary = summarise(
            latency_rows,
            ["scenario", "dist", "tau", "method", "pilot_ratio", "K"],
            ["cpu_time", "timing_mae", "timing_rmse"],
        )
        gen_fn, p = GEN[args.scenario]
        J, _ = resolve_hyperparams(
            J_map, lam_map, fallback_J, fallback_lam,
            args.tau, args.scenario, args.dist,
        )
        latency_table = _latency_wall_clock_rows(
            latency_timing_summary, args.scenario, args.dist, args.tau,
            args.K, J, p, args,
        )
        save_csv(f"{args.out}/latency_timing_raw.csv", latency_rows)
        save_csv(f"{args.out}/latency_timing_summary.csv", latency_timing_summary)
        save_csv(f"{args.out}/latency_wall_clock.csv", latency_table)

    if args.run_worker_scalability or args.run_all_tables:
        worker_rows = []
        worker_pilot = float(args.worker_pilot_ratio if args.worker_pilot_ratio is not None else args.pilot_ratio)
        for K_value in [int(v) for v in args.worker_grid.split(",") if v.strip()]:
            worker_rows.extend(
                _timing_rows_for_setting(
                    ["dcsqrnn"], args.scenario, args.dist, args.tau,
                    args.reps, args, cfg, J_map, lam_map, fallback_J, fallback_lam,
                    pilot_ratio=worker_pilot,
                    K_override=K_value,
                )
            )
        worker_summary = summarise(
            worker_rows,
            ["scenario", "dist", "tau", "method", "pilot_ratio", "K"],
            ["cpu_time", "timing_mae", "timing_rmse"],
        )
        save_csv(f"{args.out}/worker_scalability_raw.csv", worker_rows)
        save_csv(f"{args.out}/worker_scalability_summary.csv", _worker_scalability_table(worker_summary, args.T))

    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved CPU timing and merged accuracy-time outputs to {args.out}")


if __name__ == "__main__":
    main()
