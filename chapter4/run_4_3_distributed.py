#!/usr/bin/env python3
"""Chapter 4: distributed simulation comparison.

Methods: CS-QRNN reference, DCS-QRNN at 1/5/10% pilot ratios, and FedAvg-QRNN.
"""

from __future__ import annotations

import argparse
import numpy as np

from _common import ROOT, add_sim_filter_args, parse_methods
from common.config import DEFAULT_H, DEFAULT_J, DEFAULT_K, DEFAULT_LAMBDA, DEFAULT_SIM_N, DEFAULT_SIM_N_TEST, DEFAULT_T, PILOT_RATIOS, SIM_DISTS, SIM_SCENARIOS, TAUS
from common.data import GEN
from common.experiments import run_csqrnn, run_dcsqrnn, run_fedavg
from common.metrics import summarise
from common.storage import load_csv, save_csv, save_json
from common.training import load_hyperparameter_map, resolve_hyperparams


def _to_float(value, default=""):
    if value in ("", None):
        return default
    return float(value)


def _to_int(value, default=""):
    if value in ("", None):
        return default
    return int(float(value))


def _load_csqrnn_rows(path):
    if not path:
        return {}
    rows = {}
    for row in load_csv(path):
        if row.get("method") != "csqrnn":
            continue
        key = (row["scenario"], row["dist"], float(row["tau"]), int(float(row["rep"])))
        rows[key] = {
            "mae": _to_float(row.get("mae")),
            "rmse": _to_float(row.get("rmse")),
            "time": _to_float(row.get("time")),
            "converged": row.get("converged", ""),
            "nit": _to_int(row.get("nit")),
            "backend": row.get("backend", ""),
            "device": row.get("device", ""),
        }
    return rows


def _print_setting_summary(sc: str, dist: str, tau: float, rows: list[dict]):
    if not rows:
        return
    summary = summarise(rows, ["method", "pilot_ratio"], ["mae", "rmse", "time", "ract"])
    print(f"\nFinished setting: {sc}, {dist}, tau={tau}")
    print(f"{'method':<12} {'pilot':>7} {'n':>3} {'MAE mean':>12} {'MAE sd':>12} {'RMSE mean':>12} {'RMSE sd':>12} {'RACT mean':>12}")
    for row in summary:
        pilot = row["pilot_ratio"] if row["pilot_ratio"] not in ("", None) else "-"
        ract = row["ract_mean"]
        ract_text = f"{ract:.3f}" if np.isfinite(ract) else "-"
        print(
            f"{row['method']:<12} {str(pilot):>7} {int(row['n']):>3d} "
            f"{row['mae_mean']:>12.6f} {row['mae_std']:>12.6f} "
            f"{row['rmse_mean']:>12.6f} {row['rmse_std']:>12.6f} "
            f"{ract_text:>12}"
        )
    print("", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sim_filter_args(parser)
    parser.add_argument("--N", type=int, default=DEFAULT_SIM_N)
    parser.add_argument("--N-test", type=int, default=DEFAULT_SIM_N_TEST)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--J", type=int, default=DEFAULT_J)
    parser.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--hyperparams", default=None)
    parser.add_argument("--methods", default="csqrnn,dcsqrnn,fedavg")
    parser.add_argument("--csqrnn-raw", default=None,
                        help="Optional 4-2 centralized_raw.csv. When supplied, CS-QRNN rows are reused instead of refit.")
    parser.add_argument("--pilot-ratios", default="0.01,0.05,0.10")
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--device", default="auto", help="Use cuda on a GPU server, or auto.")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--torch-lr", type=float, default=0.01)
    parser.add_argument("--torch-check-every", type=int, default=25)
    parser.add_argument("--torch-maxiter", type=int, default=800)
    parser.add_argument("--torch-dcs-step-maxiter", type=int, default=400)
    parser.add_argument("--torch-lbfgs-steps", type=int, default=400)
    parser.add_argument("--torch-lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--torch-lbfgs-history-size", type=int, default=20)
    parser.add_argument("--torch-fedavg-lbfgs-steps", type=int, default=0)
    parser.add_argument("--fedavg-rounds", type=int, default=10)
    parser.add_argument("--fedavg-local-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter4_distributed"))
    args = parser.parse_args()

    scenarios = list(SIM_SCENARIOS) if args.scenario == "all" else [args.scenario]
    dists = list(SIM_DISTS) if args.dist == "all" else [args.dist]
    taus = list(TAUS) if args.all_taus else [args.tau]
    methods = parse_methods(args.methods)
    pilots = [float(v) for v in args.pilot_ratios.split(",") if v.strip()] or list(PILOT_RATIOS)
    J_map, lam_map, fallback_J, fallback_lam = load_hyperparameter_map(args.hyperparams, args.J, args.lam)
    csqrnn_map = _load_csqrnn_rows(args.csqrnn_raw)
    cfg = vars(args)

    raw_rows = []
    total = len(scenarios) * len(dists) * len(taus) * args.reps
    done = 0
    for sc in scenarios:
        gen_fn, _ = GEN[sc]
        for dist in dists:
            for tau in taus:
                cell_rows = []
                for rep in range(args.reps):
                    done += 1
                    seed = args.seed + 100000 * SIM_SCENARIOS.index(sc) + 10000 * SIM_DISTS.index(dist) + 100 * int(100 * tau) + rep
                    print(f"[{done}/{total}] distributed: {sc}, {dist}, tau={tau}, rep={rep + 1}")
                    X_tr, y_tr, _ = gen_fn(args.N, tau, dist, np.random.default_rng(seed))
                    X_te, _, Q_te = gen_fn(args.N_test, tau, dist, np.random.default_rng(seed + 1_000_000))
                    J, lam = resolve_hyperparams(J_map, lam_map, fallback_J, fallback_lam, tau, sc, dist)
                    cs_time = None
                    if "csqrnn" in methods or "dcsqrnn" in methods:
                        if csqrnn_map:
                            key = (sc, dist, float(tau), rep + 1)
                            if key not in csqrnn_map:
                                raise KeyError(f"Missing CS-QRNN row in {args.csqrnn_raw}: {key}")
                            res = dict(csqrnn_map[key])
                        else:
                            res, _ = run_csqrnn(
                                X_tr, y_tr, X_te, Q_te, tau, J, lam, args.h, seed,
                                maxiter=args.maxiter,
                                backend=args.backend,
                                device=args.device,
                                torch_dtype=args.torch_dtype,
                                torch_lr=args.torch_lr,
                                torch_check_every=args.torch_check_every,
                                torch_maxiter=args.torch_maxiter,
                                torch_lbfgs_steps=args.torch_lbfgs_steps,
                                torch_lbfgs_lr=args.torch_lbfgs_lr,
                                torch_lbfgs_history_size=args.torch_lbfgs_history_size,
                            )
                        cs_time = res["time"]
                        row = {"scenario": sc, "dist": dist, "tau": tau, "rep": rep + 1, "method": "csqrnn", "pilot_ratio": "", "J": J, "lambda": lam, **res}
                        raw_rows.append(row)
                        cell_rows.append(row)
                    if "dcsqrnn" in methods:
                        for pr in pilots:
                            res = run_dcsqrnn(
                                X_tr, y_tr, X_te, Q_te, tau, J, lam, args.h,
                                args.K, pr, args.T, seed,
                                maxiter_ref=args.maxiter,
                                backend=args.backend,
                                device=args.device,
                                torch_dtype=args.torch_dtype,
                                torch_lr=args.torch_lr,
                                torch_check_every=args.torch_check_every,
                                torch_maxiter=args.torch_maxiter,
                                torch_dcs_step_maxiter=args.torch_dcs_step_maxiter,
                                torch_lbfgs_steps=args.torch_lbfgs_steps,
                                torch_lbfgs_lr=args.torch_lbfgs_lr,
                                torch_lbfgs_history_size=args.torch_lbfgs_history_size,
                            )
                            res["ract"] = float(cs_time / max(res["time"], 1e-9)) if cs_time is not None else ""
                            row = {"scenario": sc, "dist": dist, "tau": tau, "rep": rep + 1, "method": "dcsqrnn", "pilot_ratio": pr, "J": J, "lambda": lam, **res}
                            raw_rows.append(row)
                            cell_rows.append(row)
                    if "fedavg" in methods:
                        res = run_fedavg(X_tr, y_tr, X_te, Q_te, tau, J, lam, args.K, seed, cfg)
                        row = {"scenario": sc, "dist": dist, "tau": tau, "rep": rep + 1, "method": "fedavg", "pilot_ratio": "", "J": J, "lambda": lam, **res}
                        raw_rows.append(row)
                        cell_rows.append(row)
                _print_setting_summary(sc, dist, tau, cell_rows)

    summary = summarise(raw_rows, ["scenario", "dist", "tau", "method", "pilot_ratio"], ["mae", "rmse", "time", "ract"])
    save_csv(f"{args.out}/distributed_raw.csv", raw_rows)
    save_csv(f"{args.out}/distributed_summary.csv", summary)
    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
