#!/usr/bin/env python3
"""Chapter 4: sensitivity analysis for h = c_h * sigma_hat.

This script is intended for Table tab:bandwidth_sensitivity. By default it
runs DCS-QRNN only, because the revised manuscript reports the effect of
the smoothing bandwidth on the proposed distributed estimator.
"""

from __future__ import annotations

import argparse
import numpy as np

from _common import ROOT, add_sim_filter_args, parse_methods
from common.config import (
    BANDWIDTH_C_GRID,
    DEFAULT_H,
    DEFAULT_J,
    DEFAULT_K,
    DEFAULT_LAMBDA,
    DEFAULT_REPS,
    DEFAULT_SIM_N,
    DEFAULT_SIM_N_TEST,
    DEFAULT_T,
    SIM_DISTS,
    SIM_SCENARIOS,
    TAUS,
)
from common.data import GEN
from common.experiments import run_csqrnn, run_dcsqrnn
from common.metrics import summarise
from common.storage import save_csv, save_json
from common.training import estimate_residual_sigma, load_hyperparameter_map, resolve_hyperparams


def _print_setting_summary(sc: str, dist: str, tau: float, rows: list[dict]):
    if not rows:
        return
    summary = summarise(rows, ["method", "c_h"], ["mae", "rmse", "time"])
    print(f"\nFinished bandwidth setting: {sc}, {dist}, tau={tau}")
    print(f"{'method':<10} {'c_h':>7} {'n':>3} {'MAE mean':>12} {'MAE sd':>12} {'RMSE mean':>12} {'RMSE sd':>12} {'time mean':>12}")
    for row in summary:
        print(
            f"{row['method']:<10} {float(row['c_h']):>7.2f} {int(row['n']):>3d} "
            f"{row['mae_mean']:>12.6f} {row['mae_std']:>12.6f} "
            f"{row['rmse_mean']:>12.6f} {row['rmse_std']:>12.6f} "
            f"{row['time_mean']:>12.3f}"
        )
    print("", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sim_filter_args(parser)
    parser.add_argument("--N", type=int, default=DEFAULT_SIM_N)
    parser.add_argument("--N-test", type=int, default=DEFAULT_SIM_N_TEST)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--J", type=int, default=DEFAULT_J)
    parser.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--pilot-ratio", type=float, default=0.10)
    parser.add_argument("--methods", default="dcsqrnn",
                        help="Comma-separated methods: dcsqrnn, csqrnn. Default runs only DCS-QRNN.")
    parser.add_argument("--c-grid", default=",".join(str(v) for v in BANDWIDTH_C_GRID))
    parser.add_argument("--h0", type=float, default=DEFAULT_H,
                        help="Initial bandwidth used only for estimating sigma_hat.")
    parser.add_argument("--sigma-sub", type=int, default=5000)
    parser.add_argument("--hyperparams", default=None)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--torch-lr", type=float, default=0.01)
    parser.add_argument("--torch-check-every", type=int, default=25)
    parser.add_argument("--torch-maxiter", type=int, default=800)
    parser.add_argument("--torch-dcs-step-maxiter", type=int, default=400)
    parser.add_argument("--torch-lbfgs-steps", type=int, default=400)
    parser.add_argument("--torch-lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--torch-lbfgs-history-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter4_bandwidth"))
    args = parser.parse_args()

    scenarios = list(SIM_SCENARIOS) if args.scenario == "all" else [args.scenario]
    dists = list(SIM_DISTS) if args.dist == "all" else [args.dist]
    taus = list(TAUS) if args.all_taus else [args.tau]
    c_grid = [float(v) for v in args.c_grid.split(",") if v.strip()]
    methods = parse_methods(args.methods)
    unknown = [m for m in methods if m not in {"csqrnn", "dcsqrnn"}]
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")

    J_map, lam_map, fallback_J, fallback_lam = load_hyperparameter_map(args.hyperparams, args.J, args.lam)

    rows = []
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
                    print(f"[{done}/{total}] bandwidth: {sc}, {dist}, tau={tau}, rep={rep + 1}")
                    X_tr, y_tr, _ = gen_fn(args.N, tau, dist, np.random.default_rng(seed))
                    X_te, _, Q_te = gen_fn(args.N_test, tau, dist, np.random.default_rng(seed + 1_000_000))
                    J, lam = resolve_hyperparams(J_map, lam_map, fallback_J, fallback_lam, tau, sc, dist)
                    sigma_hat = estimate_residual_sigma(
                        X_tr,
                        y_tr,
                        tau,
                        J,
                        lam,
                        np.random.default_rng(seed + 2_000_000),
                        h0=args.h0,
                        n_sub=min(args.sigma_sub, len(y_tr)),
                    )
                    for c_h in c_grid:
                        h = max(c_h * sigma_hat, 1e-4)
                        if "csqrnn" in methods:
                            cs, _ = run_csqrnn(
                                X_tr, y_tr, X_te, Q_te, tau, J, lam, h, seed,
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
                            row = {
                                "scenario": sc,
                                "dist": dist,
                                "tau": tau,
                                "rep": rep + 1,
                                "method": "csqrnn",
                                "pilot_ratio": "",
                                "c_h": c_h,
                                "sigma_hat": sigma_hat,
                                "h": h,
                                "J": J,
                                "lambda": lam,
                                **cs,
                            }
                            rows.append(row)
                            cell_rows.append(row)
                        if "dcsqrnn" in methods:
                            dcs = run_dcsqrnn(
                                X_tr, y_tr, X_te, Q_te, tau, J, lam, h,
                                args.K, args.pilot_ratio, args.T, seed,
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
                            row = {
                                "scenario": sc,
                                "dist": dist,
                                "tau": tau,
                                "rep": rep + 1,
                                "method": "dcsqrnn",
                                "pilot_ratio": args.pilot_ratio,
                                "c_h": c_h,
                                "sigma_hat": sigma_hat,
                                "h": h,
                                "J": J,
                                "lambda": lam,
                                **dcs,
                            }
                            rows.append(row)
                            cell_rows.append(row)
                _print_setting_summary(sc, dist, tau, cell_rows)

    summary = summarise(
        rows,
        ["scenario", "dist", "tau", "method", "pilot_ratio", "c_h"],
        ["h", "sigma_hat", "mae", "rmse", "time"],
    )
    save_csv(f"{args.out}/bandwidth_sensitivity_raw.csv", rows)
    save_csv(f"{args.out}/bandwidth_sensitivity_summary.csv", summary)
    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
