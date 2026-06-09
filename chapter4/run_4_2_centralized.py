#!/usr/bin/env python3
"""Chapter 4: centralized simulation comparison.

Methods: CS-QRNN, QRNN, CatBoost-Quantile, ELM-QR, RVFL-QR, and linear QR.
"""

from __future__ import annotations

import argparse
import numpy as np

from _common import ROOT, add_sim_filter_args, parse_methods
from common.config import DEFAULT_H, DEFAULT_J, DEFAULT_LAMBDA, DEFAULT_SIM_N, DEFAULT_SIM_N_TEST, SIM_DISTS, SIM_SCENARIOS, TAUS
from common.data import GEN
from common.experiments import run_catboost, run_csqrnn, run_elm, run_linear_qr, run_qrnn, run_rvfl
from common.metrics import summarise
from common.storage import save_csv, save_json
from common.training import load_hyperparameter_map, resolve_hyperparams


def _print_setting_summary(sc: str, dist: str, tau: float, rows: list[dict]):
    if not rows:
        return
    summary = summarise(rows, ["method"], ["mae", "rmse", "time"])
    print(f"\nFinished setting: {sc}, {dist}, tau={tau}")
    print(f"{'method':<10} {'n':>3} {'MAE mean':>12} {'MAE sd':>12} {'RMSE mean':>12} {'RMSE sd':>12} {'time mean':>12}")
    for row in summary:
        print(
            f"{row['method']:<10} {int(row['n']):>3d} "
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
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--J", type=int, default=DEFAULT_J)
    parser.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--hyperparams", default=None)
    parser.add_argument("--methods", default="csqrnn,qrnn,catboost,elm,rvfl,qr")
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--qrnn-maxiter", type=int, default=500)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--device", default="auto", help="Use cuda on a GPU server, or auto.")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--torch-lr", type=float, default=0.01)
    parser.add_argument("--torch-check-every", type=int, default=25)
    parser.add_argument("--torch-maxiter", type=int, default=800)
    parser.add_argument("--torch-qrnn-maxiter", type=int, default=200)
    parser.add_argument("--torch-lbfgs-steps", type=int, default=400)
    parser.add_argument("--torch-lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--torch-lbfgs-history-size", type=int, default=20)
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
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter4_centralized"))
    args = parser.parse_args()

    scenarios = list(SIM_SCENARIOS) if args.scenario == "all" else [args.scenario]
    dists = list(SIM_DISTS) if args.dist == "all" else [args.dist]
    taus = list(TAUS) if args.all_taus else [args.tau]
    methods = parse_methods(args.methods)
    J_map, lam_map, fallback_J, fallback_lam = load_hyperparameter_map(args.hyperparams, args.J, args.lam)
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
                    print(f"[{done}/{total}] centralized: {sc}, {dist}, tau={tau}, rep={rep + 1}")
                    X_tr, y_tr, _ = gen_fn(args.N, tau, dist, np.random.default_rng(seed))
                    X_te, _, Q_te = gen_fn(args.N_test, tau, dist, np.random.default_rng(seed + 1_000_000))
                    J, lam = resolve_hyperparams(J_map, lam_map, fallback_J, fallback_lam, tau, sc, dist)
                    for method in methods:
                        try:
                            if method == "csqrnn":
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
                            elif method == "qrnn":
                                res = run_qrnn(
                                    X_tr, y_tr, X_te, Q_te, tau, J, lam, seed,
                                    maxiter=args.qrnn_maxiter,
                                    backend=args.backend,
                                    device=args.device,
                                    torch_dtype=args.torch_dtype,
                                    torch_lr=args.torch_lr,
                                    torch_check_every=args.torch_check_every,
                                    torch_maxiter=args.torch_qrnn_maxiter,
                                    torch_lbfgs_steps=args.torch_lbfgs_steps,
                                    torch_lbfgs_lr=args.torch_lbfgs_lr,
                                    torch_lbfgs_history_size=args.torch_lbfgs_history_size,
                                )
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
                            row = {"scenario": sc, "dist": dist, "tau": tau, "rep": rep + 1, "method": method, "J": J, "lambda": lam, **res}
                            raw_rows.append(row)
                            cell_rows.append(row)
                        except ImportError as exc:
                            print(f"  skipped {method}: {exc}")
                _print_setting_summary(sc, dist, tau, cell_rows)

    summary = summarise(raw_rows, ["scenario", "dist", "tau", "method"], ["mae", "rmse", "time"])
    save_csv(f"{args.out}/centralized_raw.csv", raw_rows)
    save_csv(f"{args.out}/centralized_summary.csv", summary)
    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
