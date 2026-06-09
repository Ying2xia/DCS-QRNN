#!/usr/bin/env python3
"""Chapter 5: centralized real-data comparison."""

from __future__ import annotations

import argparse
import numpy as np

from _common import ROOT, add_real_data_args, add_real_filter_args, load_dataset, load_h_map, parse_methods, resolve_h, selected_datasets, selected_taus
from common.config import DEFAULT_H, DEFAULT_J, DEFAULT_LAMBDA
from common.experiments import run_catboost, run_csqrnn, run_elm, run_linear_qr, run_qrnn, run_rvfl
from common.metrics import summarise
from common.storage import save_csv, save_json
from common.training import estimate_residual_sigma, load_hyperparameter_map, resolve_hyperparams


def _print_setting_summary(dataset: str, tau: float, rows: list[dict]):
    if not rows:
        return
    summary = summarise(rows, ["method"], ["mae", "rmse", "time"])
    print(f"\nFinished real centralized setting: {dataset}, tau={tau}")
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
    add_real_filter_args(parser)
    add_real_data_args(parser)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--J", type=int, default=DEFAULT_J)
    parser.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--hyperparams", default=None)
    parser.add_argument("--c-h", type=float, default=0.10)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--sigma-sub", type=int, default=5000)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--qrnn-maxiter", type=int, default=500)
    parser.add_argument("--methods", default="csqrnn,qrnn,catboost,elm,rvfl,qr")
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
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--torch-lr", type=float, default=0.01)
    parser.add_argument("--torch-check-every", type=int, default=25)
    parser.add_argument("--torch-maxiter", type=int, default=800)
    parser.add_argument("--torch-lbfgs-steps", type=int, default=400)
    parser.add_argument("--torch-lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--torch-lbfgs-history-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter5_centralized"))
    args = parser.parse_args()

    datasets = selected_datasets(args.dataset)
    taus = selected_taus(args)
    methods = parse_methods(args.methods)
    J_map, lam_map, fallback_J, fallback_lam = load_hyperparameter_map(args.hyperparams, args.J, args.lam)
    h_map = load_h_map(args.hyperparams)
    cfg = vars(args)

    rows = []
    total = len(datasets) * len(taus) * args.reps
    done = 0
    for d_i, dataset in enumerate(datasets):
        for rep in range(args.reps):
            split_seed = args.seed + 1000 * d_i + rep
            X_tr, y_tr, X_te, y_te, _ = load_dataset(args, dataset, seed=split_seed)
            for tau in taus:
                done += 1
                seed = args.seed + 10000 * d_i + 100 * int(100 * tau) + rep
                print(f"[{done}/{total}] real centralized: {dataset}, tau={tau}, rep={rep + 1}")
                cell_rows = []
                J, lam = resolve_hyperparams(J_map, lam_map, fallback_J, fallback_lam, tau, dataset)
                if args.h is None:
                    h = resolve_h(h_map, DEFAULT_H, tau, dataset)
                    if not h_map:
                        sigma_hat = estimate_residual_sigma(
                            X_tr,
                            y_tr,
                            tau,
                            J,
                            lam,
                            np.random.default_rng(seed + 99),
                            h0=DEFAULT_H,
                            n_sub=min(args.sigma_sub, len(y_tr)),
                        )
                        h = max(args.c_h * sigma_hat, 1e-4)
                else:
                    h = args.h
                for method in methods:
                    try:
                        if method == "csqrnn":
                            res, _ = run_csqrnn(
                                X_tr, y_tr, X_te, y_te, tau, J, lam, h, seed,
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
                                X_tr, y_tr, X_te, y_te, tau, J, lam, seed,
                                maxiter=args.qrnn_maxiter,
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
                        elif method == "catboost":
                            res = run_catboost(X_tr, y_tr, X_te, y_te, tau, seed, cfg)
                        elif method == "elm":
                            res = run_elm(X_tr, y_tr, X_te, y_te, tau, seed, cfg)
                        elif method == "rvfl":
                            res = run_rvfl(X_tr, y_tr, X_te, y_te, tau, seed, cfg)
                        elif method == "qr":
                            res = run_linear_qr(X_tr, y_tr, X_te, y_te, tau)
                        else:
                            raise ValueError(method)
                        row = {
                            "dataset": dataset,
                            "tau": tau,
                            "rep": rep + 1,
                            "method": method,
                            "J": J,
                            "lambda": lam,
                            "h": h,
                            **res,
                        }
                        rows.append(row)
                        cell_rows.append(row)
                    except ImportError as exc:
                        print(f"  skipped {method}: {exc}")
                _print_setting_summary(dataset, tau, cell_rows)

    summary = summarise(rows, ["dataset", "tau", "method"], ["mae", "rmse", "time"])
    save_csv(f"{args.out}/real_centralized_raw.csv", rows)
    save_csv(f"{args.out}/real_centralized_summary.csv", summary)
    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
