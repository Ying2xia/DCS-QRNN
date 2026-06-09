#!/usr/bin/env python3
"""Chapter 5: select J and lambda by minimum BIC for real data."""

from __future__ import annotations

import argparse
import numpy as np

from _common import ROOT, add_real_data_args, add_real_filter_args, load_dataset, selected_datasets, selected_taus
from common.config import DEFAULT_H, DEFAULT_J, DEFAULT_LAMBDA, J_GRID, LAMBDA_GRID
from common.storage import save_csv, save_json
from common.training import estimate_residual_sigma, select_hyperparameters


def _grid_float(text, default):
    if text is None:
        return list(default)
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _grid_int(text, default):
    if text is None:
        return list(default)
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_real_filter_args(parser)
    add_real_data_args(parser)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--bic-N", type=int, default=5000)
    parser.add_argument("--c-h", type=float, default=0.10)
    parser.add_argument("--h0", type=float, default=DEFAULT_H)
    parser.add_argument("--J-grid", default=None)
    parser.add_argument("--lambda-grid", default=None)
    parser.add_argument("--sigma-J", type=int, default=DEFAULT_J)
    parser.add_argument("--sigma-lambda", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--sigma-sub", type=int, default=5000)
    parser.add_argument("--sigma-maxiter", type=int, default=300)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--df-mode", choices=["selected_variables", "active_hidden_nodes", "hidden_weights", "parameters"], default="selected_variables")
    parser.add_argument("--active-threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter5_hyperparameters"))
    args = parser.parse_args()

    datasets = selected_datasets(args.dataset)
    taus = selected_taus(args)
    J_grid = _grid_int(args.J_grid, J_GRID)
    lambda_grid = _grid_float(args.lambda_grid, LAMBDA_GRID)

    selected_rows, grid_rows = [], []
    total = len(datasets) * len(taus)
    done = 0
    for d_i, dataset in enumerate(datasets):
        X_tr, y_tr, _, _, _ = load_dataset(args, dataset, seed=args.seed + d_i)
        rng_data = np.random.default_rng(args.seed + 10_000 + d_i)
        n_bic = min(int(args.bic_N), len(y_tr))
        idx = rng_data.choice(len(y_tr), n_bic, replace=False)
        X_bic, y_bic = X_tr[idx], y_tr[idx]
        for tau in taus:
            done += 1
            cell_seed = args.seed + 1000 * d_i + int(100 * tau)
            print(f"[{done}/{total}] real-data BIC: {dataset}, tau={tau}")
            sigma_hat = estimate_residual_sigma(
                X_bic,
                y_bic,
                tau,
                args.sigma_J,
                args.sigma_lambda,
                np.random.default_rng(cell_seed + 99),
                h0=args.h0,
                maxiter=args.sigma_maxiter,
                n_sub=min(args.sigma_sub, len(y_bic)),
            )
            h = max(args.c_h * sigma_hat, 1e-4)
            best, grid = select_hyperparameters(
                X_bic,
                y_bic,
                tau,
                np.random.default_rng(cell_seed),
                J_grid=J_grid,
                lambda_grid=lambda_grid,
                h=h,
                maxiter=args.maxiter,
                n_restarts=args.n_restarts,
                df_mode=args.df_mode,
                active_threshold=args.active_threshold,
            )
            meta = {"dataset": dataset, "tau": tau, "sigma_hat": sigma_hat, "c_h": args.c_h, "h": h, "bic_N": n_bic}
            selected_rows.append({**meta, **best})
            grid_rows.extend({**meta, **row} for row in grid)
            print(f"  selected J={best['J']}, lambda={best['lambda']}, h={h:.6g}, BIC={best['bic']:.6f}")

    save_csv(f"{args.out}/selected_hyperparameters.csv", selected_rows)
    save_csv(f"{args.out}/bic_grid_raw.csv", grid_rows)
    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
