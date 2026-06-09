#!/usr/bin/env python3
"""Chapter 4: select J and lambda by minimum BIC for simulation settings."""

from __future__ import annotations

import argparse
import numpy as np

from _common import ROOT, add_sim_filter_args
from common.config import DEFAULT_H, J_GRID, LAMBDA_GRID, SIM_DISTS, SIM_SCENARIOS, TAUS
from common.data import GEN
from common.storage import save_csv, save_json
from common.training import select_hyperparameters


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
    add_sim_filter_args(parser)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--J-grid", default=None)
    parser.add_argument("--lambda-grid", default=None)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--df-mode", choices=["selected_variables", "active_hidden_nodes", "hidden_weights", "parameters"], default="selected_variables")
    parser.add_argument("--active-threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter4_hyperparameters"))
    args = parser.parse_args()

    scenarios = list(SIM_SCENARIOS) if args.scenario == "all" else [args.scenario]
    dists = list(SIM_DISTS) if args.dist == "all" else [args.dist]
    taus = list(TAUS) if args.all_taus else [args.tau]
    J_grid = _grid_int(args.J_grid, J_GRID)
    lambda_grid = _grid_float(args.lambda_grid, LAMBDA_GRID)

    selected_rows, grid_rows = [], []
    total = len(scenarios) * len(dists) * len(taus)
    done = 0
    for sc in scenarios:
        gen_fn, _ = GEN[sc]
        for dist in dists:
            for tau in taus:
                done += 1
                print(f"[{done}/{total}] BIC: {sc}, {dist}, tau={tau}")
                rng = np.random.default_rng(args.seed + 10000 * SIM_SCENARIOS.index(sc) + 100 * SIM_DISTS.index(dist) + int(100 * tau))
                X, y, _ = gen_fn(args.N, tau, dist, rng)
                best, grid = select_hyperparameters(
                    X, y, tau, rng,
                    J_grid=J_grid,
                    lambda_grid=lambda_grid,
                    h=args.h,
                    maxiter=args.maxiter,
                    n_restarts=args.n_restarts,
                    df_mode=args.df_mode,
                    active_threshold=args.active_threshold,
                )
                meta = {"scenario": sc, "dist": dist, "tau": tau}
                selected_rows.append({**meta, **best})
                grid_rows.extend({**meta, **row} for row in grid)
                print(f"  selected J={best['J']}, lambda={best['lambda']}, BIC={best['bic']:.6f}")

    save_csv(f"{args.out}/selected_hyperparameters.csv", selected_rows)
    save_csv(f"{args.out}/bic_grid_raw.csv", grid_rows)
    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()

