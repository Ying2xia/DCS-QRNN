#!/usr/bin/env python3
"""Chapter 4: refit only CS-QRNN and overwrite its rows in 4-2 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import ROOT, add_sim_filter_args
from common.config import DEFAULT_H, DEFAULT_J, DEFAULT_LAMBDA, DEFAULT_SIM_N, DEFAULT_SIM_N_TEST, SIM_DISTS, SIM_SCENARIOS, TAUS
from common.data import GEN
from common.experiments import run_csqrnn
from common.metrics import summarise
from common.storage import load_csv, save_csv, save_json
from common.training import load_hyperparameter_map, resolve_hyperparams


def _as_float(value):
    return float(value)


def _as_int(value):
    return int(float(value))


def _print_setting_summary(sc: str, dist: str, tau: float, rows: list[dict]):
    summary = summarise(rows, ["method"], ["mae", "rmse", "time"])
    print(f"\nRefit CS-QRNN setting: {sc}, {dist}, tau={tau}")
    print(f"{'method':<10} {'n':>3} {'MAE mean':>12} {'MAE sd':>12} {'RMSE mean':>12} {'RMSE sd':>12} {'time mean':>12}")
    for row in summary:
        print(
            f"{row['method']:<10} {int(row['n']):>3d} "
            f"{row['mae_mean']:>12.6f} {row['mae_std']:>12.6f} "
            f"{row['rmse_mean']:>12.6f} {row['rmse_std']:>12.6f} "
            f"{row['time_mean']:>12.3f}"
        )
    print("", flush=True)


def _row_matches(row: dict, targets: set[tuple]) -> bool:
    if row.get("method") != "csqrnn":
        return False
    key = (row.get("scenario"), row.get("dist"), float(row.get("tau")), int(float(row.get("rep"))))
    return key in targets


def _normalise_row(row: dict) -> dict:
    out = dict(row)
    if "tau" in out and out["tau"] != "":
        out["tau"] = float(out["tau"])
    if "rep" in out and out["rep"] != "":
        out["rep"] = int(float(out["rep"]))
    for key in ("J", "nit"):
        if key in out and out[key] not in ("", None):
            out[key] = int(float(out[key]))
    for key in ("lambda", "mae", "rmse", "time"):
        if key in out and out[key] not in ("", None):
            out[key] = float(out[key])
    return out


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
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="torch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--torch-lr", type=float, default=0.01)
    parser.add_argument("--torch-check-every", type=int, default=25)
    parser.add_argument("--torch-maxiter", type=int, default=800)
    parser.add_argument("--torch-lbfgs-steps", type=int, default=1000)
    parser.add_argument("--torch-lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--torch-lbfgs-history-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter4_centralized"),
                        help="Existing 4-2 output directory to patch in place.")
    parser.add_argument("--backup", action="store_true",
                        help="Write .bak copies of existing raw/summary CSVs before overwriting.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    raw_path = out_dir / "centralized_raw.csv"
    summary_path = out_dir / "centralized_summary.csv"

    scenarios = list(SIM_SCENARIOS) if args.scenario == "all" else [args.scenario]
    dists = list(SIM_DISTS) if args.dist == "all" else [args.dist]
    taus = list(TAUS) if args.all_taus else [args.tau]
    J_map, lam_map, fallback_J, fallback_lam = load_hyperparameter_map(args.hyperparams, args.J, args.lam)

    new_rows = []
    target_keys = set()
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
                    print(f"[{done}/{total}] refit CS-QRNN: {sc}, {dist}, tau={tau}, rep={rep + 1}")
                    X_tr, y_tr, _ = gen_fn(args.N, tau, dist, np.random.default_rng(seed))
                    X_te, _, Q_te = gen_fn(args.N_test, tau, dist, np.random.default_rng(seed + 1_000_000))
                    J, lam = resolve_hyperparams(J_map, lam_map, fallback_J, fallback_lam, tau, sc, dist)
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
                    row = {"scenario": sc, "dist": dist, "tau": tau, "rep": rep + 1, "method": "csqrnn", "J": J, "lambda": lam, **res}
                    new_rows.append(row)
                    cell_rows.append(row)
                    target_keys.add((sc, dist, float(tau), rep + 1))
                _print_setting_summary(sc, dist, tau, cell_rows)

    if raw_path.exists():
        old_rows = load_csv(raw_path)
        kept_rows = [row for row in old_rows if not _row_matches(row, target_keys)]
        if args.backup:
            save_csv(str(raw_path) + ".bak", old_rows)
            if summary_path.exists():
                save_csv(str(summary_path) + ".bak", load_csv(summary_path))
    else:
        kept_rows = []

    merged_rows = [_normalise_row(row) for row in kept_rows + new_rows]
    summary = summarise(merged_rows, ["scenario", "dist", "tau", "method"], ["mae", "rmse", "time"])

    save_csv(raw_path, merged_rows)
    save_csv(summary_path, summary)
    save_json(out_dir / "run_4_2_refit_csqrnn_config.json", vars(args))

    print(f"Patched {len(new_rows)} CS-QRNN rows in {raw_path}")
    print(f"Recomputed summary: {summary_path}")


if __name__ == "__main__":
    main()
