#!/usr/bin/env python3
"""Chapter 5: distributed real-data comparison."""

from __future__ import annotations

import argparse
import numpy as np

from _common import ROOT, add_real_data_args, add_real_filter_args, load_dataset, load_h_map, parse_methods, resolve_h, selected_datasets, selected_taus
from common.config import DEFAULT_H, DEFAULT_J, DEFAULT_LAMBDA, DEFAULT_T, PILOT_RATIOS
from common.experiments import run_csqrnn, run_dcsqrnn_workers, run_fedavg_workers
from common.metrics import summarise
from common.storage import load_csv, save_csv, save_json
from common.training import estimate_residual_sigma, load_hyperparameter_map, resolve_hyperparams


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
        key = (row["dataset"], float(row["tau"]), int(float(row["rep"])))
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


def _print_setting_summary(dataset: str, tau: float, rows: list[dict]):
    if not rows:
        return
    summary = summarise(rows, ["method", "pilot_ratio"], ["mae", "rmse", "time", "ract"])
    print(f"\nFinished real distributed setting: {dataset}, tau={tau}")
    print(f"{'method':<14} {'pilot':>7} {'n':>3} {'MAE mean':>12} {'MAE sd':>12} {'RMSE mean':>12} {'RMSE sd':>12} {'RACT mean':>12}")
    for row in summary:
        pilot = row["pilot_ratio"] if row["pilot_ratio"] not in ("", None) else "-"
        ract = row["ract_mean"]
        ract_text = f"{ract:.3f}" if np.isfinite(ract) else "-"
        print(
            f"{row['method']:<14} {str(pilot):>7} {int(row['n']):>3d} "
            f"{row['mae_mean']:>12.6f} {row['mae_std']:>12.6f} "
            f"{row['rmse_mean']:>12.6f} {row['rmse_std']:>12.6f} "
            f"{ract_text:>12}"
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
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--pilot-ratios", default=",".join(str(v) for v in PILOT_RATIOS))
    parser.add_argument("--methods", default="csqrnn,dcsqrnn,fedavg")
    parser.add_argument("--csqrnn-raw", default=None,
                        help="Optional real_centralized_raw.csv. When supplied, CS-QRNN rows are reused instead of refit.")
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
    parser.add_argument("--torch-fedavg-lbfgs-steps", type=int, default=0)
    parser.add_argument("--fedavg-rounds", type=int, default=10)
    parser.add_argument("--fedavg-local-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter5_distributed"))
    args = parser.parse_args()

    datasets = selected_datasets(args.dataset)
    taus = selected_taus(args)
    methods = parse_methods(args.methods)
    pilot_ratios = [float(v.strip()) for v in args.pilot_ratios.split(",") if v.strip()]
    J_map, lam_map, fallback_J, fallback_lam = load_hyperparameter_map(args.hyperparams, args.J, args.lam)
    h_map = load_h_map(args.hyperparams)
    csqrnn_map = _load_csqrnn_rows(args.csqrnn_raw)
    cfg = vars(args)

    rows = []
    total = len(datasets) * len(taus) * args.reps
    done = 0
    for d_i, dataset in enumerate(datasets):
        for rep in range(args.reps):
            split_seed = args.seed + 1000 * d_i + rep
            X_tr, y_tr, X_te, y_te, workers = load_dataset(args, dataset, seed=split_seed)
            for tau in taus:
                done += 1
                seed = args.seed + 10000 * d_i + 100 * int(100 * tau) + rep
                print(f"[{done}/{total}] real distributed: {dataset}, tau={tau}, rep={rep + 1}")
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

                cs_res = None
                if "csqrnn" in methods or "dcsqrnn" in methods or "fedavg" in methods:
                    if csqrnn_map:
                        key = (dataset, float(tau), rep + 1)
                        if key not in csqrnn_map:
                            raise KeyError(f"Missing CS-QRNN row in {args.csqrnn_raw}: {key}")
                        cs_res = dict(csqrnn_map[key])
                    else:
                        cs_res, _ = run_csqrnn(
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
                cs_time = cs_res["time"] if cs_res else 1.0
                if "csqrnn" in methods:
                    row = {
                        "dataset": dataset,
                        "tau": tau,
                        "rep": rep + 1,
                        "method": "csqrnn",
                        "pilot_ratio": "",
                        "J": J,
                        "lambda": lam,
                        "h": h,
                        "ract": 1.0,
                        **cs_res,
                    }
                    rows.append(row)
                    cell_rows.append(row)
                if "dcsqrnn" in methods:
                    for pr in pilot_ratios:
                        res = run_dcsqrnn_workers(
                            workers, X_te, y_te, tau, J, lam, h, pr, args.T, seed,
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
                            "dataset": dataset,
                            "tau": tau,
                            "rep": rep + 1,
                            "method": f"dcsqrnn_{int(round(100 * pr))}pct",
                            "pilot_ratio": pr,
                            "J": J,
                            "lambda": lam,
                            "h": h,
                            "ract": cs_time / max(res["time"], 1e-9),
                            **res,
                        }
                        rows.append(row)
                        cell_rows.append(row)
                if "fedavg" in methods:
                    res = run_fedavg_workers(workers, X_te, y_te, tau, J, lam, seed, cfg)
                    row = {
                        "dataset": dataset,
                        "tau": tau,
                        "rep": rep + 1,
                        "method": "fedavg",
                        "pilot_ratio": "",
                        "J": J,
                        "lambda": lam,
                        "h": h,
                        "ract": cs_time / max(res["time"], 1e-9),
                        **res,
                    }
                    rows.append(row)
                    cell_rows.append(row)
                _print_setting_summary(dataset, tau, cell_rows)

    summary = summarise(rows, ["dataset", "tau", "method"], ["mae", "rmse", "time", "ract"])
    save_csv(f"{args.out}/real_distributed_raw.csv", rows)
    save_csv(f"{args.out}/real_distributed_summary.csv", summary)
    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
