#!/usr/bin/env python3
"""Chapter 4.7: ablation analysis of CS-QRNN/DCS-QRNN components."""

from __future__ import annotations

import argparse
import time

import numpy as np

from _common import ROOT
from common.config import DEFAULT_H, DEFAULT_J, DEFAULT_K, DEFAULT_LAMBDA, DEFAULT_SIM_N, DEFAULT_SIM_N_TEST, DEFAULT_T
from common.data import GEN
from common.experiments import evaluate_prediction, run_csqrnn, run_dcsqrnn, run_qrnn
from common.metrics import summarise
from common.model import CSQRNN
from common.storage import save_csv, save_json
from common.training import fit_csqrnn, load_hyperparameter_map, resolve_hyperparams


def _sample_pilot(X, y, pilot_ratio: float, seed: int):
    rng = np.random.default_rng(seed + 100_003)
    mask = rng.random(len(y)) < float(pilot_ratio)
    min_pilot = min(len(y), max(10, int(round(float(pilot_ratio) * len(y)))))
    if int(mask.sum()) < min_pilot:
        idx_fill = rng.choice(len(y), size=min_pilot, replace=False)
        mask = np.zeros(len(y), dtype=bool)
        mask[idx_fill] = True
    idx = np.flatnonzero(mask)
    return X[idx], y[idx]


def run_pilot_only_csqrnn(X_tr, y_tr, X_te, y_eval, tau: float, J: int,
                          lam: float, h: float, pilot_ratio: float,
                          seed: int, args):
    Xp, yp = _sample_pilot(X_tr, y_tr, pilot_ratio, seed)
    model = CSQRNN(X_tr.shape[1], J, tau, h, lam)
    theta0 = model.net.init(np.random.default_rng(seed))
    t0 = time.perf_counter()
    if args.backend == "torch":
        from common.torch_backend import resolve_backend, fit_csqrnn_torch
        _, actual_device = resolve_backend("torch", args.device)
        theta, res = fit_csqrnn_torch(
            model,
            Xp,
            yp,
            theta0,
            maxiter=args.torch_maxiter,
            device=actual_device,
            dtype_name=args.torch_dtype,
            lr=args.torch_lr,
            check_every=args.torch_check_every,
            lbfgs_steps=args.torch_lbfgs_steps,
            lbfgs_lr=args.torch_lbfgs_lr,
            lbfgs_history_size=args.torch_lbfgs_history_size,
        )
        backend, device = "torch", actual_device
    else:
        theta, res = fit_csqrnn(model, Xp, yp, theta0, maxiter=args.maxiter)
        backend, device = "numpy", "cpu"
    elapsed = time.perf_counter() - t0
    pred = model.net.forward(X_te, theta)
    out = evaluate_prediction(y_eval, pred)
    out.update({
        "time": elapsed,
        "pilot_size": int(len(yp)),
        "converged": bool(res.success),
        "nit": int(getattr(res, "nit", -1)),
        "backend": backend,
        "device": device,
    })
    return out


def _print_setting(rows):
    if not rows:
        return
    first = rows[0]
    print(f"\nFinished ablation setting: {first['scenario']}, {first['dist']}, tau={first['tau']}, rep={first['rep']}")
    print(f"{'method':<18} {'MAE':>12} {'RMSE':>12} {'time':>12} {'pilot':>8}")
    for row in rows:
        pilot = row.get("pilot_size", "")
        print(
            f"{row['method']:<18} {row['mae']:>12.6f} "
            f"{row['rmse']:>12.6f} {row['time']:>12.3f} {str(pilot):>8}"
        )
    print("", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["ex21", "ex22", "ex23"], default="ex23")
    parser.add_argument("--dist", choices=["N01", "t3", "chi2_2"], default="t3")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--N", type=int, default=DEFAULT_SIM_N)
    parser.add_argument("--N-test", type=int, default=DEFAULT_SIM_N_TEST)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--J", type=int, default=DEFAULT_J)
    parser.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--pilot-ratio", type=float, default=0.05)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--hyperparams", default=None)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--backend", choices=["numpy", "torch"], default="torch")
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
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter4_ablation"))
    args = parser.parse_args()

    J_map, lam_map, fallback_J, fallback_lam = load_hyperparameter_map(args.hyperparams, args.J, args.lam)
    gen_fn, _ = GEN[args.scenario]
    rows = []
    for rep in range(args.reps):
        seed = args.seed + 100000 * ["ex21", "ex22", "ex23"].index(args.scenario) + 10000 * ["N01", "t3", "chi2_2"].index(args.dist) + 100 * int(100 * args.tau) + rep
        X_tr, y_tr, _ = gen_fn(args.N, args.tau, args.dist, np.random.default_rng(seed))
        X_te, _, Q_te = gen_fn(args.N_test, args.tau, args.dist, np.random.default_rng(seed + 1_000_000))
        J, lam = resolve_hyperparams(J_map, lam_map, fallback_J, fallback_lam, args.tau, args.scenario, args.dist)
        cell_rows = []

        qrnn = run_qrnn(
            X_tr, y_tr, X_te, Q_te, args.tau, J, lam, seed,
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
        cs, _ = run_csqrnn(
            X_tr, y_tr, X_te, Q_te, args.tau, J, lam, args.h, seed,
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
        pilot_only = run_pilot_only_csqrnn(
            X_tr, y_tr, X_te, Q_te, args.tau, J, lam, args.h,
            args.pilot_ratio, seed, args,
        )
        dcs = run_dcsqrnn(
            X_tr, y_tr, X_te, Q_te, args.tau, J, lam, args.h,
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

        for method, component, res in (
            ("qrnn", "Huber-smoothed QRNN baseline", qrnn),
            ("csqrnn", "Convolution-smoothed full-data QRNN", cs),
            ("pilot_only_csqrnn", "Pilot-only CS-QRNN without global-gradient correction", pilot_only),
            ("dcsqrnn", "Full DCS-QRNN with pilot sampling and global-gradient correction", dcs),
        ):
            row = {
                "scenario": args.scenario,
                "dist": args.dist,
                "tau": args.tau,
                "rep": rep + 1,
                "method": method,
                "component": component,
                "pilot_ratio": args.pilot_ratio if method in {"pilot_only_csqrnn", "dcsqrnn"} else "",
                "J": J,
                "lambda": lam,
                "h": args.h,
                **res,
            }
            rows.append(row)
            cell_rows.append(row)
        _print_setting(cell_rows)

    summary = summarise(rows, ["scenario", "dist", "tau", "method", "component"], ["mae", "rmse", "time"])
    save_csv(f"{args.out}/ablation_raw.csv", rows)
    save_csv(f"{args.out}/ablation_summary.csv", summary)
    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
