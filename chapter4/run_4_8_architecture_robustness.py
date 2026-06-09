#!/usr/bin/env python3
"""Chapter 4.8: architecture robustness for deeper MLPs.

This script runs the table labelled tab:architecture_robustness in the
revised manuscript. It compares CS-QRNN and DCS-QRNN using one-, two-, and
three-hidden-layer feedforward MLPs under a representative simulation setting.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np

from _common import ROOT, parse_methods
from common.config import (
    DEFAULT_H,
    DEFAULT_J,
    DEFAULT_K,
    DEFAULT_LAMBDA,
    DEFAULT_REPS,
    DEFAULT_SIM_N,
    DEFAULT_SIM_N_TEST,
    DEFAULT_T,
)
from common.data import GEN
from common.metrics import mae, rmse, summarise
from common.storage import save_csv, save_json
from common.training import load_hyperparameter_map, resolve_hyperparams


_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the deeper-architecture experiment.") from exc
    return torch


def _resolve_device(device: str):
    torch = _import_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def _dtype(name: str):
    torch = _import_torch()
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def _tensor(x, device: str, dtype_name: str):
    torch = _import_torch()
    return torch.as_tensor(np.asarray(x), dtype=_dtype(dtype_name), device=device)


@dataclass(frozen=True)
class DeepMLPSpec:
    p: int
    width: int
    depth: int

    @property
    def layer_sizes(self):
        return [self.p] + [self.width] * self.depth + [1]

    @property
    def n_params(self):
        total = 0
        sizes = self.layer_sizes
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            total += in_dim * out_dim + out_dim
        return int(total)

    @property
    def hidden_weight_count(self):
        sizes = self.layer_sizes
        total = 0
        for layer_id, (in_dim, out_dim) in enumerate(zip(sizes[:-1], sizes[1:])):
            if layer_id < self.depth:
                total += in_dim * out_dim
        return int(max(total, 1))

    @property
    def architecture_label(self):
        if self.depth == 1:
            return "One-hidden-layer MLP"
        if self.depth == 2:
            return "Two-hidden-layer MLP"
        if self.depth == 3:
            return "Three-hidden-layer MLP"
        return f"{self.depth}-hidden-layer MLP"


def init_theta(spec: DeepMLPSpec, rng: np.random.Generator):
    theta = np.zeros(spec.n_params, dtype=float)
    offset = 0
    sizes = spec.layer_sizes
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        n_w = in_dim * out_dim
        theta[offset:offset + n_w] = rng.normal(0.0, 1.0 / math.sqrt(max(in_dim, 1)), n_w)
        offset += n_w
        offset += out_dim
    return theta


def _unpack(theta, spec: DeepMLPSpec):
    params = []
    offset = 0
    sizes = spec.layer_sizes
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        n_w = in_dim * out_dim
        W = theta[offset:offset + n_w].reshape(out_dim, in_dim)
        offset += n_w
        b = theta[offset:offset + out_dim]
        offset += out_dim
        params.append((W, b))
    return params


def _forward(X, theta, spec: DeepMLPSpec):
    torch = _import_torch()
    H = X
    params = _unpack(theta, spec)
    for W, b in params[:-1]:
        H = torch.tanh(H @ W.T + b)
    W_out, b_out = params[-1]
    return (H @ W_out.T + b_out).reshape(-1)


def _normal_pdf(x):
    torch = _import_torch()
    return torch.exp(-0.5 * x * x) / _SQRT2PI


def _normal_cdf(x):
    torch = _import_torch()
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _cs_loss(e, tau: float, h: float):
    h = max(float(h), 1e-8)
    a = e / h
    return (float(tau) - _normal_cdf(-a)) * e + h * _normal_pdf(a)


def _objective(theta, X, y, spec: DeepMLPSpec, tau: float, h: float, lam: float):
    pred = _forward(X, theta, spec)
    e = y - pred
    loss = _cs_loss(e, tau, h).mean()
    if lam > 0:
        hidden_pen = 0.0
        for W, _ in _unpack(theta, spec)[:-1]:
            hidden_pen = hidden_pen + (W * W).sum()
        loss = loss + float(lam) / spec.hidden_weight_count * hidden_pen
    return loss


@dataclass
class FitResult:
    theta: np.ndarray
    fun: float
    nit: int
    success: bool


def fit_deep_csqrnn(X, y, theta0, spec: DeepMLPSpec, tau: float, h: float,
                    lam: float, device: str, dtype_name: str,
                    maxiter: int, lr: float, check_every: int,
                    lbfgs_steps: int, lbfgs_lr: float,
                    lbfgs_history_size: int) -> FitResult:
    torch = _import_torch()
    X_t = _tensor(X, device, dtype_name)
    y_t = _tensor(y, device, dtype_name)
    theta = torch.nn.Parameter(_tensor(theta0, device, dtype_name).clone())
    opt = torch.optim.Adam([theta], lr=float(lr))

    best_theta = theta.detach().clone()
    best_obj = float("inf")
    last_obj = None
    nit = 0
    success = False

    for step in range(1, int(maxiter) + 1):
        nit = step
        opt.zero_grad(set_to_none=True)
        loss = _objective(theta, X_t, y_t, spec, tau, h, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=100.0)
        opt.step()

        if step % int(check_every) == 0 or step == int(maxiter):
            obj = float(loss.detach().cpu().item())
            if np.isfinite(obj) and obj < best_obj:
                best_obj = obj
                best_theta = theta.detach().clone()
            if last_obj is not None and abs(last_obj - obj) <= 1e-7 * (1.0 + abs(last_obj)):
                success = True
                break
            last_obj = obj

    if int(lbfgs_steps) > 0:
        with torch.no_grad():
            theta.copy_(best_theta)
        lbfgs = torch.optim.LBFGS(
            [theta],
            lr=float(lbfgs_lr),
            max_iter=int(lbfgs_steps),
            history_size=int(lbfgs_history_size),
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            obj = _objective(theta, X_t, y_t, spec, tau, h, lam)
            obj.backward()
            return obj

        try:
            lbfgs.step(closure)
            with torch.no_grad():
                obj = float(_objective(theta, X_t, y_t, spec, tau, h, lam).detach().cpu().item())
                if np.isfinite(obj) and obj <= best_obj:
                    best_obj = obj
                    best_theta = theta.detach().clone()
            nit += int(lbfgs_steps)
            success = True
        except (RuntimeError, IndexError):
            with torch.no_grad():
                theta.copy_(best_theta)

    out = best_theta.detach().cpu().numpy().astype(float)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return FitResult(out, float(best_obj), int(nit), bool(success))


def gradient_deep_csqrnn(X, y, theta_np, spec: DeepMLPSpec, tau: float,
                         h: float, lam: float, device: str, dtype_name: str):
    torch = _import_torch()
    X_t = _tensor(X, device, dtype_name)
    y_t = _tensor(y, device, dtype_name)
    theta = _tensor(theta_np, device, dtype_name).clone().detach()
    theta.requires_grad_(True)
    loss = _objective(theta, X_t, y_t, spec, tau, h, lam)
    loss.backward()
    grad = theta.grad.detach().cpu().numpy().astype(float)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return grad


def predict_deep(X, theta_np, spec: DeepMLPSpec, device: str, dtype_name: str):
    torch = _import_torch()
    with torch.no_grad():
        X_t = _tensor(X, device, dtype_name)
        theta = _tensor(theta_np, device, dtype_name)
        pred = _forward(X_t, theta, spec).detach().cpu().numpy().astype(float)
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return pred


def fit_architecture_cs(X_tr, y_tr, X_te, Q_te, spec: DeepMLPSpec,
                        tau: float, h: float, lam: float, seed: int, args):
    theta0 = init_theta(spec, np.random.default_rng(seed))
    t0 = time.perf_counter()
    fit = fit_deep_csqrnn(
        X_tr, y_tr, theta0, spec, tau, h, lam, args.device,
        args.torch_dtype, args.torch_maxiter, args.torch_lr,
        args.torch_check_every, args.torch_lbfgs_steps,
        args.torch_lbfgs_lr, args.torch_lbfgs_history_size,
    )
    elapsed = time.perf_counter() - t0
    pred = predict_deep(X_te, fit.theta, spec, args.device, args.torch_dtype)
    return {
        "mae": mae(Q_te, pred),
        "rmse": rmse(Q_te, pred),
        "time": elapsed,
        "converged": fit.success,
        "nit": fit.nit,
    }


def fit_architecture_dcs(X_tr, y_tr, X_te, Q_te, spec: DeepMLPSpec,
                         tau: float, h: float, lam: float, seed: int, args):
    rng = np.random.default_rng(seed + 100_003)
    mask = rng.random(len(y_tr)) < float(args.pilot_ratio)
    min_pilot = min(len(y_tr), max(10, int(round(float(args.pilot_ratio) * len(y_tr)))))
    if int(mask.sum()) < min_pilot:
        idx_fill = rng.choice(len(y_tr), size=min_pilot, replace=False)
        mask = np.zeros(len(y_tr), dtype=bool)
        mask[idx_fill] = True
    idx = np.flatnonzero(mask)
    Xp, yp = X_tr[idx], y_tr[idx]

    theta0 = init_theta(spec, np.random.default_rng(seed))
    t0 = time.perf_counter()
    fit = fit_deep_csqrnn(
        Xp, yp, theta0, spec, tau, h, lam, args.device,
        args.torch_dtype, args.torch_maxiter, args.torch_lr,
        args.torch_check_every, args.torch_lbfgs_steps,
        args.torch_lbfgs_lr, args.torch_lbfgs_history_size,
    )
    theta = fit.theta
    for _ in range(int(args.T)):
        g_global = gradient_deep_csqrnn(
            X_tr, y_tr, theta, spec, tau, h, lam, args.device, args.torch_dtype,
        )
        g_norm = float(np.linalg.norm(g_global))
        theta_start = theta - float(args.step_scale) * g_global / g_norm if np.isfinite(g_norm) and g_norm >= 1e-12 else theta.copy()
        fit = fit_deep_csqrnn(
            Xp, yp, theta_start, spec, tau, h, lam, args.device,
            args.torch_dtype, args.torch_dcs_step_maxiter, args.torch_lr,
            args.torch_check_every, args.torch_lbfgs_steps,
            args.torch_lbfgs_lr, args.torch_lbfgs_history_size,
        )
        theta = fit.theta
    elapsed = time.perf_counter() - t0
    pred = predict_deep(X_te, theta, spec, args.device, args.torch_dtype)
    return {
        "mae": mae(Q_te, pred),
        "rmse": rmse(Q_te, pred),
        "time": elapsed,
        "converged": fit.success,
        "nit": fit.nit,
        "pilot_size": int(len(yp)),
    }


def _print_summary(rows):
    summary = summarise(rows, ["method", "architecture", "tau"], ["mae", "rmse", "time"])
    print("\nFinished architecture robustness")
    print(f"{'method':<10} {'architecture':<24} {'tau':>5} {'n':>3} {'MAE':>12} {'RMSE':>12} {'time':>12}")
    for row in summary:
        print(
            f"{row['method']:<10} {row['architecture']:<24} {float(row['tau']):>5.1f} "
            f"{int(row['n']):>3d} {row['mae_mean']:>12.6f} {row['rmse_mean']:>12.6f} "
            f"{row['time_mean']:>12.3f}"
        )
    print("", flush=True)


def _print_completed_setting(rows):
    if not rows:
        return
    first = rows[0]
    print(
        f"\nFinished setting: {first['scenario']}, {first['dist']}, "
        f"tau={first['tau']}, {first['architecture']}, rep={first['rep']}"
    )
    print(f"{'method':<10} {'MAE':>12} {'RMSE':>12} {'time':>12} {'nit':>8}")
    for row in rows:
        print(
            f"{row['method']:<10} {row['mae']:>12.6f} "
            f"{row['rmse']:>12.6f} {row['time']:>12.3f} "
            f"{int(row.get('nit', -1)):>8d}"
        )
    print("", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["ex21", "ex22", "ex23"], default="ex23")
    parser.add_argument("--dist", choices=["N01", "t3", "chi2_2"], default="t3")
    parser.add_argument("--taus", default="0.1,0.5,0.9")
    parser.add_argument("--N", type=int, default=DEFAULT_SIM_N)
    parser.add_argument("--N-test", type=int, default=DEFAULT_SIM_N_TEST)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--J", type=int, default=DEFAULT_J)
    parser.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--h", type=float, default=DEFAULT_H)
    parser.add_argument("--pilot-ratio", type=float, default=0.10)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--architectures", default="1,2,3",
                        help="Comma-separated hidden-layer counts.")
    parser.add_argument("--methods", default="csqrnn,dcsqrnn")
    parser.add_argument("--hyperparams", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--torch-lr", type=float, default=0.01)
    parser.add_argument("--torch-check-every", type=int, default=25)
    parser.add_argument("--torch-maxiter", type=int, default=800)
    parser.add_argument("--torch-dcs-step-maxiter", type=int, default=400)
    parser.add_argument("--torch-lbfgs-steps", type=int, default=400)
    parser.add_argument("--torch-lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--torch-lbfgs-history-size", type=int, default=20)
    parser.add_argument("--step-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", default=str(ROOT / "results" / "chapter4_architecture"))
    args = parser.parse_args()

    args.device = _resolve_device(args.device)
    taus = [float(v) for v in args.taus.split(",") if v.strip()]
    depths = [int(v) for v in args.architectures.split(",") if v.strip()]
    methods = parse_methods(args.methods)
    unknown = [m for m in methods if m not in {"csqrnn", "dcsqrnn"}]
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")

    J_map, lam_map, fallback_J, fallback_lam = load_hyperparameter_map(args.hyperparams, args.J, args.lam)
    gen_fn, p = GEN[args.scenario]

    rows = []
    total = len(taus) * len(depths) * args.reps
    done = 0
    for tau in taus:
        J, lam = resolve_hyperparams(J_map, lam_map, fallback_J, fallback_lam, tau, args.scenario, args.dist)
        for rep in range(args.reps):
            seed = args.seed + 100000 * ["ex21", "ex22", "ex23"].index(args.scenario) + 10000 * ["N01", "t3", "chi2_2"].index(args.dist) + 100 * int(100 * tau) + rep
            X_tr, y_tr, _ = gen_fn(args.N, tau, args.dist, np.random.default_rng(seed))
            X_te, _, Q_te = gen_fn(args.N_test, tau, args.dist, np.random.default_rng(seed + 1_000_000))
            for depth in depths:
                done += 1
                spec = DeepMLPSpec(p=p, width=J, depth=depth)
                setting_rows = []
                print(f"[{done}/{total}] architecture: {args.scenario}, {args.dist}, tau={tau}, depth={depth}, rep={rep + 1}")
                if "csqrnn" in methods:
                    res = fit_architecture_cs(X_tr, y_tr, X_te, Q_te, spec, tau, args.h, lam, seed, args)
                    row = {
                        "scenario": args.scenario,
                        "dist": args.dist,
                        "tau": tau,
                        "rep": rep + 1,
                        "method": "csqrnn",
                        "architecture": spec.architecture_label,
                        "depth": depth,
                        "width": J,
                        "lambda": lam,
                        "h": args.h,
                        "pilot_ratio": "",
                        "device": args.device,
                        **res,
                    }
                    rows.append(row)
                    setting_rows.append(row)
                if "dcsqrnn" in methods:
                    res = fit_architecture_dcs(X_tr, y_tr, X_te, Q_te, spec, tau, args.h, lam, seed, args)
                    row = {
                        "scenario": args.scenario,
                        "dist": args.dist,
                        "tau": tau,
                        "rep": rep + 1,
                        "method": "dcsqrnn",
                        "architecture": spec.architecture_label,
                        "depth": depth,
                        "width": J,
                        "lambda": lam,
                        "h": args.h,
                        "pilot_ratio": args.pilot_ratio,
                        "device": args.device,
                        **res,
                    }
                    rows.append(row)
                    setting_rows.append(row)
                _print_completed_setting(setting_rows)
    _print_summary(rows)
    summary = summarise(
        rows,
        ["scenario", "dist", "tau", "method", "architecture", "depth", "width"],
        ["mae", "rmse", "time"],
    )
    save_csv(f"{args.out}/architecture_robustness_raw.csv", rows)
    save_csv(f"{args.out}/architecture_robustness_summary.csv", summary)
    save_json(f"{args.out}/run_config.json", vars(args))
    print(f"Saved outputs to {args.out}")


if __name__ == "__main__":
    main()
