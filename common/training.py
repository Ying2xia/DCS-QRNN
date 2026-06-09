"""Model fitting, BIC hyperparameter selection, and utility loaders."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import minimize

from .config import DEFAULT_H, DEFAULT_J, DEFAULT_LAMBDA, J_GRID, LAMBDA_GRID
from .model import CSQRNN, QRNNLayer, cs_loss, huber_loss, huber_score


def fit_csqrnn(model: CSQRNN, X, y, theta0, maxiter: int = 2000):
    res = minimize(
        fun=lambda t: model.loss(t, X, y),
        jac=lambda t: model.gradient(t, X, y),
        x0=theta0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7},
    )
    return res.x, res


def fit_qrnn_huber(net: QRNNLayer, tau: float, lam: float,
                   X, y, theta0, eps_schedule=None, maxiter: int = 500):
    if eps_schedule is None:
        eps_schedule = [0.5 * (0.5 ** i) for i in range(12)]

    p, J = net.p, net.J
    dWh = J * p
    c_pen = lam / max(p * J, 1)

    def obj(th, eps):
        e = y - net.forward(X, th)
        return float(np.mean(huber_loss(e, tau, eps))) + c_pen * float(th[:dWh] @ th[:dWh])

    def grd(th, eps):
        n = len(y)
        e = y - net.forward(X, th)
        g = -(huber_score(e, tau, eps) @ net.jacobian(X, th)) / n
        gpen = np.zeros_like(th)
        gpen[:dWh] = 2.0 * c_pen * th[:dWh]
        return g + gpen

    th = theta0.copy()
    last_res = None
    for eps in eps_schedule:
        last_res = minimize(
            fun=lambda t, e=eps: obj(t, e),
            jac=lambda t, e=eps: grd(t, e),
            x0=th,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )
        th = last_res.x
    return th, last_res


def fit_linear_qr_huber(X, y, tau: float, eps: float = 0.001,
                        maxiter: int = 5000):
    n = len(y)
    Xa = np.hstack([np.ones((n, 1)), X])

    def obj(beta):
        e = y - Xa @ beta
        return float(np.mean(huber_loss(e, tau, eps)))

    def grd(beta):
        e = y - Xa @ beta
        return -(huber_score(e, tau, eps) @ Xa) / n

    res = minimize(
        obj,
        jac=grd,
        x0=np.zeros(Xa.shape[1]),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-4},
    )
    return res.x, res


def predict_linear_qr(X, beta):
    return np.hstack([np.ones((X.shape[0], 1)), X]) @ beta


def parameter_count(p: int, J: int) -> int:
    return int(J * p + J + J + 1)


def effective_df(model: CSQRNN, theta: np.ndarray,
                 mode: str = "selected_variables",
                 active_threshold: float = 1e-4) -> int:
    W_h, _, W_o, _ = model.net.unpack(theta)
    if mode == "selected_variables":
        active = np.linalg.norm(W_h, axis=0) > active_threshold
        return int(max(1, active.sum()))
    if mode == "active_hidden_nodes":
        active = (np.linalg.norm(W_h, axis=1) > active_threshold) & (np.abs(W_o) > active_threshold)
        return int(max(1, active.sum()))
    if mode == "hidden_weights":
        return int(max(1, np.sum(np.abs(W_h) > active_threshold)))
    if mode == "parameters":
        return parameter_count(model.p, model.J)
    raise ValueError(f"unknown df mode: {mode}")


def bic_score(mean_loss: float, n: int, df: int) -> float:
    return float(np.log(max(mean_loss, 1e-12)) + (np.log(n) / n) * df)


def select_hyperparameters(X, y, tau: float, rng: np.random.Generator,
                           J_grid: Iterable[int] = J_GRID,
                           lambda_grid: Iterable[float] = LAMBDA_GRID,
                           h: float = DEFAULT_H,
                           maxiter: int = 300,
                           n_restarts: int = 1,
                           df_mode: str = "selected_variables",
                           active_threshold: float = 1e-4):
    """Grid-search J and lambda by minimum BIC."""
    rows = []
    p = X.shape[1]
    for J in J_grid:
        for lam in lambda_grid:
            candidates = []
            for restart in range(max(1, int(n_restarts))):
                model = CSQRNN(p, int(J), tau, h, float(lam))
                theta0 = model.net.init(rng)
                theta, res = fit_csqrnn(model, X, y, theta0, maxiter=maxiter)
                loss = model.mean_data_loss(theta, X, y)
                candidates.append((loss, restart, theta, res, model))
            loss, restart, theta, res, model = min(candidates, key=lambda item: item[0])
            df = effective_df(model, theta, mode=df_mode, active_threshold=active_threshold)
            rows.append({
                "tau": float(tau),
                "J": int(J),
                "lambda": float(lam),
                "h": float(h),
                "mean_loss": float(loss),
                "bic": bic_score(loss, len(y), df),
                "df": int(df),
                "df_mode": df_mode,
                "best_restart": int(restart),
                "n_restarts": int(max(1, n_restarts)),
                "converged": bool(res.success),
                "nit": int(getattr(res, "nit", -1)),
            })
    rows = sorted(rows, key=lambda r: r["bic"])
    return dict(rows[0]), rows


def estimate_residual_sigma(X, y, tau: float, J: int, lam: float,
                            rng: np.random.Generator,
                            h0: float = DEFAULT_H,
                            maxiter: int = 500,
                            n_sub: Optional[int] = None) -> float:
    if n_sub is not None and len(y) > n_sub:
        idx = rng.choice(len(y), int(n_sub), replace=False)
        X_fit, y_fit = X[idx], y[idx]
    else:
        X_fit, y_fit = X, y
    model = CSQRNN(X.shape[1], J, tau, h0, lam)
    theta0 = model.net.init(rng)
    theta, _ = fit_csqrnn(model, X_fit, y_fit, theta0, maxiter=maxiter)
    resid = y_fit - model.net.forward(X_fit, theta)
    return float(max(np.std(resid, ddof=1), 1e-8))


def load_hyperparameter_map(path, default_J: int = DEFAULT_J,
                            default_lambda: float = DEFAULT_LAMBDA):
    """Return lookup dicts keyed by flexible experiment identifiers."""
    if not path:
        return {}, {}, default_J, default_lambda
    rows = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    J_map = {}
    lam_map = {}
    for row in rows:
        keys = []
        base = []
        for name in ("dataset", "scenario", "dist", "error"):
            if row.get(name):
                base.append(str(row[name]))
        tau = float(row["tau"])
        keys.append(tuple(base + [tau]))
        keys.append((tau,))
        for key in keys:
            J_map[key] = int(float(row["J"]))
            lam_map[key] = float(row.get("lambda", row.get("lam")))
    return J_map, lam_map, default_J, default_lambda


def resolve_hyperparams(J_map, lam_map, fallback_J, fallback_lam,
                        tau: float, *labels):
    keys = [tuple([str(v) for v in labels if v is not None] + [float(tau)]), (float(tau),)]
    for key in keys:
        if key in J_map:
            return int(J_map[key]), float(lam_map[key])
    return int(fallback_J), float(fallback_lam)

