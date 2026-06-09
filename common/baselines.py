"""Modern baseline methods for quantile prediction."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .model import huber_loss, huber_score


def fit_catboost_quantile(X, y, tau: float, iterations: int = 300,
                          depth: int = 6, learning_rate: float = 0.05,
                          l2_leaf_reg: float = 3.0, seed: int = 0,
                          thread_count: int = 1, task_type: str = "CPU",
                          devices: str = "0"):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("CatBoost-Quantile requires `pip install catboost`.") from exc
    params = dict(
        loss_function=f"Quantile:alpha={tau}",
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=float(learning_rate),
        l2_leaf_reg=float(l2_leaf_reg),
        random_seed=int(seed),
        allow_writing_files=False,
        verbose=False,
    )
    if str(task_type).upper() == "GPU":
        params.update(task_type="GPU", devices=str(devices))
    else:
        params.update(task_type="CPU", thread_count=int(thread_count))
    model = CatBoostRegressor(**params)
    model.fit(X, y)
    return model


class RandomFeatureQR:
    def __init__(self, W, b, beta, include_linear: bool):
        self.W = W
        self.b = b
        self.beta = beta
        self.include_linear = include_linear

    def features(self, X):
        H = np.tanh(X @ self.W.T + self.b)
        if self.include_linear:
            return np.hstack([X, H, np.ones((X.shape[0], 1))])
        return np.hstack([H, np.ones((X.shape[0], 1))])

    def predict(self, X):
        return self.features(X) @ self.beta


def _fit_quantile_output(Phi, y, tau: float, lam: float,
                         eps: float, maxiter: int):
    n, d = Phi.shape
    beta0, *_ = np.linalg.lstsq(Phi, y, rcond=None)

    def obj(beta):
        e = y - Phi @ beta
        return float(np.mean(huber_loss(e, tau, eps))) + lam * float(beta[:-1] @ beta[:-1])

    def grd(beta):
        e = y - Phi @ beta
        g = -(huber_score(e, tau, eps) @ Phi) / n
        gpen = np.zeros(d)
        gpen[:-1] = 2.0 * lam * beta[:-1]
        return g + gpen

    res = minimize(
        obj,
        jac=grd,
        x0=beta0,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "ftol": 1e-10, "gtol": 1e-6},
    )
    return res.x, res


def _random_layer(p: int, n_hidden: int, seed: int):
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0 / np.sqrt(max(p, 1)), size=(int(n_hidden), p))
    b = rng.uniform(-1.0, 1.0, size=int(n_hidden))
    return W, b


def fit_elm_qr(X, y, tau: float, n_hidden: int = 50,
               lam: float = 1e-4, eps: float = 0.01,
               seed: int = 0, maxiter: int = 1000):
    W, b = _random_layer(X.shape[1], n_hidden, seed)
    model = RandomFeatureQR(W, b, None, include_linear=False)
    beta, res = _fit_quantile_output(model.features(X), y, tau, lam, eps, maxiter)
    return RandomFeatureQR(W, b, beta, include_linear=False), res


def fit_rvfl_qr(X, y, tau: float, n_hidden: int = 50,
                lam: float = 1e-4, eps: float = 0.01,
                seed: int = 0, maxiter: int = 1000):
    W, b = _random_layer(X.shape[1], n_hidden, seed)
    model = RandomFeatureQR(W, b, None, include_linear=True)
    beta, res = _fit_quantile_output(model.features(X), y, tau, lam, eps, maxiter)
    return RandomFeatureQR(W, b, beta, include_linear=True), res
