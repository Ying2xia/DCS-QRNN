"""Distributed DCS-QRNN and FedAvg-QRNN helpers."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import minimize

from .model import CSQRNN
from .training import fit_qrnn_huber


class Worker:
    def __init__(self, model: CSQRNN, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.N_k = len(y)

    def gradient(self, theta: np.ndarray):
        return self.model.gradient(theta, self.X, self.y)


def make_workers(model: CSQRNN, X, y, K: int, strategy: str = "sorted",
                 seed: int = 0) -> List[Worker]:
    rng = np.random.default_rng(seed)
    if strategy == "random":
        order = rng.permutation(len(y))
    elif strategy == "sorted":
        order = np.argsort(X[:, 0])
    elif strategy == "block":
        order = np.arange(len(y))
    else:
        raise ValueError(strategy)
    return [Worker(model, X[idx], y[idx]) for idx in np.array_split(order, int(K))]


def agg_grad(workers: List[Worker], theta: np.ndarray):
    N = sum(w.N_k for w in workers)
    return sum(w.N_k * w.gradient(theta) for w in workers) / N


def surrogate_step(model: CSQRNN, Xp, yp, theta, gN,
                   maxiter: int = 200, step_scale: float = 0.05):
    """One stable gradient-corrected DCS update.

    The exact corrected pilot loss contains a linear term. For a nonconvex
    neural network with an unpenalized output layer, directly minimizing that
    objective can be numerically unbounded. We therefore use the same stable
    implementation as the original simulation code: take a normalized global
    gradient step from the pilot estimator, then re-optimize the valid pilot
    loss from that warm start.
    """
    g_norm = float(np.linalg.norm(gN))
    if not np.isfinite(g_norm) or g_norm < 1e-12:
        return theta.copy()
    theta_warm = theta - float(step_scale) * gN / g_norm
    res = minimize(
        fun=lambda t: model.loss(t, Xp, yp),
        jac=lambda t: model.gradient(t, Xp, yp),
        x0=theta_warm,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7},
    )
    return res.x if np.all(np.isfinite(res.x)) else theta_warm


class DCSQRNNMaster:
    def __init__(self, model: CSQRNN, pilot_ratio: float = 0.10,
                 T: int = 1, seed: int = 0):
        self.model = model
        self.pilot_ratio = float(pilot_ratio)
        self.T = int(T)
        self.seed = int(seed)
        self.theta_ = None

    def sample_pilot(self, X, y):
        rng = np.random.default_rng(self.seed + 100_003)
        mask = rng.random(len(y)) < self.pilot_ratio
        min_pilot = min(len(y), max(10, int(round(self.pilot_ratio * len(y)))))
        if int(mask.sum()) < min_pilot:
            idx_fill = rng.choice(len(y), size=min_pilot, replace=False)
            mask = np.zeros(len(y), dtype=bool)
            mask[idx_fill] = True
        idx = np.flatnonzero(mask)
        return X[idx], y[idx]

    def fit_pilot(self, Xp, yp, theta0):
        res = minimize(
            fun=lambda t: self.model.loss(t, Xp, yp),
            jac=lambda t: self.model.gradient(t, Xp, yp),
            x0=theta0,
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7},
        )
        return res.x

    def fit(self, workers: List[Worker], X_all, y_all, theta0):
        Xp, yp = self.sample_pilot(X_all, y_all)
        theta = self.fit_pilot(Xp, yp, theta0.copy())
        for _ in range(self.T):
            gN = agg_grad(workers, theta)
            theta = surrogate_step(self.model, Xp, yp, theta, gN)
        self.theta_ = theta
        self.pilot_size_ = int(len(yp))
        return self

    def predict(self, X):
        if self.theta_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model.net.forward(X, self.theta_)


def fit_fedavg_qrnn(net, tau: float, lam: float, workers_data,
                    theta0, n_rounds: int = 10, local_maxiter: int = 20,
                    eps_schedule=None):
    if eps_schedule is None:
        eps_schedule = np.geomspace(0.5, 0.001, num=int(n_rounds))
    weights = np.array([len(y) for _, y in workers_data], dtype=float)
    weights /= weights.sum()
    theta = theta0.copy()
    for r in range(int(n_rounds)):
        eps = float(eps_schedule[min(r, len(eps_schedule) - 1)])
        local = []
        for Xk, yk in workers_data:
            th, _ = fit_qrnn_huber(net, tau, lam, Xk, yk, theta,
                                   eps_schedule=[eps], maxiter=local_maxiter)
            local.append(th)
        theta = np.average(np.vstack(local), axis=0, weights=weights)
    return theta, {"rounds": int(n_rounds), "local_maxiter": int(local_maxiter)}
