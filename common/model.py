"""Single-hidden-layer QRNN and convolution-smoothed quantile loss."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as _norm


class QRNNLayer:
    """Single-hidden-layer feedforward network with tanh hidden activation."""

    def __init__(self, p: int, J: int):
        self.p = int(p)
        self.J = int(J)
        self.d = self.J * self.p + self.J + self.J + 1

    def unpack(self, theta: np.ndarray):
        p, J = self.p, self.J
        i = 0
        W_h = theta[i:i + J * p].reshape(J, p)
        i += J * p
        b_h = theta[i:i + J]
        i += J
        W_o = theta[i:i + J]
        i += J
        b_o = float(theta[i])
        return W_h, b_h, W_o, b_o

    def init(self, rng: np.random.Generator) -> np.ndarray:
        theta = np.zeros(self.d)
        theta[:self.J * self.p] = rng.normal(0, 1 / np.sqrt(self.p), self.J * self.p)
        lo = self.J * self.p + self.J
        theta[lo:lo + self.J] = rng.normal(0, 1 / np.sqrt(self.J), self.J)
        return theta

    def forward(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        W_h, b_h, W_o, b_o = self.unpack(theta)
        H = np.tanh(X @ W_h.T + b_h)
        return H @ W_o + b_o

    def jacobian(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        W_h, b_h, W_o, _ = self.unpack(theta)
        n, p = X.shape
        J = self.J
        H = np.tanh(X @ W_h.T + b_h)
        dH = 1.0 - H ** 2
        G = np.empty((n, self.d))
        i = 0
        G[:, i:i + J * p] = (
            W_o[None, :, None] * dH[:, :, None] * X[:, None, :]
        ).reshape(n, J * p)
        i += J * p
        G[:, i:i + J] = W_o[None, :] * dH
        i += J
        G[:, i:i + J] = H
        i += J
        G[:, i] = 1.0
        return G


def cs_loss(e: np.ndarray, tau: float, h: float) -> np.ndarray:
    h = max(float(h), 1e-8)
    return (tau - _norm.cdf(-e / h)) * e + h * _norm.pdf(e / h)


def cs_score(e: np.ndarray, tau: float, h: float) -> np.ndarray:
    h = max(float(h), 1e-8)
    return tau - _norm.cdf(-e / h)


def _huber(u: np.ndarray, eps: float) -> np.ndarray:
    au = np.abs(u)
    return np.where(au <= eps, au ** 2 / (2 * eps), au - eps / 2)


def huber_loss(e: np.ndarray, tau: float, eps: float) -> np.ndarray:
    w = np.where(e >= 0, tau, 1.0 - tau)
    return w * _huber(e, eps)


def huber_score(e: np.ndarray, tau: float, eps: float) -> np.ndarray:
    dh = np.where(np.abs(e) <= eps, e / eps, np.sign(e))
    w = np.where(e >= 0, tau, 1.0 - tau)
    return w * dh


class CSQRNN:
    """Regularised convolution-smoothed QRNN objective."""

    def __init__(self, p: int, J: int, tau: float,
                 h: float = 0.1, lam: float = 0.01):
        self.net = QRNNLayer(p, J)
        self.p = int(p)
        self.J = int(J)
        self.tau = float(tau)
        self.h = float(h)
        self.lam = float(lam)

    def _penalty(self, theta: np.ndarray):
        dWh = self.J * self.p
        c = self.lam / max(self.p * self.J, 1)
        Wh = theta[:dWh]
        val = c * float(Wh @ Wh)
        grad = np.zeros_like(theta)
        grad[:dWh] = 2.0 * c * Wh
        return val, grad

    def loss(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        e = y - self.net.forward(X, theta)
        pen, _ = self._penalty(theta)
        return float(np.mean(cs_loss(e, self.tau, self.h))) + pen

    def mean_data_loss(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        e = y - self.net.forward(X, theta)
        return float(np.mean(cs_loss(e, self.tau, self.h)))

    def gradient(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(y)
        e = y - self.net.forward(X, theta)
        psi = cs_score(e, self.tau, self.h)
        G = self.net.jacobian(X, theta)
        _, gpen = self._penalty(theta)
        return -(psi @ G) / n + gpen

