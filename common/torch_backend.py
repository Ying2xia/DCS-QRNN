"""Optional PyTorch/CUDA backend for the expensive neural-network fits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import CSQRNN, QRNNLayer


_TORCH = None
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def import_torch():
    global _TORCH
    if _TORCH is not None:
        return _TORCH
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this environment.") from exc
    _TORCH = torch
    return _TORCH


def resolve_backend(backend: str = "auto", device: str = "auto"):
    """Return (backend, device), falling back to NumPy when auto cannot use CUDA."""
    backend = str(backend).lower()
    device = str(device).lower()
    if backend == "numpy":
        return "numpy", "cpu"
    if backend not in {"auto", "torch"}:
        raise ValueError(f"unknown backend: {backend}")
    try:
        torch = import_torch()
    except RuntimeError:
        if backend == "auto":
            return "numpy", "cpu"
        raise
    if device == "auto":
        if torch.cuda.is_available():
            return "torch", "cuda"
        if backend == "auto":
            return "numpy", "cpu"
        return "torch", "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        if backend == "auto":
            return "numpy", "cpu"
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return "torch", device


def _dtype(name: str):
    torch = import_torch()
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"unknown torch dtype: {name}")


def _tensor(x, device: str, dtype_name: str):
    torch = import_torch()
    return torch.as_tensor(np.asarray(x), dtype=_dtype(dtype_name), device=device)


def _unpack(theta, p: int, J: int):
    i = 0
    W_h = theta[i:i + J * p].reshape(J, p)
    i += J * p
    b_h = theta[i:i + J]
    i += J
    W_o = theta[i:i + J]
    i += J
    b_o = theta[i]
    return W_h, b_h, W_o, b_o


def _forward(X, theta, p: int, J: int):
    W_h, b_h, W_o, b_o = _unpack(theta, p, J)
    H = import_torch().tanh(X @ W_h.T + b_h)
    return H @ W_o + b_o


def _normal_pdf(x):
    torch = import_torch()
    return torch.exp(-0.5 * x * x) / _SQRT2PI


def _normal_cdf(x):
    torch = import_torch()
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _cs_loss(e, tau: float, h: float):
    h = max(float(h), 1e-8)
    a = e / h
    return (float(tau) - _normal_cdf(-a)) * e + h * _normal_pdf(a)


def _huber_loss(e, tau: float, eps: float):
    torch = import_torch()
    eps = max(float(eps), 1e-8)
    ae = torch.abs(e)
    base = torch.where(ae <= eps, e * e / (2.0 * eps), ae - 0.5 * eps)
    return torch.where(e >= 0.0, float(tau) * base, (1.0 - float(tau)) * base)


def _objective(theta, X, y, p: int, J: int, tau: float, lam: float,
               h: float, loss_kind: str, epsilon: float):
    pred = _forward(X, theta, p, J)
    e = y - pred
    if loss_kind == "cs":
        loss = _cs_loss(e, tau, h).mean()
    elif loss_kind == "huber":
        loss = _huber_loss(e, tau, epsilon).mean()
    else:
        raise ValueError(loss_kind)
    if lam > 0.0:
        W_h, _, _, _ = _unpack(theta, p, J)
        loss = loss + float(lam) / max(p * J, 1) * (W_h * W_h).sum()
    return loss


@dataclass
class TorchOptResult:
    success: bool
    nit: int
    fun: float
    message: str


def fit_objective_torch(X, y, theta0, p: int, J: int, tau: float, lam: float,
                        h: float, loss_kind: str, epsilon: float,
                        maxiter: int, device: str, dtype_name: str = "float32",
                        lr: float = 0.01, check_every: int = 25,
                        lbfgs_steps: int = 400, lbfgs_lr: float = 0.8,
                        lbfgs_history_size: int = 20):
    """Full-batch Adam warm-up plus optional L-BFGS refinement."""
    torch = import_torch()
    X_t = _tensor(X, device, dtype_name)
    y_t = _tensor(y, device, dtype_name)
    theta = torch.nn.Parameter(_tensor(theta0, device, dtype_name).clone())
    opt = torch.optim.Adam([theta], lr=float(lr))

    best_theta = theta.detach().clone()
    best_obj = float("inf")
    last_obj = None
    success = False
    nit = 0

    for step in range(1, int(maxiter) + 1):
        nit = step
        opt.zero_grad(set_to_none=True)
        loss = _objective(theta, X_t, y_t, p, J, tau, lam, h, loss_kind, epsilon)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=100.0)
        opt.step()

        if step % int(check_every) == 0 or step == int(maxiter):
            obj = float(loss.detach().cpu().item())
            if obj < best_obj:
                best_obj = obj
                best_theta = theta.detach().clone()
            if last_obj is not None and abs(last_obj - obj) <= 1e-7 * (1.0 + abs(last_obj)):
                success = True
                break
            last_obj = obj

    message = "torch Adam"
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
            obj = _objective(theta, X_t, y_t, p, J, tau, lam, h, loss_kind, epsilon)
            obj.backward()
            return obj

        lbfgs.step(closure)
        with torch.no_grad():
            final_obj = float(_objective(theta, X_t, y_t, p, J, tau, lam, h, loss_kind, epsilon).detach().cpu().item())
            if final_obj <= best_obj:
                best_obj = final_obj
                best_theta = theta.detach().clone()
        nit += int(lbfgs_steps)
        success = True
        message = "torch Adam + L-BFGS"

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return (
        best_theta.detach().cpu().numpy().astype(float),
        TorchOptResult(bool(success), int(nit), float(best_obj), message),
    )


def fit_csqrnn_torch(model: CSQRNN, X, y, theta0, maxiter: int = 800,
                     device: str = "cuda", dtype_name: str = "float32",
                     lr: float = 0.01, check_every: int = 25,
                     lbfgs_steps: int = 400, lbfgs_lr: float = 0.8,
                     lbfgs_history_size: int = 20):
    return fit_objective_torch(
        X, y, theta0, model.p, model.J, model.tau, model.lam, model.h,
        "cs", 0.01, maxiter, device, dtype_name, lr, check_every,
        lbfgs_steps, lbfgs_lr, lbfgs_history_size,
    )


def fit_qrnn_huber_torch(net: QRNNLayer, tau: float, lam: float, X, y,
                         theta0, eps_schedule=None, maxiter: int = 200,
                         device: str = "cuda", dtype_name: str = "float32",
                         lr: float = 0.01, check_every: int = 25,
                         lbfgs_steps: int = 400, lbfgs_lr: float = 0.8,
                         lbfgs_history_size: int = 20):
    if eps_schedule is None:
        eps_schedule = [0.5, 0.2, 0.05, 0.01, 0.001]
    theta = np.asarray(theta0, dtype=float)
    res = None
    last_idx = len(eps_schedule) - 1
    for idx, eps in enumerate(eps_schedule):
        theta, res = fit_objective_torch(
            X, y, theta, net.p, net.J, tau, lam, 0.1, "huber", float(eps),
            maxiter, device, dtype_name, lr, check_every,
            lbfgs_steps if idx == last_idx else 0,
            lbfgs_lr,
            lbfgs_history_size,
        )
    return theta, res


def gradient_csqrnn_torch(model: CSQRNN, theta_np, X, y,
                          device: str = "cuda", dtype_name: str = "float32"):
    torch = import_torch()
    X_t = _tensor(X, device, dtype_name)
    y_t = _tensor(y, device, dtype_name)
    theta = _tensor(theta_np, device, dtype_name).clone().detach()
    theta.requires_grad_(True)
    loss = _objective(theta, X_t, y_t, model.p, model.J, model.tau, model.lam, model.h, "cs", 0.01)
    loss.backward()
    grad = theta.grad.detach().cpu().numpy().astype(float)
    del X_t, y_t, theta, loss
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return grad


def fit_dcsqrnn_torch(model: CSQRNN, X, y, theta0, K: int,
                      pilot_ratio: float, T: int, seed: int,
                      maxiter_pilot: int = 800, maxiter_step: int = 400,
                      device: str = "cuda", dtype_name: str = "float32",
                      lr: float = 0.01, check_every: int = 25,
                      step_scale: float = 0.05, lbfgs_steps: int = 400,
                      lbfgs_lr: float = 0.8, lbfgs_history_size: int = 20):
    rng = np.random.default_rng(seed + 100_003)
    mask = rng.random(len(y)) < float(pilot_ratio)
    min_pilot = min(len(y), max(10, int(round(float(pilot_ratio) * len(y)))))
    if int(mask.sum()) < min_pilot:
        idx_fill = rng.choice(len(y), size=min_pilot, replace=False)
        mask = np.zeros(len(y), dtype=bool)
        mask[idx_fill] = True
    idx = np.flatnonzero(mask)
    n_pilot = len(idx)
    Xp, yp = X[idx], y[idx]
    theta, res = fit_csqrnn_torch(
        model, Xp, yp, theta0, maxiter=maxiter_pilot, device=device,
        dtype_name=dtype_name, lr=lr, check_every=check_every,
        lbfgs_steps=lbfgs_steps, lbfgs_lr=lbfgs_lr,
        lbfgs_history_size=lbfgs_history_size,
    )
    for _ in range(int(T)):
        g_global = gradient_csqrnn_torch(model, theta, X, y, device=device, dtype_name=dtype_name)
        g_norm = float(np.linalg.norm(g_global))
        if np.isfinite(g_norm) and g_norm >= 1e-12:
            theta_start = theta - float(step_scale) * g_global / g_norm
        else:
            theta_start = theta.copy()
        theta, res = fit_csqrnn_torch(
            model, Xp, yp, theta_start, maxiter=maxiter_step, device=device,
            dtype_name=dtype_name, lr=lr, check_every=check_every,
            lbfgs_steps=lbfgs_steps, lbfgs_lr=lbfgs_lr,
            lbfgs_history_size=lbfgs_history_size,
        )
    return theta, {"rounds": int(T), "pilot_size": int(n_pilot), "sampling": "poisson", "result": res}


def fit_fedavg_qrnn_torch(net: QRNNLayer, tau: float, lam: float,
                          workers_data, theta0, n_rounds: int = 10,
                          local_maxiter: int = 20, device: str = "cuda",
                          dtype_name: str = "float32", lr: float = 0.01,
                          check_every: int = 10, lbfgs_steps: int = 0,
                          lbfgs_lr: float = 0.8,
                          lbfgs_history_size: int = 20):
    eps_schedule = np.geomspace(0.5, 0.001, num=int(n_rounds))
    weights = np.array([len(y) for _, y in workers_data], dtype=float)
    weights /= weights.sum()
    theta = np.asarray(theta0, dtype=float)
    last_res = None
    for r in range(int(n_rounds)):
        eps = float(eps_schedule[min(r, len(eps_schedule) - 1)])
        local = []
        for Xk, yk in workers_data:
            th, last_res = fit_qrnn_huber_torch(
                net, tau, lam, Xk, yk, theta, eps_schedule=[eps],
                maxiter=local_maxiter, device=device, dtype_name=dtype_name,
                lr=lr, check_every=check_every, lbfgs_steps=lbfgs_steps,
                lbfgs_lr=lbfgs_lr, lbfgs_history_size=lbfgs_history_size,
            )
            local.append(th)
        theta = np.average(np.vstack(local), axis=0, weights=weights)
    return theta, {"rounds": int(n_rounds), "local_maxiter": int(local_maxiter), "result": last_res}
