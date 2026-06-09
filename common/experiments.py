"""Reusable experiment runners for one dataset split and one tau."""

from __future__ import annotations

import time

import numpy as np

from .baselines import fit_catboost_quantile, fit_elm_qr, fit_rvfl_qr
from .distributed import DCSQRNNMaster, Worker, fit_fedavg_qrnn, make_workers
from .metrics import mae, rmse
from .model import CSQRNN
from .training import fit_csqrnn, fit_linear_qr_huber, fit_qrnn_huber, predict_linear_qr


def evaluate_prediction(y_true, y_pred, prefix: str = ""):
    return {
        f"{prefix}mae": mae(y_true, y_pred),
        f"{prefix}rmse": rmse(y_true, y_pred),
    }


def _backend_device(backend: str, device: str):
    if str(backend).lower() == "numpy":
        return "numpy", "cpu"
    from .torch_backend import resolve_backend
    return resolve_backend(backend, device)


def run_csqrnn(X_tr, y_tr, X_te, y_eval, tau: float, J: int, lam: float,
               h: float, seed: int, maxiter: int = 2000,
               backend: str = "numpy", device: str = "auto",
               torch_dtype: str = "float32", torch_lr: float = 0.01,
               torch_check_every: int = 25, torch_maxiter=None,
               torch_lbfgs_steps: int = 400, torch_lbfgs_lr: float = 0.8,
               torch_lbfgs_history_size: int = 20):
    model = CSQRNN(X_tr.shape[1], J, tau, h, lam)
    theta0 = model.net.init(np.random.default_rng(seed))
    t0 = time.perf_counter()
    actual_backend, actual_device = _backend_device(backend, device)
    if actual_backend == "torch":
        from .torch_backend import fit_csqrnn_torch
        theta, res = fit_csqrnn_torch(
            model, X_tr, y_tr, theta0,
            maxiter=int(torch_maxiter or maxiter),
            device=actual_device,
            dtype_name=torch_dtype,
            lr=torch_lr,
            check_every=torch_check_every,
            lbfgs_steps=torch_lbfgs_steps,
            lbfgs_lr=torch_lbfgs_lr,
            lbfgs_history_size=torch_lbfgs_history_size,
        )
    else:
        theta, res = fit_csqrnn(model, X_tr, y_tr, theta0, maxiter=maxiter)
    elapsed = time.perf_counter() - t0
    pred = model.net.forward(X_te, theta)
    out = evaluate_prediction(y_eval, pred)
    out.update({"time": elapsed, "converged": bool(res.success), "nit": int(getattr(res, "nit", -1)), "backend": actual_backend, "device": actual_device})
    return out, theta


def run_qrnn(X_tr, y_tr, X_te, y_eval, tau: float, J: int, lam: float,
             seed: int, maxiter: int = 500, backend: str = "numpy",
             device: str = "auto", torch_dtype: str = "float32",
             torch_lr: float = 0.01, torch_check_every: int = 25,
             torch_maxiter=None, torch_lbfgs_steps: int = 400,
             torch_lbfgs_lr: float = 0.8,
             torch_lbfgs_history_size: int = 20):
    model = CSQRNN(X_tr.shape[1], J, tau, h=0.1, lam=lam)
    theta0 = model.net.init(np.random.default_rng(seed))
    t0 = time.perf_counter()
    actual_backend, actual_device = _backend_device(backend, device)
    if actual_backend == "torch":
        from .torch_backend import fit_qrnn_huber_torch
        theta, res = fit_qrnn_huber_torch(
            model.net, tau, lam, X_tr, y_tr, theta0,
            maxiter=int(torch_maxiter or maxiter),
            device=actual_device,
            dtype_name=torch_dtype,
            lr=torch_lr,
            check_every=torch_check_every,
            lbfgs_steps=torch_lbfgs_steps,
            lbfgs_lr=torch_lbfgs_lr,
            lbfgs_history_size=torch_lbfgs_history_size,
        )
    else:
        theta, res = fit_qrnn_huber(model.net, tau, lam, X_tr, y_tr, theta0, maxiter=maxiter)
    elapsed = time.perf_counter() - t0
    pred = model.net.forward(X_te, theta)
    out = evaluate_prediction(y_eval, pred)
    out.update({"time": elapsed, "converged": bool(res.success), "nit": int(getattr(res, "nit", -1)), "backend": actual_backend, "device": actual_device})
    return out


def run_linear_qr(X_tr, y_tr, X_te, y_eval, tau: float):
    t0 = time.perf_counter()
    beta, res = fit_linear_qr_huber(X_tr, y_tr, tau)
    elapsed = time.perf_counter() - t0
    pred = predict_linear_qr(X_te, beta)
    out = evaluate_prediction(y_eval, pred)
    out.update({"time": elapsed, "converged": bool(res.success), "nit": int(getattr(res, "nit", -1))})
    return out


def run_catboost(X_tr, y_tr, X_te, y_eval, tau: float, seed: int,
                 cfg: dict):
    t0 = time.perf_counter()
    model = fit_catboost_quantile(
        X_tr,
        y_tr,
        tau,
        iterations=cfg.get("catboost_iterations", 300),
        depth=cfg.get("catboost_depth", 6),
        learning_rate=cfg.get("catboost_lr", 0.05),
        l2_leaf_reg=cfg.get("catboost_l2", 3.0),
        seed=seed,
        task_type=cfg.get("catboost_task_type", "CPU"),
        devices=cfg.get("catboost_devices", "0"),
    )
    elapsed = time.perf_counter() - t0
    pred = model.predict(X_te)
    out = evaluate_prediction(y_eval, pred)
    out.update({"time": elapsed, "converged": True, "nit": cfg.get("catboost_iterations", 300)})
    return out


def run_elm(X_tr, y_tr, X_te, y_eval, tau: float, seed: int, cfg: dict):
    t0 = time.perf_counter()
    model, res = fit_elm_qr(
        X_tr,
        y_tr,
        tau,
        n_hidden=cfg.get("rf_hidden", 50),
        lam=cfg.get("rf_lam", 1e-4),
        eps=cfg.get("rf_eps", 0.01),
        seed=seed,
        maxiter=cfg.get("rf_maxiter", 1000),
    )
    elapsed = time.perf_counter() - t0
    pred = model.predict(X_te)
    out = evaluate_prediction(y_eval, pred)
    out.update({"time": elapsed, "converged": bool(res.success), "nit": int(getattr(res, "nit", -1))})
    return out


def run_rvfl(X_tr, y_tr, X_te, y_eval, tau: float, seed: int, cfg: dict):
    t0 = time.perf_counter()
    model, res = fit_rvfl_qr(
        X_tr,
        y_tr,
        tau,
        n_hidden=cfg.get("rf_hidden", 50),
        lam=cfg.get("rf_lam", 1e-4),
        eps=cfg.get("rf_eps", 0.01),
        seed=seed,
        maxiter=cfg.get("rf_maxiter", 1000),
    )
    elapsed = time.perf_counter() - t0
    pred = model.predict(X_te)
    out = evaluate_prediction(y_eval, pred)
    out.update({"time": elapsed, "converged": bool(res.success), "nit": int(getattr(res, "nit", -1))})
    return out


def run_dcsqrnn(X_tr, y_tr, X_te, y_eval, tau: float, J: int, lam: float,
                h: float, K: int, pilot_ratio: float, T: int,
                seed: int, maxiter_ref: int = 2000,
                backend: str = "numpy", device: str = "auto",
                torch_dtype: str = "float32", torch_lr: float = 0.01,
                torch_check_every: int = 25, torch_maxiter=None,
                torch_dcs_step_maxiter=None, torch_lbfgs_steps: int = 400,
                torch_lbfgs_lr: float = 0.8,
                torch_lbfgs_history_size: int = 20):
    model = CSQRNN(X_tr.shape[1], J, tau, h, lam)
    theta0 = model.net.init(np.random.default_rng(seed))
    t0 = time.perf_counter()
    actual_backend, actual_device = _backend_device(backend, device)
    if actual_backend == "torch":
        from .torch_backend import fit_dcsqrnn_torch
        theta, info = fit_dcsqrnn_torch(
            model, X_tr, y_tr, theta0, K, pilot_ratio, T, seed,
            maxiter_pilot=int(torch_maxiter or maxiter_ref),
            maxiter_step=int(torch_dcs_step_maxiter or max(100, int((torch_maxiter or maxiter_ref) // 2))),
            device=actual_device,
            dtype_name=torch_dtype,
            lr=torch_lr,
            check_every=torch_check_every,
            lbfgs_steps=torch_lbfgs_steps,
            lbfgs_lr=torch_lbfgs_lr,
            lbfgs_history_size=torch_lbfgs_history_size,
        )
        pred = model.net.forward(X_te, theta)
    else:
        workers = make_workers(model, X_tr, y_tr, K, strategy="sorted", seed=seed)
        master = DCSQRNNMaster(model, pilot_ratio=pilot_ratio, T=T, seed=seed)
        master.fit(workers, X_tr, y_tr, theta0)
        info = {"rounds": int(T), "pilot_size": getattr(master, "pilot_size_", "")}
        pred = master.predict(X_te)
    elapsed = time.perf_counter() - t0
    out = evaluate_prediction(y_eval, pred)
    out.update({
        "time": elapsed,
        "pilot_ratio": float(pilot_ratio),
        "pilot_size": info.get("pilot_size", ""),
        "rounds": int(info.get("rounds", T)),
        "backend": actual_backend,
        "device": actual_device,
    })
    return out


def run_dcsqrnn_workers(workers_data, X_te, y_eval, tau: float, J: int,
                        lam: float, h: float, pilot_ratio: float, T: int,
                        seed: int, backend: str = "numpy", device: str = "auto",
                        torch_dtype: str = "float32", torch_lr: float = 0.01,
                        torch_check_every: int = 25, torch_maxiter=None,
                        torch_dcs_step_maxiter=None, torch_lbfgs_steps: int = 400,
                        torch_lbfgs_lr: float = 0.8,
                        torch_lbfgs_history_size: int = 20,
                        maxiter_ref: int = 2000):
    X_all = np.vstack([Xk for Xk, _ in workers_data])
    y_all = np.concatenate([yk for _, yk in workers_data])
    model = CSQRNN(X_all.shape[1], J, tau, h, lam)
    theta0 = model.net.init(np.random.default_rng(seed))
    t0 = time.perf_counter()
    actual_backend, actual_device = _backend_device(backend, device)
    if actual_backend == "torch":
        from .torch_backend import fit_dcsqrnn_torch
        theta, info = fit_dcsqrnn_torch(
            model, X_all, y_all, theta0, len(workers_data), pilot_ratio, T, seed,
            maxiter_pilot=int(torch_maxiter or maxiter_ref),
            maxiter_step=int(torch_dcs_step_maxiter or max(100, int((torch_maxiter or maxiter_ref) // 2))),
            device=actual_device,
            dtype_name=torch_dtype,
            lr=torch_lr,
            check_every=torch_check_every,
            lbfgs_steps=torch_lbfgs_steps,
            lbfgs_lr=torch_lbfgs_lr,
            lbfgs_history_size=torch_lbfgs_history_size,
        )
        pred = model.net.forward(X_te, theta)
    else:
        workers = [Worker(model, Xk, yk) for Xk, yk in workers_data]
        master = DCSQRNNMaster(model, pilot_ratio=pilot_ratio, T=T, seed=seed)
        master.fit(workers, X_all, y_all, theta0)
        info = {"rounds": int(T), "pilot_size": getattr(master, "pilot_size_", "")}
        pred = master.predict(X_te)
    elapsed = time.perf_counter() - t0
    out = evaluate_prediction(y_eval, pred)
    out.update({
        "time": elapsed,
        "pilot_ratio": float(pilot_ratio),
        "pilot_size": info.get("pilot_size", ""),
        "rounds": int(info.get("rounds", T)),
        "backend": actual_backend,
        "device": actual_device,
    })
    return out


def run_fedavg(X_tr, y_tr, X_te, y_eval, tau: float, J: int, lam: float,
               K: int, seed: int, cfg: dict):
    model = CSQRNN(X_tr.shape[1], J, tau, h=0.1, lam=lam)
    theta0 = model.net.init(np.random.default_rng(seed))
    workers = make_workers(model, X_tr, y_tr, K, strategy="sorted", seed=seed)
    workers_data = [(w.X, w.y) for w in workers]
    t0 = time.perf_counter()
    actual_backend, actual_device = _backend_device(cfg.get("backend", "numpy"), cfg.get("device", "auto"))
    if actual_backend == "torch":
        from .torch_backend import fit_fedavg_qrnn_torch
        theta, info = fit_fedavg_qrnn_torch(
            model.net,
            tau,
            lam,
            workers_data,
            theta0,
            n_rounds=cfg.get("fedavg_rounds", 10),
            local_maxiter=cfg.get("fedavg_local_iter", 20),
            device=actual_device,
            dtype_name=cfg.get("torch_dtype", "float32"),
            lr=cfg.get("torch_lr", 0.01),
            check_every=cfg.get("torch_check_every", 10),
            lbfgs_steps=cfg.get("torch_fedavg_lbfgs_steps", 0),
            lbfgs_lr=cfg.get("torch_lbfgs_lr", 0.8),
            lbfgs_history_size=cfg.get("torch_lbfgs_history_size", 20),
        )
    else:
        theta, info = fit_fedavg_qrnn(
            model.net,
            tau,
            lam,
            workers_data,
            theta0,
            n_rounds=cfg.get("fedavg_rounds", 10),
            local_maxiter=cfg.get("fedavg_local_iter", 20),
        )
    elapsed = time.perf_counter() - t0
    pred = model.net.forward(X_te, theta)
    out = evaluate_prediction(y_eval, pred)
    out.update({"time": elapsed, "rounds": int(info["rounds"]), "backend": actual_backend, "device": actual_device})
    return out


def run_fedavg_workers(workers_data, X_te, y_eval, tau: float, J: int,
                       lam: float, seed: int, cfg: dict):
    p = workers_data[0][0].shape[1]
    model = CSQRNN(p, J, tau, h=0.1, lam=lam)
    theta0 = model.net.init(np.random.default_rng(seed))
    t0 = time.perf_counter()
    actual_backend, actual_device = _backend_device(cfg.get("backend", "numpy"), cfg.get("device", "auto"))
    if actual_backend == "torch":
        from .torch_backend import fit_fedavg_qrnn_torch
        theta, info = fit_fedavg_qrnn_torch(
            model.net,
            tau,
            lam,
            workers_data,
            theta0,
            n_rounds=cfg.get("fedavg_rounds", 10),
            local_maxiter=cfg.get("fedavg_local_iter", 20),
            device=actual_device,
            dtype_name=cfg.get("torch_dtype", "float32"),
            lr=cfg.get("torch_lr", 0.01),
            check_every=cfg.get("torch_check_every", 10),
            lbfgs_steps=cfg.get("torch_fedavg_lbfgs_steps", 0),
            lbfgs_lr=cfg.get("torch_lbfgs_lr", 0.8),
            lbfgs_history_size=cfg.get("torch_lbfgs_history_size", 20),
        )
    else:
        theta, info = fit_fedavg_qrnn(
            model.net,
            tau,
            lam,
            workers_data,
            theta0,
            n_rounds=cfg.get("fedavg_rounds", 10),
            local_maxiter=cfg.get("fedavg_local_iter", 20),
        )
    elapsed = time.perf_counter() - t0
    pred = model.net.forward(X_te, theta)
    out = evaluate_prediction(y_eval, pred)
    out.update({"time": elapsed, "rounds": int(info["rounds"]), "backend": actual_backend, "device": actual_device})
    return out
