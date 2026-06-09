"""Simulation DGPs and real-data loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import chi2 as _chi2
from scipy.stats import norm as _norm
from scipy.stats import t as _t

from .config import LEGACY_DATA_DIR


def _q(tau: float, dist: str) -> float:
    if dist == "N01":
        return float(_norm.ppf(tau))
    if dist == "t3":
        return float(_t.ppf(tau, df=3))
    if dist == "chi2_2":
        return float(_chi2.ppf(tau, df=2))
    raise ValueError(dist)


def _draw(N: int, dist: str, rng: np.random.Generator) -> np.ndarray:
    if dist == "N01":
        return rng.standard_normal(N)
    if dist == "t3":
        return rng.standard_t(df=3, size=N)
    if dist == "chi2_2":
        return rng.chisquare(df=2, size=N)
    raise ValueError(dist)


def gen_ex21(N: int, tau: float, dist: str, rng: np.random.Generator):
    X1 = rng.uniform(0.0, 1.0, N)
    X2 = rng.uniform(0.0, 1.0, N)
    X = np.stack([X1, X2], axis=1)
    m = np.sin(np.pi * X1) + np.sin(np.pi * X2)
    eps = _draw(N, dist, rng)
    return X, m + eps, m + _q(tau, dist)


def gen_ex22(N: int, tau: float, dist: str, rng: np.random.Generator):
    x = rng.uniform(-4.0, 4.0, N)
    sig = (1.0 + 0.2 * x) / 5.0
    m = (1.0 - x + 2.0 * x ** 2) * np.exp(-x ** 2)
    eps = _draw(N, dist, rng)
    return x[:, None], m + sig * eps, m + sig * _q(tau, dist)


def gen_ex23(N: int, tau: float, dist: str, rng: np.random.Generator):
    X1 = rng.uniform(0.0, 1.0, N)
    X2 = rng.uniform(0.0, 1.0, N)
    X = np.stack([X1, X2], axis=1)

    def g(a1, a2):
        return np.exp(-8.0 * ((X1 - a1) ** 2 + (X2 - a2) ** 2))

    m = 40.0 * g(0.5, 0.7) * g(0.2, 0.5) + g(0.7, 0.2)
    eps = _draw(N, dist, rng)
    return X, m + eps, m + _q(tau, dist)


GEN = {
    "ex21": (gen_ex21, 2),
    "ex22": (gen_ex22, 1),
    "ex23": (gen_ex23, 2),
}


def standardize_X(X_train: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(0)
    sd = X_train.std(0) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd


def load_household(filepath=None, n_train: int = 1_000_000,
                   n_test: Optional[int] = None, seed: int = 0):
    """Load UCI Household Power data; returns X_train, y_train, X_test, y_test."""
    import pandas as pd

    path = Path(filepath) if filepath else LEGACY_DATA_DIR / "household_power_consumption.txt"
    df = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        usecols=[
            "Global_active_power", "Global_reactive_power", "Voltage",
            "Global_intensity", "Sub_metering_1", "Sub_metering_2",
            "Sub_metering_3",
        ],
        dtype="float32",
        low_memory=False,
    ).dropna().reset_index(drop=True)

    y = df["Global_active_power"].values.astype(float)
    X = df[[c for c in df.columns if c != "Global_active_power"]].values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_train = min(int(n_train), len(y) - 1)
    te_all = perm[n_train:]
    if n_test is not None:
        te_all = te_all[:min(int(n_test), len(te_all))]
    tr, te = perm[:n_train], te_all
    return X[tr], y[tr], X[te], y[te]


def load_airquality(data_dir=None, n_test: int = 20_000, seed: int = 0):
    """Load Beijing Multi-Site Air Quality; returns station shards, X_test, y_test."""
    import glob
    import pandas as pd

    path = Path(data_dir) if data_dir else LEGACY_DATA_DIR / "air"
    files = sorted(glob.glob(str(path / "PRSA_Data_*.csv")))
    if not files:
        raise FileNotFoundError(f"No PRSA_Data_*.csv files found in {path}")

    wd_map = {d: i for i, d in enumerate([
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ])}
    y_col = "PM2.5"
    num_cols = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    x_cols = num_cols + ["wd"]

    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d["wd"] = d["wd"].map(wd_map)
        d = d.dropna(subset=[y_col] + x_cols).reset_index(drop=True)
        dfs.append(d)

    all_df = pd.concat(dfs, ignore_index=True)
    X_all = all_df[x_cols].values.astype(float)
    y_all = all_df[y_col].values.astype(float)
    X_mu, X_sd = X_all.mean(0), X_all.std(0) + 1e-8
    y_mu, y_sd = float(y_all.mean()), float(y_all.std()) + 1e-8

    station_data = []
    for d in dfs:
        Xk = (d[x_cols].values.astype(float) - X_mu) / X_sd
        yk = (d[y_col].values.astype(float) - y_mu) / y_sd
        station_data.append((Xk, yk))

    X_scaled = (X_all - X_mu) / X_sd
    y_scaled = (y_all - y_mu) / y_sd
    rng = np.random.default_rng(seed)
    test_idx = rng.choice(len(y_scaled), min(n_test, len(y_scaled)), replace=False)
    return station_data, X_scaled[test_idx], y_scaled[test_idx]
