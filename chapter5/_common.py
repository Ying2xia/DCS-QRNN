"""Chapter 5 script helpers."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import TAUS
from common.data import load_airquality, load_household


DATASETS = ("household", "airquality")


def add_real_filter_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", choices=["household", "airquality", "both"], default="both")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--all-taus", action="store_true")
    return parser


def add_real_data_args(parser: argparse.ArgumentParser):
    parser.add_argument("--household-file", default=str(HERE / "household_power_consumption.txt"))
    parser.add_argument("--air-dir", default=str(HERE / "air"))
    parser.add_argument("--household-n-train", type=int, default=1_000_000)
    parser.add_argument("--household-n-test", type=int, default=50_000)
    parser.add_argument("--air-n-test", type=int, default=20_000)
    parser.add_argument(
        "--air-max-per-station",
        type=int,
        default=None,
        help="Optional cap per station. Useful for quick test runs.",
    )
    return parser


def selected_datasets(value: str):
    return list(DATASETS) if value == "both" else [value]


def selected_taus(args):
    return list(TAUS) if args.all_taus else [args.tau]


def parse_methods(value: str):
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def split_household_workers(X, y, K: int):
    order = np.argsort(X[:, 0])
    return [(X[idx], y[idx]) for idx in np.array_split(order, int(K))]


def _cap_station_data(station_data, max_per_station, seed: int):
    if max_per_station is None:
        return station_data
    rng = np.random.default_rng(seed)
    out = []
    for Xk, yk in station_data:
        n = min(int(max_per_station), len(yk))
        idx = rng.choice(len(yk), n, replace=False)
        out.append((Xk[idx], yk[idx]))
    return out


def load_dataset(args, dataset: str, seed: int = 0):
    if dataset == "household":
        X_tr, y_tr, X_te, y_te = load_household(
            args.household_file,
            n_train=args.household_n_train,
            n_test=args.household_n_test,
            seed=seed,
        )
        workers = split_household_workers(X_tr, y_tr, getattr(args, "K", 10))
        return X_tr, y_tr, X_te, y_te, workers

    station_data, X_te, y_te = load_airquality(
        args.air_dir,
        n_test=args.air_n_test,
        seed=seed,
    )
    station_data = _cap_station_data(station_data, args.air_max_per_station, seed + 17)
    X_tr = np.vstack([Xk for Xk, _ in station_data])
    y_tr = np.concatenate([yk for _, yk in station_data])
    return X_tr, y_tr, X_te, y_te, station_data


def load_h_map(path):
    h_map = {}
    if not path:
        return h_map
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            if "h" not in row or row["h"] in ("", None):
                continue
            tau = float(row["tau"])
            dataset = row.get("dataset")
            if dataset:
                h_map[(dataset, tau)] = float(row["h"])
            h_map[(tau,)] = float(row["h"])
    return h_map


def resolve_h(h_map, fallback_h: float, tau: float, dataset: str):
    for key in ((dataset, float(tau)), (float(tau),)):
        if key in h_map:
            return float(h_map[key])
    return float(fallback_h)
