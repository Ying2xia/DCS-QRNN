"""Metrics and summary helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(e ** 2)))


def summarise(rows: list[dict], group_keys: Iterable[str],
              value_keys: Iterable[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    group_keys = list(group_keys)
    value_keys = list(value_keys)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)

    out = []
    for key, group in sorted(groups.items()):
        item = {k: v for k, v in zip(group_keys, key)}
        item["n"] = len(group)
        for metric in value_keys:
            vals = np.array([float(r[metric]) for r in group if r.get(metric) not in ("", None)], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            item[f"{metric}_std"] = float(np.std(vals)) if len(vals) else np.nan
        out.append(item)
    return out

