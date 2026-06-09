"""Small CSV/JSON storage helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Optional


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(path, rows: list[dict], fieldnames: Optional[Iterable[str]] = None) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    fieldnames = list(fieldnames)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_json(path, payload: dict) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_csv(path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))
