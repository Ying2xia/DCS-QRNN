"""Chapter 4 script helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def add_sim_filter_args(parser: argparse.ArgumentParser):
    parser.add_argument("--scenario", choices=["ex21", "ex22", "ex23", "all"], default="ex21")
    parser.add_argument("--dist", choices=["N01", "t3", "chi2_2", "all"], default="N01")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--all-taus", action="store_true")
    return parser


def parse_methods(value: str):
    return [v.strip().lower() for v in value.split(",") if v.strip()]

