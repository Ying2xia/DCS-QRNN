"""Configuration shared by Chapter 4 and Chapter 5 experiment scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = PROJECT_ROOT.parent
LEGACY_DATA_DIR = REVIEW_ROOT / "CSQRNN_code"

SIM_SCENARIOS = ("ex21", "ex22", "ex23")
SIM_DISTS = ("N01", "t3", "chi2_2")
TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)

J_GRID = tuple(range(1, 11))
LAMBDA_GRID = tuple(round(0.01 * i, 2) for i in range(1, 11))
PILOT_RATIOS = (0.01, 0.05, 0.10)
BANDWIDTH_C_GRID = (0.01, 0.05, 0.10, 0.50, 1.00)

DEFAULT_J = 10
DEFAULT_LAMBDA = 0.01
DEFAULT_H = 0.1
DEFAULT_T = 1
DEFAULT_K = 10
DEFAULT_SIM_N = 200_000
DEFAULT_SIM_N_TEST = 2_000
DEFAULT_REPS = 10

