"""
dgp.py  –  Data-generating processes, Section 4 of paper.pdf
=============================================================
Three scenarios from Wang et al. (2022) Experiment 2 (Examples 2.1–2.3),
each paired with three error distributions.

Example 2.1  (p=2, additive nonlinear)
    Y = sin(X₁) + exp(-X₂²) + ε
    X₁, X₂ ~ N(0,1) independently

Example 2.2  (p=1, heteroscedastic)
    Y = (1 - X + 2X²)·exp(-X²) + σ(X)·ε
    X ~ U(-4, 4),  σ(X) = (1 + 0.2X) / 5

Example 2.3  (p=2, complex nonlinear)
    Y = 40·exp{-8[(X₁-0.5)²+(X₂-0.7)²]}
          × exp{-8[(X₁-0.2)²+(X₂-0.5)²]}
          + exp{-8[(X₁-0.7)²+(X₂-0.2)²]} + ε
    X₁, X₂ ~ U(0, 1) independently

Error distributions
-------------------
  'N01'    : standard normal  N(0,1)
  't3'     : Student-t with 3 degrees of freedom
  'chi2_2' : chi-squared with 2 degrees of freedom

For quantile regression the oracle τ-th conditional quantile is:
    Q_τ(Y|X) = m(X) + σ(X)·Q_τ(ε)
where m(X) is the mean function, σ(X) is the scale function (1 for
Examples 2.1 and 2.3), and Q_τ(ε) is the τ-th quantile of ε.

Each function returns (X, y, Q_true) where:
    X      : (N, p)  covariates
    y      : (N,)    response
    Q_true : (N,)    oracle τ-th conditional quantile
"""

import numpy as np
from scipy.stats import norm as _norm, t as _t, chi2 as _chi2


# ─────────────────────────────────────────────────────────────────────────────
# Error distribution helpers
# ─────────────────────────────────────────────────────────────────────────────

def _q(tau: float, dist: str) -> float:
    """τ-th quantile of the error distribution."""
    if   dist == 'N01':    return float(_norm.ppf(tau))
    elif dist == 't3':     return float(_t.ppf(tau, df=3))
    elif dist == 'chi2_2': return float(_chi2.ppf(tau, df=2))
    raise ValueError(f"Unknown dist '{dist}'. Use 'N01', 't3', or 'chi2_2'.")

def _draw(N: int, dist: str, rng: np.random.Generator) -> np.ndarray:
    """Draw N i.i.d. samples from the error distribution."""
    if   dist == 'N01':    return rng.standard_normal(N)
    elif dist == 't3':     return rng.standard_t(df=3, size=N)
    elif dist == 'chi2_2': return rng.chisquare(df=2, size=N)
    raise ValueError(f"Unknown dist '{dist}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2.1  –  Additive nonlinear model  (p = 2)
# ─────────────────────────────────────────────────────────────────────────────

def gen_ex21(N: int, tau: float, dist: str,
             rng: np.random.Generator):
    """
    Wang et al. (2022) Example 2.1 – additive nonlinear, p=2.

        Y = sin(X₁) + exp(-X₂²) + ε

    X₁, X₂ ~ N(0,1) i.i.d.;  ε ~ dist (independent of X).
    Q_τ(Y|X) = sin(X₁) + exp(-X₂²) + Q_τ(ε)
    """
    # X1  = rng.standard_normal(N)
    # X2  = rng.standard_normal(N)
    # X   = np.stack([X1, X2], axis=1)          # (N, 2)
    # m   = np.sin(X1) + np.exp(-X2**2)
    X1 = rng.uniform(0.0, 1.0, N)
    X2 = rng.uniform(0.0, 1.0, N)
    X = np.stack([X1, X2], axis=1)
    m = np.sin(np.pi * X1) + np.sin(np.pi * X2)
    eps = _draw(N, dist, rng)
    y   = m + eps
    Q   = m + _q(tau, dist)
    return X, y, Q


# ─────────────────────────────────────────────────────────────────────────────
# Example 2.2  –  Heteroscedastic model  (p = 1)
# ─────────────────────────────────────────────────────────────────────────────

def gen_ex22(N: int, tau: float, dist: str,
             rng: np.random.Generator):
    """
    Wang et al. (2022) Example 2.2 – heteroscedastic, p=1.

        Y = (1 - X + 2X²)·exp(-X²) + σ(X)·ε

    X ~ U(-4, 4);  σ(X) = (1 + 0.2X)/5;  ε ~ dist.
    Q_τ(Y|X) = (1-X+2X²)·exp(-X²) + σ(X)·Q_τ(ε)
    """
    x   = rng.uniform(-4.0, 4.0, N)
    sig = (1.0 + 0.2 * x) / 5.0
    m   = (1.0 - x + 2.0 * x**2) * np.exp(-x**2)
    eps = _draw(N, dist, rng)
    y   = m + sig * eps
    Q   = m + sig * _q(tau, dist)
    return x[:, None], y, Q               # X: (N, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Example 2.3  –  Complex nonlinear model  (p = 2)
# ─────────────────────────────────────────────────────────────────────────────

def gen_ex23(N: int, tau: float, dist: str,
             rng: np.random.Generator):
    """
    Wang et al. (2022) Example 2.3 – complex nonlinear, p=2.

        Y = 40·exp{-8[(X₁-0.5)²+(X₂-0.7)²]}
              × exp{-8[(X₁-0.2)²+(X₂-0.5)²]}
              + exp{-8[(X₁-0.7)²+(X₂-0.2)²]} + ε

    X₁, X₂ ~ U(0,1) i.i.d.;  ε ~ dist.
    Q_τ(Y|X) = m(X) + Q_τ(ε)
    """
    X1  = rng.uniform(0.0, 1.0, N)
    X2  = rng.uniform(0.0, 1.0, N)
    X   = np.stack([X1, X2], axis=1)          # (N, 2)

    def _g(a1, a2, b1, b2):
        return np.exp(-8.0 * ((X1 - a1)**2 + (X2 - a2)**2))

    m   = (40.0 * _g(0.5, 0.7, None, None)
               * _g(0.2, 0.5, None, None)
           + _g(0.7, 0.2, None, None))
    eps = _draw(N, dist, rng)
    y   = m + eps
    Q   = m + _q(tau, dist)
    return X, y, Q


# ─────────────────────────────────────────────────────────────────────────────
# Registry  –  name → (generator_fn, p)
# ─────────────────────────────────────────────────────────────────────────────

GEN = {
    'ex21': (gen_ex21, 2),   # Example 2.1: additive nonlinear, p=2
    'ex22': (gen_ex22, 1),   # Example 2.2: heteroscedastic,    p=1
    'ex23': (gen_ex23, 2),   # Example 2.3: complex nonlinear,  p=2
}
