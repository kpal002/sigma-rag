"""
sigma_rag/stats.py
------------------
Pure-numpy implementations of the scipy.stats functions used by σ-RAG.

This makes the core package dependency-free beyond numpy, so it runs
in any environment without needing scipy installed.

When scipy IS available it is used automatically (faster, more precise).
"""

from __future__ import annotations

import math

import numpy as np


# ── Normal CDF (Φ) ─────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """
    Standard normal CDF Φ(x) = P(Z ≤ x) for Z ~ N(0,1).

    Uses the complementary error function identity:
        Φ(x) = erfc(-x / √2) / 2

    Accurate to ~15 significant figures across the full real line.

    Args:
        x: Quantile value.

    Returns:
        Probability in [0, 1].
    """
    return float(math.erfc(-x / math.sqrt(2)) / 2.0)


def norm_sf(x: float) -> float:
    """
    Standard normal survival function: 1 - Φ(x) = P(Z > x).

    Args:
        x: Quantile value.

    Returns:
        Probability in [0, 1].
    """
    return float(math.erfc(x / math.sqrt(2)) / 2.0)


# ── KS test (one-sample, vs fitted normal) ─────────────────────────────────

def kstest_norm(samples: np.ndarray, mu: float, sigma: float) -> tuple[float, float]:
    """
    One-sample Kolmogorov-Smirnov test against N(mu, sigma^2).

    Compares the empirical CDF of samples to the theoretical normal CDF.

    Args:
        samples: 1-D array of observed values.
        mu:      Hypothesised normal mean.
        sigma:   Hypothesised normal std.

    Returns:
        Tuple of (ks_statistic, p_value).
        - ks_statistic: max |F_n(x) - F(x)|  in [0, 1].
        - p_value:      Approximate p-value via the Kolmogorov distribution.
                        Small p → reject normality.
    """
    n = len(samples)
    if n == 0:
        return 0.0, 1.0

    # Standardise
    z = np.sort((samples - mu) / sigma)

    # Theoretical CDF at each sorted point
    theoretical = np.array([norm_cdf(float(zi)) for zi in z])

    # Empirical CDF
    empirical_above = np.arange(1, n + 1) / n   # F_n(x⁺)
    empirical_below = np.arange(0, n) / n        # F_n(x⁻)

    # KS statistic: max deviation
    ks_stat = float(
        max(
            np.max(np.abs(empirical_above - theoretical)),
            np.max(np.abs(empirical_below - theoretical)),
        )
    )

    # Approximate p-value via the Kolmogorov distribution
    # P(K ≤ ks_stat) where K is the Kolmogorov distribution
    # Using the approximation: p ≈ 2 * Σ (-1)^(k+1) exp(-2k²t²)  for t = ks*√n
    t = ks_stat * math.sqrt(n)
    p_value = _kolmogorov_p(t)

    return ks_stat, p_value


def _kolmogorov_p(t: float) -> float:
    """
    Complementary Kolmogorov CDF: P(K > t).

    Approximation via the alternating series (converges rapidly for t > 0.2):
        Q(t) = 2 Σ_{k=1}^{∞} (-1)^{k+1} exp(-2k²t²)

    Args:
        t: Scaled KS statistic t = D_n * √n.

    Returns:
        Approximate p-value in [0, 1].
    """
    if t <= 0:
        return 1.0
    if t > 3.5:
        return 0.0  # negligible for large t

    total = 0.0
    for k in range(1, 50):
        term = ((-1) ** (k + 1)) * math.exp(-2 * k * k * t * t)
        total += term
        if abs(term) < 1e-10:
            break

    return float(min(max(2.0 * total, 0.0), 1.0))


# ── Unified interface (uses scipy when available) ───────────────────────────

def _try_scipy():
    """Return scipy.stats if available, else None."""
    try:
        from scipy import stats
        return stats
    except ImportError:
        return None


_scipy_stats = _try_scipy()


def cdf(x: float) -> float:
    """
    Standard normal CDF — uses scipy when available, falls back to pure numpy.

    Args:
        x: Quantile.

    Returns:
        Φ(x) in [0, 1].
    """
    if _scipy_stats is not None:
        return float(_scipy_stats.norm.cdf(x))
    return norm_cdf(x)


def sf(x: float) -> float:
    """
    Standard normal survival function — uses scipy when available.

    Args:
        x: Quantile.

    Returns:
        1 - Φ(x) in [0, 1].
    """
    if _scipy_stats is not None:
        return float(_scipy_stats.norm.sf(x))
    return norm_sf(x)


def ks_test(
    samples: np.ndarray, mu: float, sigma: float
) -> tuple[float, float]:
    """
    One-sample KS test vs N(mu, sigma^2) — uses scipy when available.

    Args:
        samples: Observed sample array.
        mu:      Hypothesised mean.
        sigma:   Hypothesised std.

    Returns:
        (ks_statistic, p_value).
    """
    if _scipy_stats is not None:
        result = _scipy_stats.kstest(samples, "norm", args=(mu, sigma))
        return float(result.statistic), float(result.pvalue)
    return kstest_norm(samples, mu, sigma)
