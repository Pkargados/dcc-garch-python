"""
utils.py — DCC-GARCH pre-computation utilities.

Pure functions. No optimizer state. Called once before optimization.

References:
  Engle (2002) — DCC
  Cappiello, Engle & Sheppard (2006) — ADCC
  MEMORY.md §9.2
"""

import numpy as np
import scipy.linalg


def estimate_Qbar(Z: np.ndarray) -> np.ndarray:
    """
    Estimate Q̄ = E[z_t z_t'] as the sample covariance of Z.

    Uses ddof=1 (DEV-01: matches R cov() denominator 1/(T-1)).
    Result is symmetrized to suppress floating-point asymmetry.

    Parameters
    ----------
    Z : ndarray, shape (T, N)
        Standardized residuals.

    Returns
    -------
    Q_bar : ndarray, shape (N, N)
    """
    Q_bar = np.cov(Z.T, ddof=1)
    Q_bar = (Q_bar + Q_bar.T) / 2
    return Q_bar


def validate_Qbar(Q_bar: np.ndarray) -> None:
    """
    Verify Q̄ is positive definite.

    No regularization is applied. Raises ValueError if not PD.
    Must be called before compute_delta.

    Parameters
    ----------
    Q_bar : ndarray, shape (N, N)

    Raises
    ------
    ValueError
        If the minimum eigenvalue of Q_bar is non-positive.
    """
    eigvals = np.linalg.eigvalsh(Q_bar)
    if eigvals.min() <= 0:
        raise ValueError(
            f"Q_bar is not positive definite. "
            f"Minimum eigenvalue: {eigvals.min():.6e}"
        )


def make_N_matrix(Z: np.ndarray) -> np.ndarray:
    """
    Compute asymmetric innovations n_t = z_t · 𝟏[z_t < 0].

    DEV-10: replicates rmgarch .asymI — indicator is 1 if z < 0, 0 if z >= 0.
    x=0 maps to 0 (not 0.5).

    Parameters
    ----------
    Z : ndarray, shape (T, N)
        Standardized residuals.

    Returns
    -------
    N_mat : ndarray, shape (T, N)
        Asymmetric innovations.
    """
    return Z * (Z < 0).astype(float)


def estimate_Nbar(N_mat: np.ndarray) -> np.ndarray:
    """
    Estimate N̄ = E[n_t n_t'] using the uncentered estimator.

    Formula: (N_mat.T @ N_mat) / (T - 1)

    Deliberate deviation from rmgarch (DEV-01b):
      rmgarch uses cov() which centers n_t first.
      The uncentered estimator estimates E[n_t n_t'] exactly,
      which preserves E[Q_t] = Q̄ unconditionally in the ADCC recursion.

    Parameters
    ----------
    N_mat : ndarray, shape (T, N)
        Asymmetric innovations from make_N_matrix.

    Returns
    -------
    N_bar : ndarray, shape (N, N)
    """
    T = N_mat.shape[0]
    N_bar = (N_mat.T @ N_mat) / (T - 1)
    N_bar = (N_bar + N_bar.T) / 2
    return N_bar


def compute_delta(Q_bar: np.ndarray, N_bar: np.ndarray) -> float:
    """
    Compute δ = max eigenvalue of Q̄^{-1/2} N̄ Q̄^{-1/2}.

    Used in the ADCC stationarity constraint: a + b + δ·g < 1.
    Matches rmgarch .adcccon computation exactly (DEV-09).

    Solved as a generalized eigenvalue problem: N̄ v = λ Q̄ v,
    which is equivalent to the standard eigenvalue problem on Q̄^{-1/2} N̄ Q̄^{-1/2}.

    Requires Q_bar to be PD (call validate_Qbar first).

    Parameters
    ----------
    Q_bar : ndarray, shape (N, N) — must be positive definite
    N_bar : ndarray, shape (N, N)

    Returns
    -------
    delta : float — max generalized eigenvalue
    """
    eigvals = scipy.linalg.eigh(N_bar, Q_bar, eigvals_only=True)
    return float(eigvals.max())
