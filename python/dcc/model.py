"""
model.py — DCC-GARCH recursion, objective, and log-likelihood.

Single implementation of _update_Q shared by the optimizer objective
and the post-convergence path-storage functions. No duplication.

References:
  Engle (2002) — DCC
  Cappiello, Engle & Sheppard (2006) — ADCC
  MEMORY.md §8.2, §9.3
"""

import numpy as np
import scipy.linalg

# ---------------------------------------------------------------------------
# Module-level constants (MEMORY.md §9.3)
# ---------------------------------------------------------------------------
PENALTY = 1e6   # Stateless; returned on any numerical failure (DEV-07, Fix 6)
EPS     = 1e-6  # Constraint margin


# ---------------------------------------------------------------------------
# Private: single-step Q_t update
# ---------------------------------------------------------------------------

def _update_Q(
    Q_prev:  np.ndarray,
    z_prev:  np.ndarray,
    n_prev,           # ndarray (N,) for ADCC; None for DCC
    AUQ:     np.ndarray,
    a: float,
    b: float,
    g: float,
    model: str,
) -> np.ndarray:
    """
    Single-step DCC/ADCC Q_t recursion.

    DCC  (Engle 2002):
        Q_t = (1-a-b)·Q̄ + a·z_{t-1}z_{t-1}' + b·Q_{t-1}

    ADCC (Cappiello et al. 2006, full Cappiello form — DEV-11):
        Q_t = [(1-a-b)·Q̄ - g·N̄] + a·z_{t-1}z_{t-1}' + g·n_{t-1}n_{t-1}' + b·Q_{t-1}

    AUQ must be pre-computed ONCE per objective call, outside the t-loop (Fix 9):
        DCC:  AUQ = (1-a-b) * Q_bar
        ADCC: AUQ = (1-a-b) * Q_bar - g * N_bar

    Result is symmetrized to suppress floating-point drift (DEV-04).

    Parameters
    ----------
    Q_prev : ndarray, shape (N, N) — Q_{t-1}
    z_prev : ndarray, shape (N,)   — z_{t-1}
    n_prev : ndarray, shape (N,) or None — n_{t-1}; None for DCC
    AUQ    : ndarray, shape (N, N) — pre-computed intercept
    a, b, g : float — DCC/ADCC parameters
    model  : str — 'DCC' or 'ADCC'

    Returns
    -------
    Q_next : ndarray, shape (N, N)
    """
    Q_next = AUQ + a * np.outer(z_prev, z_prev) + b * Q_prev
    if model == 'ADCC':
        Q_next = Q_next + g * np.outer(n_prev, n_prev)
    return (Q_next + Q_next.T) / 2   # DEV-04: symmetrize in-place


# ---------------------------------------------------------------------------
# Public: post-convergence path storage
# ---------------------------------------------------------------------------

def compute_Q(
    Z:      np.ndarray,
    N_mat,            # ndarray (T, N) for ADCC; None for DCC (Fix 4)
    params: tuple,
    Q_bar:  np.ndarray,
    N_bar,            # ndarray (N, N) for ADCC; None for DCC (Fix 4)
    model:  str,
) -> np.ndarray:
    """
    Compute the full Q_t path after optimization.

    Called once post-convergence; not called during optimization.
    Uses _update_Q — the single shared recursion (Fix 7).

    Parameters
    ----------
    Z      : ndarray, shape (T, N) — standardized residuals
    N_mat  : ndarray, shape (T, N) or None — asymmetric innovations
    params : tuple — (a, b) for DCC; (a, b, g) for ADCC
    Q_bar  : ndarray, shape (N, N)
    N_bar  : ndarray, shape (N, N) or None
    model  : str — 'DCC' or 'ADCC'

    Returns
    -------
    Q : ndarray, shape (T, N, N)
        Q[0] = Q_bar; Q[t] for t = 1..T-1 via _update_Q.
    """
    T, N = Z.shape
    a, b = params[0], params[1]
    g = params[2] if model == 'ADCC' else 0.0

    # Pre-compute intercept once (Fix 9)
    if model == 'DCC':
        AUQ = (1 - a - b) * Q_bar
    else:
        AUQ = (1 - a - b) * Q_bar - g * N_bar

    Q = np.empty((T, N, N))
    Q[0] = Q_bar.copy()   # Q[0] = Q̄ (DEV-02/03)

    for t in range(1, T):
        n_prev = N_mat[t - 1] if model == 'ADCC' else None
        Q[t] = _update_Q(Q[t - 1], Z[t - 1], n_prev, AUQ, a, b, g, model)

    return Q


def compute_R(Q: np.ndarray) -> np.ndarray:
    """
    Normalize Q_t path to conditional correlation matrices R_t.

    R_t[i,j] = Q_t[i,j] / sqrt(Q_t[i,i] · Q_t[j,j])
    Equivalent to D_t^{-1} Q_t D_t^{-1} without matrix inversion (MEMORY.md §7.6).

    R_t inherits symmetry from the symmetrized Q_t (DEV-04).
    Called once post-convergence.

    Parameters
    ----------
    Q : ndarray, shape (T, N, N)

    Returns
    -------
    R : ndarray, shape (T, N, N)

    Raises
    ------
    ValueError
        If any diagonal element of Q_t is non-positive (numerical breakdown).
    """
    T, N, _ = Q.shape
    R = np.empty_like(Q)

    for t in range(T):
        diag_q = np.diag(Q[t])
        if np.any(diag_q <= 0):
            raise ValueError(
                f"Q[{t}] has non-positive diagonal element(s): {diag_q}. "
                "Parameters may be invalid or optimization failed."
            )
        sqrt_diag = np.sqrt(diag_q)
        R[t] = Q[t] / np.outer(sqrt_diag, sqrt_diag)

    return R


# ---------------------------------------------------------------------------
# Optimizer objective
# ---------------------------------------------------------------------------

def dcc_objective(
    params: np.ndarray,
    Z:      np.ndarray,
    Q_bar:  np.ndarray,
    N_bar,            # ndarray (N, N) for ADCC; None for DCC
    N_mat,            # ndarray (T, N) for ADCC; None for DCC (Fix 4)
    model:  str,
) -> float:
    """
    DCC/ADCC optimizer objective (Stage 1).

    Minimizes: +½ Σ_t [ log|R_t| + z_t' R_t^{-1} z_t ]

    log|R_t| via Cholesky: 2·Σ_i log(L_{ii})  (DEV-06)
    z_t' R_t^{-1} z_t via forward substitution: ||L^{-1} z_t||²  (DEV-05)
    z_t'z_t dropped: constant w.r.t. (a, b, g)  (DEV-12)

    Returns PENALTY on any numerical failure; does NOT modify Q_t (DEV-07, Fix 6).
    AUQ computed once outside t-loop (Fix 9).

    Parameters
    ----------
    params : ndarray — [a, b] or [a, b, g]
    Z      : ndarray, shape (T, N) — fixed inputs
    Q_bar  : ndarray, shape (N, N)
    N_bar  : ndarray, shape (N, N) or None
    N_mat  : ndarray, shape (T, N) or None
    model  : str — 'DCC' or 'ADCC'

    Returns
    -------
    float — objective value (positive) or PENALTY
    """
    T = Z.shape[0]
    a, b = params[0], params[1]
    g = params[2] if model == 'ADCC' else 0.0

    # Pre-compute intercept once per call (Fix 9)
    if model == 'DCC':
        AUQ = (1 - a - b) * Q_bar
    else:
        AUQ = (1 - a - b) * Q_bar - g * N_bar

    obj  = 0.0
    Q_t  = Q_bar.copy()   # Q[0] = Q̄ (DEV-02/03)

    for t in range(1, T):
        n_prev = N_mat[t - 1] if model == 'ADCC' else None
        Q_t = _update_Q(Q_t, Z[t - 1], n_prev, AUQ, a, b, g, model)  # Fix 7

        # Diagonal guard — before sqrt (Fix 8)
        if np.any(np.diag(Q_t) <= 0):
            return PENALTY

        sqrt_diag = np.sqrt(np.diag(Q_t))
        R_t = Q_t / np.outer(sqrt_diag, sqrt_diag)

        # Cholesky-based log-det and quadratic form (DEV-05, DEV-06, DEV-07)
        try:
            L = np.linalg.cholesky(R_t)
        except np.linalg.LinAlgError:
            return PENALTY

        logdet = 2.0 * np.sum(np.log(np.diag(L)))              # log|R_t|
        y      = scipy.linalg.solve_triangular(L, Z[t], lower=True)
        quad   = y @ y                                          # z_t' R_t^{-1} z_t

        obj += logdet + quad   # DEV-12: constants dropped

    return 0.5 * obj   # positive; minimized by optimizer


# ---------------------------------------------------------------------------
# Full log-likelihood (post-optimization, for reporting)
# ---------------------------------------------------------------------------

def loglikelihood(
    params:  tuple,
    Z:       np.ndarray,
    sigmas:  np.ndarray,
    Q_bar:   np.ndarray,
    N_bar,            # ndarray (N, N) or None
    N_mat,            # ndarray (T, N) or None (Fix 4)
    model:   str,
) -> float:
    """
    Full Gaussian DCC log-likelihood (Stage 2).

    L = -½ Σ_t [ N·log(2π) + log|Σ_t²| + log|R_t| + z_t' R_t^{-1} z_t ]
      = -½ Σ_t [ N·log(2π) + 2·Σ_i log(σ_{i,t}) + logdet_chol(R_t) + quad ]

    Uses _update_Q for recursion (Fix 7 — single implementation).
    AUQ computed once before loop (Fix 9).

    Called once after optimization; not used as optimizer objective.

    Parameters
    ----------
    params : tuple — (a, b) or (a, b, g)
    Z      : ndarray, shape (T, N)
    sigmas : ndarray, shape (T, N) — GARCH conditional volatilities (σ, not σ²)
    Q_bar  : ndarray, shape (N, N)
    N_bar  : ndarray, shape (N, N) or None
    N_mat  : ndarray, shape (T, N) or None
    model  : str — 'DCC' or 'ADCC'

    Returns
    -------
    float — full log-likelihood scalar
    """
    T, N = Z.shape
    a, b = params[0], params[1]
    g = params[2] if model == 'ADCC' else 0.0

    log2pi = np.log(2 * np.pi)

    # Pre-compute intercept once (Fix 9)
    if model == 'DCC':
        AUQ = (1 - a - b) * Q_bar
    else:
        AUQ = (1 - a - b) * Q_bar - g * N_bar

    llh = 0.0
    Q_t = Q_bar.copy()

    for t in range(1, T):
        n_prev = N_mat[t - 1] if model == 'ADCC' else None
        Q_t = _update_Q(Q_t, Z[t - 1], n_prev, AUQ, a, b, g, model)

        sqrt_diag = np.sqrt(np.diag(Q_t))
        R_t = Q_t / np.outer(sqrt_diag, sqrt_diag)

        L       = np.linalg.cholesky(R_t)
        logdet  = 2.0 * np.sum(np.log(np.diag(L)))
        y       = scipy.linalg.solve_triangular(L, Z[t], lower=True)
        quad    = y @ y

        log_sig = 2.0 * np.sum(np.log(sigmas[t]))   # log|Σ_t²| = 2·Σ log σ_{i,t}

        llh += N * log2pi + log_sig + logdet + quad

    return -0.5 * llh
