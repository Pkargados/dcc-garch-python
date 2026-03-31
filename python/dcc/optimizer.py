"""
optimizer.py — DCC/ADCC constraint definitions and fit() entry point.

Orchestrates pre-computation (utils.py) and optimization (model.py).
Uses SLSQP with hard inequality constraints (DEV-08).

References:
  Engle (2002) — DCC
  Cappiello, Engle & Sheppard (2006) — ADCC
  MEMORY.md §8.2, §9.4
"""

import numpy as np
import scipy.optimize

from .utils import (
    estimate_Qbar,
    validate_Qbar,
    make_N_matrix,
    estimate_Nbar,
    compute_delta,
)
from .model import (
    dcc_objective,
    compute_Q,
    compute_R,
    loglikelihood,
    EPS,
)


# ---------------------------------------------------------------------------
# Constraint functions for SLSQP
# ---------------------------------------------------------------------------

def dcc_constraint(params: np.ndarray, eps: float) -> float:
    """
    DCC stationarity constraint: a + b < 1.

    SLSQP requires constraints of the form g(x) >= 0.
    Returns (1 - eps) - a - b >= 0.

    Parameters
    ----------
    params : ndarray — [a, b]
    eps    : float  — constraint margin (EPS = 1e-6)

    Returns
    -------
    float — value that must be >= 0 for feasibility
    """
    return (1.0 - eps) - params[0] - params[1]


def adcc_constraint(params: np.ndarray, delta: float, eps: float) -> float:
    """
    ADCC stationarity constraint: a + b + δ·g < 1.

    Matches rmgarch .adcccon exactly (DEV-09).
    δ = max eigenvalue of Q̄^{-1/2} N̄ Q̄^{-1/2} — computed from data.

    SLSQP requires constraints of the form g(x) >= 0.
    Returns (1 - eps) - a - b - δ·g >= 0.

    Parameters
    ----------
    params : ndarray — [a, b, g]
    delta  : float  — max generalized eigenvalue of (N_bar, Q_bar)
    eps    : float  — constraint margin (EPS = 1e-6)

    Returns
    -------
    float — value that must be >= 0 for feasibility
    """
    return (1.0 - eps) - params[0] - params[1] - delta * params[2]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def fit(
    Z:       np.ndarray,
    sigmas:  np.ndarray,
    model:   str = 'DCC',
    x0:      np.ndarray = None,
) -> dict:
    """
    Estimate DCC or ADCC parameters via two-stage QML.

    Stage 1 GARCH is pre-completed (inputs: Z, sigmas).
    This function handles Stage 2 DCC only.

    Pre-computation (utils.py) → Optimization (SLSQP) → Post-convergence outputs.

    Parameters
    ----------
    Z      : ndarray, shape (T, N) — standardized residuals z_{i,t} = ε_{i,t}/σ_{i,t}
    sigmas : ndarray, shape (T, N) — GARCH conditional volatilities (σ, not σ²)
    model  : str — 'DCC' (default) or 'ADCC'
    x0     : ndarray or None — initial parameter vector; auto-set if None

    Returns
    -------
    dict with keys:
        params    — ndarray: estimated (a, b) for DCC; (a, b, g) for ADCC
        Q         — ndarray, shape (T, N, N): pseudo-correlation path
        R         — ndarray, shape (T, N, N): conditional correlation path
        H         — ndarray, shape (T, N, N): conditional covariance path
        sigmas    — ndarray, shape (T, N): GARCH conditional vols (pass-through of input)
        llh       — float: full Gaussian log-likelihood
        converged — bool: optimizer success flag
        delta     — float or None: ADCC δ scalar; None for DCC

    Raises
    ------
    ValueError
        If Z or sigmas contain NaN/Inf, or if Q_bar is not positive definite.
    """
    if model not in ('DCC', 'ADCC'):
        raise ValueError(f"model must be 'DCC' or 'ADCC', got '{model}'")

    # -----------------------------------------------------------------------
    # Step 1: Input guards
    # -----------------------------------------------------------------------
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z contains NaN or Inf values.")
    if not np.all(np.isfinite(sigmas)):
        raise ValueError("sigmas contains NaN or Inf values.")

    # -----------------------------------------------------------------------
    # Step 2: Pre-computation (utils.py)
    # -----------------------------------------------------------------------
    Q_bar = estimate_Qbar(Z)        # DEV-01: np.cov ddof=1
    validate_Qbar(Q_bar)            # Fix 5: PD check; raises if not PD

    N_mat  = make_N_matrix(Z)       # always computed; needed for ADCC
    N_bar  = None
    delta  = None

    if model == 'ADCC':
        N_bar = estimate_Nbar(N_mat)    # DEV-01b, Fix 1: uncentered estimator
        delta = compute_delta(Q_bar, N_bar)  # DEV-09: generalized eigenvalue

    # -----------------------------------------------------------------------
    # Step 3: Bounds
    # -----------------------------------------------------------------------
    n_params = 2 if model == 'DCC' else 3
    bounds   = [(1e-10, 1 - EPS)] * n_params

    # -----------------------------------------------------------------------
    # Step 4: Constraints (DEV-08: hard constraints via SLSQP)
    # -----------------------------------------------------------------------
    if model == 'DCC':
        constraints = [
            {'type': 'ineq', 'fun': dcc_constraint, 'args': (EPS,)}   # Fix 3
        ]
    else:
        constraints = [
            {'type': 'ineq', 'fun': adcc_constraint, 'args': (delta, EPS)}  # Fix 3
        ]

    # -----------------------------------------------------------------------
    # Step 5: Starting values (Fix 2: feasibility-aware for ADCC)
    # -----------------------------------------------------------------------
    if x0 is None:
        if model == 'DCC':
            x0 = np.array([0.05, 0.90])
        else:
            a0, b0 = 0.05, 0.90
            # Guarantee feasibility: a0 + b0 + δ·g0 < 1
            g0 = min(0.01, 0.5 * (1.0 - a0 - b0) / delta)
            x0 = np.array([a0, b0, g0])

    # -----------------------------------------------------------------------
    # Step 6: Optimization (SLSQP) — Fix 3: args= fully wired
    # -----------------------------------------------------------------------
    if model == 'DCC':
        obj_args = (Z, Q_bar, None, None, 'DCC')
    else:
        obj_args = (Z, Q_bar, N_bar, N_mat, 'ADCC')

    result = scipy.optimize.minimize(
        fun=dcc_objective,
        x0=x0,
        args=obj_args,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
    )

    # -----------------------------------------------------------------------
    # Step 7: Post-convergence outputs (Fix 4: N_mat explicit throughout)
    # -----------------------------------------------------------------------
    params_est = result.x

    Q_path = compute_Q(Z, N_mat if model == 'ADCC' else None,
                       params_est, Q_bar, N_bar, model)
    R_path = compute_R(Q_path)

    # H_t = Σ_t · R_t · Σ_t  (element-wise: H[t,i,j] = σ[t,i] · R[t,i,j] · σ[t,j])
    H_path = sigmas[:, :, np.newaxis] * R_path * sigmas[:, np.newaxis, :]

    llh = loglikelihood(
        params_est, Z, sigmas, Q_bar,
        N_bar, N_mat if model == 'ADCC' else None, model
    )

    return {
        'params':    params_est,
        'Q':         Q_path,
        'R':         R_path,
        'H':         H_path,
        'sigmas':    sigmas,    # (T, N) GARCH conditional vols passed in — convenient for vol-scaled strategies
        'llh':       llh,
        'converged': result.success,
        'delta':     delta,
    }
