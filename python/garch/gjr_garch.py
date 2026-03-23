"""
gjr_garch.py — GJR-GARCH(1,1,1) univariate volatility module.

Specification (ground truth: notebooks/Data_Exploration_Univariate_Models.ipynb):
  arch_model(returns * 100, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='t')

Returns are scaled x100 before fitting so that conditional volatilities (sigmas)
are in % daily units — matching the units used in dcc_inputs.pkl and the DCC library.

Public API:
  fit_gjr_garch(returns)         -> dict per asset
  fit_multivariate_gjr(returns)  -> dict with Z (T,N) and sigmas (T,N)

No plotting, no file I/O, no notebook artifacts.
"""

import numpy as np
from arch import arch_model


# ---------------------------------------------------------------------------
# Single-asset fit
# ---------------------------------------------------------------------------

def fit_gjr_garch(returns: np.ndarray) -> dict:
    """
    Fit GJR-GARCH(1,1,1) with Student-t innovations to a single return series.

    The series is scaled x100 internally so that the conditional volatility
    (sigma) is in % daily units.  The standardized residuals are unit-free.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        Log returns (NOT scaled).  Scaling is done internally.

    Returns
    -------
    dict with keys:
        sigmas        : ndarray (T,) — conditional volatility in % daily
        residuals     : ndarray (T,) — raw residuals (epsilon_t = r_t - mu_hat) x100
        std_residuals : ndarray (T,) — standardized residuals z_t = epsilon_t / sigma_t
        params        : Series — fitted parameters (omega, alpha[1], gamma[1], beta[1], nu)
        result        : ARCHModelResult — full arch result object
    """
    returns = np.asarray(returns, dtype=float)

    model = arch_model(
        returns * 100,
        mean='Constant',
        vol='GARCH',
        p=1,
        o=1,
        q=1,
        dist='t',
    )
    res = model.fit(disp='off')

    return {
        'sigmas':        np.asarray(res.conditional_volatility),   # % daily
        'residuals':     np.asarray(res.resid),                    # (r-mu)*100
        'std_residuals': np.asarray(res.std_resid),                # z_t
        'params':        res.params,
        'result':        res,
    }


# ---------------------------------------------------------------------------
# Single-asset filter (fixed params — fast daily path)
# ---------------------------------------------------------------------------

def filter_gjr_garch(returns: np.ndarray, params) -> dict:
    """
    Run the GJR-GARCH recursion with FIXED parameters (no optimization).

    Used for the daily live update: params are estimated once (or monthly)
    and stored; each day we re-run the recursion on the updated returns history
    to get today's sigma_t and z_t.  This is ~100x faster than re-fitting.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        Full log returns history (NOT scaled).
    params  : pandas Series or array-like
        Parameter vector from a prior fit_gjr_garch() call.
        Must contain: mu, omega, alpha[1], gamma[1], beta[1], nu
        (i.e., res.params from arch_model.fit()).

    Returns
    -------
    dict with keys:
        sigmas        : ndarray (T,) — conditional volatility in % daily
        std_residuals : ndarray (T,) — standardized residuals z_t
    """
    returns = np.asarray(returns, dtype=float)

    model = arch_model(
        returns * 100,
        mean='Constant',
        vol='GARCH',
        p=1,
        o=1,
        q=1,
        dist='t',
    )
    res = model.fix(params)

    return {
        'sigmas':        np.asarray(res.conditional_volatility),
        'std_residuals': np.asarray(res.std_resid),
    }


# ---------------------------------------------------------------------------
# Multi-asset batch fit
# ---------------------------------------------------------------------------

def fit_multivariate_gjr(returns_matrix: np.ndarray) -> dict:
    """
    Fit GJR-GARCH(1,1,1) independently to each column of a returns matrix.

    Parameters
    ----------
    returns_matrix : ndarray, shape (T, N)
        Matrix of log returns (NOT scaled), one column per asset.

    Returns
    -------
    dict with keys:
        Z      : ndarray (T, N) — standardized residuals, one column per asset
        sigmas : ndarray (T, N) — conditional volatilities in % daily, one col per asset
        params : list of N pandas Series — fitted params per asset
        results: list of N ARCHModelResult objects
    """
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    T, N = returns_matrix.shape

    Z      = np.empty((T, N))
    sigmas = np.empty((T, N))
    params  = []
    results = []

    for i in range(N):
        fit = fit_gjr_garch(returns_matrix[:, i])
        Z[:, i]      = fit['std_residuals']
        sigmas[:, i] = fit['sigmas']
        params.append(fit['params'])
        results.append(fit['result'])

    return {
        'Z':       Z,
        'sigmas':  sigmas,
        'params':  params,
        'results': results,
    }
