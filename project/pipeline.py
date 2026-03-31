"""
pipeline.py — End-to-end GJR-GARCH + DCC/ADCC pipeline.

Connects the univariate volatility layer (python.garch) to the
multivariate correlation layer (python.dcc).

Input:  raw log returns matrix (T, N)
Output: full result dict — params, Q, R, H, llh, Z, sigmas

Usage
-----
    import numpy as np
    from project.pipeline import run_pipeline

    # returns: (T, N) log-return matrix, one column per asset
    result = run_pipeline(returns, model='DCC')
    result = run_pipeline(returns, model='ADCC')

    print(result['dcc']['params'])
    H_t = result['dcc']['H']    # (T, N, N) conditional covariance
"""

import numpy as np
import sys
import os

# Allow imports from project root regardless of working directory
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from python.garch import fit_multivariate_gjr
from python.dcc import fit


def run_pipeline(
    returns: np.ndarray,
    model: str = 'DCC',
) -> dict:
    """
    Full GJR-GARCH → DCC/ADCC pipeline.

    Step 1: Fit GJR-GARCH(1,1,1)-t independently per asset.
            Returns scaled x100 internally; sigmas in % daily.
    Step 2: Construct Z (T,N) and sigmas (T,N).
    Step 3: Fit DCC or ADCC on Z.
    Step 4: Compute full Q_t, R_t, H_t paths post-convergence.

    Parameters
    ----------
    returns : ndarray, shape (T, N)
        Log returns (unscaled). One column per asset.
    model : str
        'DCC' or 'ADCC'.

    Returns
    -------
    dict with keys:
        garch : dict
            Z       : ndarray (T, N) — standardized residuals
            sigmas  : ndarray (T, N) — conditional vols in % daily
            params  : list of N Series — per-asset GJR params
            results : list of N ARCHModelResult — full arch fit objects (use .forecast() for one-step-ahead vol)
        dcc : dict
            params    : tuple — (a, b) or (a, b, g)
            llh       : float — full log-likelihood
            converged : bool
            Q         : ndarray (T, N, N) — pseudo-correlation matrices
            R         : ndarray (T, N, N) — conditional correlations
            H         : ndarray (T, N, N) — conditional covariances
    """
    returns = np.asarray(returns, dtype=float)

    # ------------------------------------------------------------------
    # Step 1+2 — univariate GJR-GARCH per asset
    # ------------------------------------------------------------------
    garch_out = fit_multivariate_gjr(returns)
    Z      = garch_out['Z']        # (T, N)
    sigmas = garch_out['sigmas']   # (T, N), % daily

    # ------------------------------------------------------------------
    # Step 3 — DCC/ADCC estimation
    # ------------------------------------------------------------------
    dcc_result = fit(Z, sigmas, model=model)

    # ------------------------------------------------------------------
    # Step 4 — fit() already computes Q, R, H internally; nothing to add.
    # ------------------------------------------------------------------
    return {
        'garch': {
            'Z':       Z,
            'sigmas':  sigmas,
            'params':  garch_out['params'],
            'results': garch_out['results'],  # list of N ARCHModelResult — for forecasting / DM vol scaling
        },
        'dcc': dcc_result,   # keys: params, Q, R, H, llh, converged, delta
    }
