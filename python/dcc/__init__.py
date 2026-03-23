"""
python/dcc — DCC/ADCC-GARCH estimation package.

Public API
----------
High-level entry point:
    fit(Z, sigmas, model='DCC', x0=None) -> dict

Post-convergence path functions:
    compute_Q(Z, N_mat, params, Q_bar, N_bar, model) -> ndarray (T, N, N)
    compute_R(Q) -> ndarray (T, N, N)
    loglikelihood(params, Z, sigmas, Q_bar, N_bar, N_mat, model) -> float

Pre-computation utilities:
    estimate_Qbar(Z) -> ndarray (N, N)
    validate_Qbar(Q_bar) -> None
    make_N_matrix(Z) -> ndarray (T, N)
    estimate_Nbar(N_mat) -> ndarray (N, N)
    compute_delta(Q_bar, N_bar) -> float

Example
-------
    import pickle
    import numpy as np
    from python.dcc import fit

    with open('data/dcc_inputs.pkl', 'rb') as f:
        data = pickle.load(f)

    result = fit(data['Z'], data['sigmas'], model='DCC')
    print(result['params'])   # (a, b)
    print(result['llh'])      # full log-likelihood
    print(result['converged'])
"""

from .optimizer import fit
from .model import compute_Q, compute_R, loglikelihood
from .utils import (
    estimate_Qbar,
    validate_Qbar,
    make_N_matrix,
    estimate_Nbar,
    compute_delta,
)

__all__ = [
    # Entry point
    'fit',
    # Path functions
    'compute_Q',
    'compute_R',
    'loglikelihood',
    # Utilities
    'estimate_Qbar',
    'validate_Qbar',
    'make_N_matrix',
    'estimate_Nbar',
    'compute_delta',
]
