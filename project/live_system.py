"""
live_system.py — Production DCC-GARCH daily filter + periodic re-estimation.

Architecture (from outputs/extension_pipeline.md and outputs/interpretation.md):

  Layer 1 — Daily filter (microseconds):
    Update Q_t via _update_Q, normalize to R_t and H_t.
    No optimization. State is one matrix Q_t.

  Layer 2 — Periodic re-estimation (rolling window):
    Re-fit GJR-GARCH per asset + DCC/ADCC on rolling N-day window.
    Replace stored parameters and reset filter state.

Class: DCCSystem

Usage
-----
    import numpy as np
    from project.live_system import DCCSystem

    # Initialize with estimated parameters (from run_pipeline or fit())
    sys = DCCSystem(
        params    = result['dcc']['params'],   # (a,b) or (a,b,g)
        Q_bar     = result['dcc']['Q_bar'],    # unconditional target
        N_bar     = result['dcc']['N_bar'],    # ADCC only, else None
        delta     = result['dcc']['delta'],    # ADCC only, else None
        Q_state   = result['dcc']['Q'][-1],   # last Q_t from fit
        garch_params = result['garch']['params'],  # list of N Series
        model     = 'DCC',
    )

    # Each day:
    H_t = sys.update(z_t, sigma_t)   # z_t, sigma_t: (N,) arrays

    # Monthly re-estimation on rolling window:
    sys.refit(returns_window, model='DCC')
"""

import numpy as np
import sys
import os

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from python.dcc.model import _update_Q
from python.dcc.utils import estimate_Qbar, make_N_matrix, estimate_Nbar, compute_delta
from python.garch import fit_gjr_garch, fit_multivariate_gjr
from python.dcc import fit


class DCCSystem:
    """
    Stateful DCC/ADCC filter for daily production use.

    State: Q_t — the current pseudo-correlation matrix.
    Parameters: (a, b) or (a, b, g) — estimated offline, updated periodically.

    Two operations:
      update()  — one-step filter advance (daily, microseconds)
      refit()   — full re-estimation on rolling window (monthly, ~30s)
    """

    def __init__(
        self,
        params:       np.ndarray,
        Q_bar:        np.ndarray,
        Q_state:      np.ndarray,
        model:        str = 'DCC',
        N_bar:        np.ndarray = None,
        delta:        float = None,
        garch_params: list = None,
    ):
        """
        Parameters
        ----------
        params       : ndarray — (a, b) for DCC; (a, b, g) for ADCC
        Q_bar        : ndarray (N, N) — unconditional pseudo-correlation target
        Q_state      : ndarray (N, N) — current Q_{t-1} (last known state)
        model        : str — 'DCC' or 'ADCC'
        N_bar        : ndarray (N, N) or None — ADCC only
        delta        : float or None — ADCC stationarity scalar
        garch_params : list of N pandas Series or None — stored GJR params per asset
        """
        if model not in ('DCC', 'ADCC'):
            raise ValueError(f"model must be 'DCC' or 'ADCC', got '{model}'")

        self.model        = model
        self.params       = np.asarray(params, dtype=float)
        self.Q_bar        = np.asarray(Q_bar, dtype=float)
        self.Q_state      = np.asarray(Q_state, dtype=float)
        self.N_bar        = N_bar
        self.delta        = delta
        self.garch_params = garch_params   # list[N] of Series

        # Pre-compute AUQ intercept — constant until next refit()
        self._refresh_AUQ()

    # ------------------------------------------------------------------
    # Layer 1: daily filter update
    # ------------------------------------------------------------------

    def update(
        self,
        z_t:     np.ndarray,
        sigma_t: np.ndarray,
    ) -> dict:
        """
        Advance filter by one day.

        Computes:
          n_t = z_t * (z_t < 0)            (ADCC only)
          Q_t = AUQ + a*outer(z,z) + [g*outer(n,n)] + b*Q_{t-1}
          R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
          H_t = sigma_t[:,None] * R_t * sigma_t[None,:]

        Parameters
        ----------
        z_t     : ndarray (N,) — standardized residual at time t
                  (from GARCH filter: z_t = (r_t - mu) / sigma_t, scaled if needed)
        sigma_t : ndarray (N,) — conditional volatility at time t (% daily)

        Returns
        -------
        dict with keys:
            Q : ndarray (N, N) — updated pseudo-correlation matrix
            R : ndarray (N, N) — conditional correlation matrix
            H : ndarray (N, N) — conditional covariance matrix
        """
        z_t     = np.asarray(z_t, dtype=float)
        sigma_t = np.asarray(sigma_t, dtype=float)

        n_t = z_t * (z_t < 0) if self.model == 'ADCC' else None

        a = self.params[0]
        b = self.params[1]
        g = self.params[2] if self.model == 'ADCC' else 0.0

        Q_new = _update_Q(self.Q_state, z_t, n_t, self._AUQ, a, b, g, self.model)

        # Normalize to correlation
        sqrt_diag = np.sqrt(np.diag(Q_new))
        R_t = Q_new / np.outer(sqrt_diag, sqrt_diag)

        # Covariance
        H_t = sigma_t[:, None] * R_t * sigma_t[None, :]

        # Advance state
        self.Q_state = Q_new

        return {'Q': Q_new, 'R': R_t, 'H': H_t}

    # ------------------------------------------------------------------
    # Layer 2: periodic re-estimation on rolling window
    # ------------------------------------------------------------------

    def refit(
        self,
        returns_window: np.ndarray,
        model: str = None,
    ) -> None:
        """
        Re-estimate GJR-GARCH + DCC/ADCC parameters on a rolling window,
        then reset filter state with fresh estimates.

        Recommended cadence: monthly, rolling 5-year (1260-day) window.
        Rolling window allows asymmetry parameter g to shrink as crisis
        episodes drop out, avoiding the in-sample over-fitting identified
        in the OOS evaluation (outputs/interpretation.md §OOS).

        Parameters
        ----------
        returns_window : ndarray (T_window, N)
            Log returns for the rolling window period (unscaled).
        model : str or None
            Override model type; defaults to self.model.
        """
        if model is None:
            model = self.model
        if model not in ('DCC', 'ADCC'):
            raise ValueError(f"model must be 'DCC' or 'ADCC', got '{model}'")

        returns_window = np.asarray(returns_window, dtype=float)

        # Re-fit GJR-GARCH per asset
        garch_out = fit_multivariate_gjr(returns_window)
        Z_win     = garch_out['Z']
        sigma_win = garch_out['sigmas']

        # Re-fit DCC/ADCC
        result = fit(Z_win, sigma_win, model=model)

        # Update stored parameters
        self.model        = model
        self.params       = result['params']
        self.delta        = result['delta']
        self.garch_params = garch_out['params']

        # Recompute Q_bar from new window
        self.Q_bar = estimate_Qbar(Z_win)

        # Recompute N_bar if ADCC (fit() doesn't expose it in its result dict)
        N_mat = make_N_matrix(Z_win)
        if model == 'ADCC':
            self.N_bar = estimate_Nbar(N_mat)
            self.delta = compute_delta(self.Q_bar, self.N_bar)
        else:
            self.N_bar = None

        # Refresh AUQ intercept before warm-starting the filter
        self._refresh_AUQ()

        # Warm-start Q_state: run filter forward on window data with new params
        # so the state reflects recent correlation dynamics, not just Q_bar.
        a = self.params[0]
        b = self.params[1]
        g = self.params[2] if model == 'ADCC' else 0.0
        Q_t = self.Q_bar.copy()
        for t in range(1, Z_win.shape[0]):
            n_prev = N_mat[t - 1] if model == 'ADCC' else None
            Q_t = _update_Q(Q_t, Z_win[t - 1], n_prev, self._AUQ, a, b, g, model)
        self.Q_state = Q_t

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_AUQ(self) -> None:
        """Pre-compute AUQ = (1-a-b)*Q_bar [-g*N_bar]. Called on init and refit."""
        a = self.params[0]
        b = self.params[1]
        g = self.params[2] if self.model == 'ADCC' else 0.0

        if self.model == 'DCC':
            self._AUQ = (1 - a - b) * self.Q_bar
        else:
            if self.N_bar is None:
                raise ValueError("N_bar is required for ADCC but was not provided.")
            self._AUQ = (1 - a - b) * self.Q_bar - g * self.N_bar

    @classmethod
    def from_pipeline_result(cls, result: dict) -> 'DCCSystem':
        """
        Convenience constructor: build DCCSystem directly from run_pipeline() output.

        Model type (DCC vs ADCC) is inferred from the result: if result['dcc']['delta']
        is not None, the result came from an ADCC fit.

        Parameters
        ----------
        result : dict — output of project.pipeline.run_pipeline()

        Returns
        -------
        DCCSystem
        """
        dcc   = result['dcc']
        model = 'ADCC' if dcc.get('delta') is not None else 'DCC'

        Z     = result['garch']['Z']
        Q_bar = estimate_Qbar(Z)
        N_mat = make_N_matrix(Z)

        N_bar = estimate_Nbar(N_mat) if model == 'ADCC' else None
        delta = dcc.get('delta')

        return cls(
            params       = dcc['params'],
            Q_bar        = Q_bar,
            Q_state      = dcc['Q'][-1],   # last Q_t from the fitted path
            model        = model,
            N_bar        = N_bar,
            delta        = delta,
            garch_params = result['garch']['params'],
        )
