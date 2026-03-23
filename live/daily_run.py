"""
daily_run.py — Daily orchestrator for the live DCC-GARCH system.

Run once per trading day at ~4:30 pm ET (after US market close).

Steps:
  1. Download today's closing prices (yfinance)
  2. Store prices and compute log returns in DB
  3. Run GJR-GARCH filter with stored params -> sigma_t, z_t per asset
  4. Update DCC filter -> Q_t, R_t, H_{t+1|t}
  5. Evaluate yesterday's forecast (QLIKE)
  6. Persist forecast, eval result, DCC state to DB

Usage
-----
    # Daily invocation (cron / scheduler):
    python live/daily_run.py

    # First-run bootstrap (initializes DB from historical data):
    python live/daily_run.py --bootstrap
"""

import argparse
import os
import pickle
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from live.data_layer import (
    DB_PATH, TICKERS,
    init_db, download_latest_prices, download_full_history, store_prices,
    load_returns, load_garch_params, save_garch_params,
    load_dcc_params, save_dcc_params,
    load_dcc_state, save_dcc_state,
    load_forecast, save_forecast, bulk_save_forecasts,
    save_eval, db_has_been_initialized,
)
from python.garch import fit_multivariate_gjr, filter_gjr_garch
from python.dcc.model import _update_Q
from python.dcc.utils import estimate_Qbar, make_N_matrix, estimate_Nbar, compute_delta


# ── QLIKE loss ────────────────────────────────────────────────────────────────

def compute_qlike(H: np.ndarray, r: np.ndarray) -> float:
    """
    QLIKE loss: log|H| + r' H^{-1} r

    Units: H must be in (% daily)^2 and r in % daily (both scaled x100).

    Parameters
    ----------
    H : ndarray (N, N) — covariance forecast
    r : ndarray (N,)   — realized return vector (same units as H)

    Returns
    -------
    float — QLIKE scalar (lower is better)
    """
    L      = np.linalg.cholesky(H)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    y      = np.linalg.solve(L, r)
    quad   = float(y @ y)
    return logdet + quad


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap(model: str = 'DCC') -> None:
    """
    First-run initialization: download full price history from Yahoo Finance,
    fit GJR+DCC, then run the filter forward to set the initial DCC state.

    All data comes from Yahoo Finance so the bootstrap and daily updates share
    the same price source — no unit mismatch.

    Stores to: data/live.db
    """
    print('[bootstrap] Initializing live.db from Yahoo Finance history...')

    # Always start from a clean DB so stale data from prior bootstraps
    # (e.g., Bloomberg total return index values) cannot corrupt returns.
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print('[bootstrap] Removed stale live.db.')
    init_db(DB_PATH)

    print('[bootstrap] Downloading full price history (this may take ~30s)...')
    prices = download_full_history(TICKERS, start='1998-01-01')
    print(f'[bootstrap] Downloaded: {prices.shape[0]} rows, {prices.shape[1]} assets')
    print(f'[bootstrap] Date range: {prices.index[0].date()} to {prices.index[-1].date()}')

    print('[bootstrap] Storing prices to DB...')
    store_prices(prices, DB_PATH)

    returns_df = load_returns(TICKERS, db_path=DB_PATH)  # % daily log returns
    print(f'[bootstrap] Returns shape: {returns_df.shape}')

    # ── Fit GJR-GARCH ─────────────────────────────────────────────────────────
    print('[bootstrap] Fitting GJR-GARCH per asset...')
    returns_raw = returns_df.values / 100.0   # unscale: module scales internally
    garch_out = fit_multivariate_gjr(returns_raw)
    today_str = str(date.today())
    save_garch_params(garch_out['params'], today_str, TICKERS, DB_PATH)
    print('[bootstrap] GJR-GARCH params saved.')

    Z      = garch_out['Z']       # (T, N)
    sigmas = garch_out['sigmas']  # (T, N) % daily

    # ── Fit DCC ───────────────────────────────────────────────────────────────
    from python.dcc import fit
    print(f'[bootstrap] Fitting {model} on Z ({Z.shape})...')
    result  = fit(Z, sigmas, model=model)
    params  = result['params']
    delta   = result['delta']
    save_dcc_params(params, today_str, model, delta, DB_PATH)
    print(f'[bootstrap] {model} params: {params}')

    # ── Build DCC state ───────────────────────────────────────────────────────
    Q_bar = estimate_Qbar(Z)
    N_mat = make_N_matrix(Z)
    N_bar = estimate_Nbar(N_mat) if model == 'ADCC' else None
    a, b  = float(params[0]), float(params[1])
    g     = float(params[2]) if model == 'ADCC' else 0.0
    AUQ   = (1 - a - b) * Q_bar - (g * N_bar if N_bar is not None else 0)

    # ── Filter forward and collect full forecast history ─────────────────────
    print('[bootstrap] Running filter forward and collecting forecast history...')
    Q_t = Q_bar.copy()
    forecast_rows = []
    for t in range(1, Z.shape[0]):
        n_prev = N_mat[t - 1] if model == 'ADCC' else None
        Q_t = _update_Q(Q_t, Z[t - 1], n_prev, AUQ, a, b, g, model)
        sqrt_diag = np.sqrt(np.diag(Q_t))
        R_t       = Q_t / np.outer(sqrt_diag, sqrt_diag)
        sigma_t   = sigmas[t]
        H_t       = sigma_t[:, None] * R_t * sigma_t[None, :]
        r_t       = returns_df.values[t]
        date_t    = str(returns_df.index[t].date())
        forecast_rows.append((date_t, model, H_t, sigma_t, r_t))

    last_date = str(returns_df.index[-1].date())
    save_dcc_state(last_date, model, Q_t, Q_bar, N_bar, DB_PATH)
    print(f'[bootstrap] DCC state saved for {last_date}.')

    print(f'[bootstrap] Saving {len(forecast_rows)} historical forecasts...')
    bulk_save_forecasts(forecast_rows, DB_PATH)

    print('[bootstrap] Backfilling QLIKE history...')
    backfill_qlike(model=model)
    print('[bootstrap] Done. live.db is ready.')


# ── Daily update ──────────────────────────────────────────────────────────────

def run_daily(model: str = 'DCC') -> None:
    """
    Main daily job. Safe to run even if already executed today (idempotent).
    """
    print('[daily] Starting daily DCC update...')

    # ── Step 1: Download and store latest prices ───────────────────────────────
    try:
        prices = download_latest_prices(TICKERS, lookback_days=10)
    except Exception as e:
        print(f'[daily] WARNING: yfinance download failed: {e}')
        print('[daily] Trying to proceed with cached data.')
        prices = None

    if prices is not None and not prices.empty:
        store_prices(prices, DB_PATH)
        today_str = str(prices.index[-1].date())
        print(f'[daily] Latest price date: {today_str}')
    else:
        # Fall back to whatever is already in the DB
        from live.data_layer import get_latest_date
        today_str = get_latest_date(DB_PATH) or str(date.today())
        print(f'[daily] Using cached data, latest date: {today_str}')

    # ── Step 2: Load full returns history ─────────────────────────────────────
    returns_df = load_returns(TICKERS, db_path=DB_PATH)
    if returns_df.shape[0] < 100:
        print('[daily] ERROR: insufficient return history. Run --bootstrap first.')
        return

    returns_raw = returns_df.values / 100.0   # unscale for GJR module
    T, N = returns_raw.shape

    # ── Step 3: GJR-GARCH filter (fixed params) ───────────────────────────────
    garch_params = load_garch_params(TICKERS, DB_PATH)
    if garch_params is None:
        print('[daily] No GARCH params in DB. Run --bootstrap first.')
        return

    z_t_all     = np.empty(N)
    sigma_t_all = np.empty(N)
    for i, (ticker, params_i) in enumerate(zip(TICKERS, garch_params)):
        out = filter_gjr_garch(returns_raw[:, i], params_i)
        z_t_all[i]     = out['std_residuals'][-1]
        sigma_t_all[i] = out['sigmas'][-1]

    print(f'[daily] GARCH filter done. sigma_t: {sigma_t_all.round(4)}')

    # ── Step 4: Load DCC state and update ─────────────────────────────────────
    state = load_dcc_state(model, DB_PATH)
    if state is None:
        print('[daily] No DCC state in DB. Run --bootstrap first.')
        return

    dcc_p = load_dcc_params(model, DB_PATH)
    a, b  = dcc_p['a'], dcc_p['b']
    g     = dcc_p['g'] if dcc_p['g'] is not None else 0.0
    Q_bar = state['q_bar']
    N_bar = state['n_bar']
    Q_prev = state['q_state']

    AUQ   = (1 - a - b) * Q_bar - (g * N_bar if N_bar is not None else 0)
    n_t   = z_t_all * (z_t_all < 0) if model == 'ADCC' else None
    Q_new = _update_Q(Q_prev, z_t_all, n_t, AUQ, a, b, g, model)

    sqrt_diag = np.sqrt(np.diag(Q_new))
    R_t = Q_new / np.outer(sqrt_diag, sqrt_diag)
    H_t = sigma_t_all[:, None] * R_t * sigma_t_all[None, :]

    # ── Step 5: Evaluate yesterday's forecast ────────────────────────────────
    # The forecast stored on the previous update is for today's date.
    # "Today's realized return" = last row of returns_df.
    prev_forecast = load_forecast(state['date'], model, DB_PATH)
    if prev_forecast is not None:
        r_today_pct = returns_df.values[-1]   # % daily (x100)
        try:
            qlike = compute_qlike(prev_forecast['H'], r_today_pct)
            save_eval(today_str, model, qlike, DB_PATH)
            print(f'[daily] QLIKE for {today_str}: {qlike:.4f}')
        except np.linalg.LinAlgError:
            print('[daily] WARNING: H_prev not PD, skipping QLIKE.')
    else:
        print(f'[daily] No prior forecast found for {state["date"]}, skipping eval.')

    # ── Step 6: Persist new state and forecast ────────────────────────────────
    r_today_pct = returns_df.values[-1]
    save_dcc_state(today_str, model, Q_new, Q_bar, N_bar, DB_PATH)
    save_forecast(today_str, model, H_t, sigma_t_all, r_today_pct, DB_PATH)
    print(f'[daily] Forecast and state saved for {today_str}.')
    print('[daily] Done.')


# ── Monthly refit ─────────────────────────────────────────────────────────────

def run_monthly_refit(model: str = 'DCC', window_days: int = 1260) -> None:
    """
    Re-estimate GJR-GARCH + DCC/ADCC on a rolling 5-year window.

    Updates garch_params and dcc_params tables.
    Resets DCC state by running filter forward on window with new params.
    """
    print(f'[refit] Starting monthly re-estimation ({model}, window={window_days}d)...')

    returns_df  = load_returns(TICKERS, last_n=window_days, db_path=DB_PATH)
    returns_raw = returns_df.values / 100.0
    print(f'[refit] Window: {returns_df.index[0].date()} to {returns_df.index[-1].date()}, T={len(returns_df)}')

    # Re-fit GJR
    print('[refit] Fitting GJR-GARCH...')
    garch_out = fit_multivariate_gjr(returns_raw)
    today_str = str(date.today())
    save_garch_params(garch_out['params'], today_str, TICKERS, DB_PATH)

    Z      = garch_out['Z']
    sigmas = garch_out['sigmas']

    # Re-fit DCC
    from python.dcc import fit
    print(f'[refit] Fitting {model}...')
    result = fit(Z, sigmas, model=model)
    params = result['params']
    delta  = result['delta']
    save_dcc_params(params, today_str, model, delta, DB_PATH)
    print(f'[refit] {model} params: {params}')

    # Warm-start DCC state
    Q_bar = estimate_Qbar(Z)
    N_mat = make_N_matrix(Z)
    N_bar = estimate_Nbar(N_mat) if model == 'ADCC' else None
    a, b  = float(params[0]), float(params[1])
    g     = float(params[2]) if model == 'ADCC' else 0.0
    AUQ   = (1 - a - b) * Q_bar - (g * N_bar if N_bar is not None else 0)

    Q_t = Q_bar.copy()
    for t in range(1, Z.shape[0]):
        n_prev = N_mat[t - 1] if model == 'ADCC' else None
        Q_t = _update_Q(Q_t, Z[t - 1], n_prev, AUQ, a, b, g, model)

    last_date = str(returns_df.index[-1].date())
    save_dcc_state(last_date, model, Q_t, Q_bar, N_bar, DB_PATH)
    print(f'[refit] Done. State saved for {last_date}.')


# ── QLIKE backfill ────────────────────────────────────────────────────────────

def backfill_qlike(model: str = 'DCC') -> None:
    """
    Compute QLIKE for all historical forecast pairs and populate eval_results.

    For each consecutive pair of forecast rows (t, t+1):
        QLIKE at t+1 = log|H_t| + r_{t+1}' H_t^{-1} r_{t+1}
    where H_t is stored in the forecast row for date t,
    and r_{t+1} is stored as r_vec in the forecast row for date t+1.
    """
    import sqlite3
    print(f'[backfill] Loading all forecasts for {model}...')
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT forecast_date, h_matrix, r_vec FROM forecasts "
            "WHERE model = ? ORDER BY forecast_date",
            (model,),
        ).fetchall()

    print(f'[backfill] Computing QLIKE for {len(rows) - 1} days...')
    evals = []
    skipped = 0
    for i in range(len(rows) - 1):
        dt_next, _, r_b = rows[i + 1]
        _, h_b, _       = rows[i]
        if r_b is None:
            skipped += 1
            continue
        H = pickle.loads(h_b)
        r = pickle.loads(r_b)
        try:
            qlike = compute_qlike(H, r)
            evals.append((dt_next, model, float(qlike)))
        except np.linalg.LinAlgError:
            skipped += 1

    with sqlite3.connect(DB_PATH) as con:
        con.executemany(
            "INSERT OR REPLACE INTO eval_results (eval_date, model, qlike) VALUES (?, ?, ?)",
            evals,
        )
    print(f'[backfill] Done. Saved {len(evals)} QLIKE values ({skipped} skipped).')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCC-GARCH daily live system')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Initialize DB from historical ETF Excel files')
    parser.add_argument('--refit', action='store_true',
                        help='Run monthly re-estimation')
    parser.add_argument('--backfill-qlike', action='store_true',
                        help='Backfill QLIKE history from stored forecasts')
    parser.add_argument('--model', default='DCC', choices=['DCC', 'ADCC'],
                        help='Model to use (default: DCC)')
    args = parser.parse_args()

    if args.bootstrap:
        bootstrap(model=args.model)
    elif args.refit:
        init_db(DB_PATH)
        run_monthly_refit(model=args.model)
    elif args.backfill_qlike:
        init_db(DB_PATH)
        backfill_qlike(model=args.model)
    else:
        init_db(DB_PATH)
        if not db_has_been_initialized(DB_PATH):
            print('[daily] DB not initialized. Running bootstrap first...')
            bootstrap(model=args.model)
        run_daily(model=args.model)
