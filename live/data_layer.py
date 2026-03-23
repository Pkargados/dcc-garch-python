"""
data_layer.py — yfinance download and SQLite persistence for the live DCC system.

Responsibilities:
  - Download closing prices from yfinance
  - Read/write all model state to SQLite (live.db)
  - No model logic here — pure I/O

Database: data/live.db  (auto-created on first call to init_db())
"""

import os
import pickle
import sqlite3
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ── Browser-like session (bypasses Yahoo Finance 429 rate limiting) ────────────
#
# Yahoo Finance returns HTTP 429 or empty JSON when requests come from
# non-browser User-Agents.  Injecting a session with Chrome headers fixes this.

def _make_yf_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/122.0.0.0 Safari/537.36'
        ),
        'Accept':          'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer':         'https://finance.yahoo.com/',
        'Origin':          'https://finance.yahoo.com',
    })
    return s

# ── Constants ──────────────────────────────────────────────────────────────────

TICKERS = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.abspath(os.path.join(_HERE, '..'))
DB_PATH = os.path.join(_ROOT, 'data', 'live.db')


# ── Database initialization ────────────────────────────────────────────────────

def init_db(db_path: str = DB_PATH) -> None:
    """Create all tables if they do not exist."""
    with sqlite3.connect(db_path) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            date        TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            close       REAL NOT NULL,
            return_pct  REAL,
            PRIMARY KEY (date, ticker)
        );

        CREATE TABLE IF NOT EXISTS garch_params (
            fitted_date TEXT NOT NULL,
            ticker      TEXT NOT NULL,
            mu          REAL,
            omega       REAL,
            alpha       REAL,
            gamma       REAL,
            beta        REAL,
            nu          REAL,
            PRIMARY KEY (fitted_date, ticker)
        );

        CREATE TABLE IF NOT EXISTS dcc_params (
            fitted_date TEXT NOT NULL,
            model       TEXT NOT NULL,
            a           REAL,
            b           REAL,
            g           REAL,
            delta       REAL,
            PRIMARY KEY (fitted_date, model)
        );

        CREATE TABLE IF NOT EXISTS dcc_state (
            date        TEXT NOT NULL,
            model       TEXT NOT NULL,
            q_state     BLOB,
            q_bar       BLOB,
            n_bar       BLOB,
            PRIMARY KEY (date, model)
        );

        CREATE TABLE IF NOT EXISTS forecasts (
            forecast_date TEXT NOT NULL,
            model         TEXT NOT NULL,
            h_matrix      BLOB,
            sigma_vec     BLOB,
            r_vec         BLOB,
            PRIMARY KEY (forecast_date, model)
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            eval_date   TEXT NOT NULL,
            model       TEXT NOT NULL,
            qlike       REAL,
            PRIMARY KEY (eval_date, model)
        );
        """)


# ── Price download ────────────────────────────────────────────────────────────

def _fetch_yahoo(ticker: str, period1: int, period2: int,
                 session: requests.Session) -> pd.Series:
    """
    Fetch daily adjusted close prices from Yahoo Finance v8 chart API.
    Returns a pd.Series indexed by normalized date, or raises on failure.
    """
    import time as _time
    url = (
        f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
        f'?interval=1d&period1={period1}&period2={period2}'
    )
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    result = data['chart']['result']
    if not result:
        raise ValueError(f'No data for {ticker}')
    timestamps = result[0]['timestamp']
    closes     = result[0]['indicators']['adjclose'][0]['adjclose']
    idx = pd.to_datetime(timestamps, unit='s').normalize()
    s   = pd.Series(closes, index=idx, name=ticker, dtype=float)
    # Keep last entry per day in case Yahoo returns intraday duplicates
    s   = s[~s.index.duplicated(keep='last')]
    return s


def download_full_history(tickers: list = TICKERS,
                          start: str = '1998-01-01') -> pd.DataFrame:
    """
    Download full adjusted close price history for all tickers from Yahoo Finance.

    Used for the one-time bootstrap.  Fetches from `start` to today.

    Parameters
    ----------
    tickers : list
    start   : str — ISO date string for the beginning of history

    Returns
    -------
    pd.DataFrame — close prices, index=date, columns=tickers
    """
    import time
    p1      = int(pd.Timestamp(start).timestamp())
    p2      = int(pd.Timestamp(date.today() + timedelta(days=1)).timestamp())
    session = _make_yf_session()

    frames = {}
    for ticker in tickers:
        try:
            frames[ticker] = _fetch_yahoo(ticker, p1, p2, session)
            print(f'  {ticker}: {len(frames[ticker])} rows')
            time.sleep(0.3)
        except Exception as e:
            print(f'  WARNING: {ticker} failed: {e}')

    if not frames:
        raise ValueError('No data downloaded from Yahoo Finance.')

    prices = pd.DataFrame(frames).reindex(columns=tickers)
    prices = prices.dropna(how='all')
    return prices

def download_latest_prices(tickers: list = TICKERS, lookback_days: int = 10) -> pd.DataFrame:
    """
    Download recent closing prices from yfinance using a browser-like session.

    Yahoo Finance rate-limits non-browser User-Agents (HTTP 429 / empty JSON).
    We inject a Chrome-headers session to bypass this.

    Returns a DataFrame with columns = tickers, index = date (date), sorted ascending.
    Only rows where ALL tickers have data are returned.

    Parameters
    ----------
    tickers       : list of ticker strings
    lookback_days : int — calendar days to fetch (10 covers weekends/holidays)
    """
    import time
    end     = date.today()
    start_d = end - timedelta(days=lookback_days)
    p1      = int(pd.Timestamp(start_d).timestamp())
    p2      = int(pd.Timestamp(end + timedelta(days=1)).timestamp())
    session = _make_yf_session()

    frames = {}
    for ticker in tickers:
        try:
            frames[ticker] = _fetch_yahoo(ticker, p1, p2, session)
            time.sleep(0.2)
        except Exception as e:
            print(f'[download] WARNING: {ticker} failed: {type(e).__name__}: {e}')

    if not frames:
        raise ValueError(
            'Yahoo Finance returned no data for any ticker. '
            'Check network connectivity.'
        )

    prices = pd.DataFrame(frames).reindex(columns=tickers)
    prices = prices.dropna(how='all')
    return prices


def store_prices(prices: pd.DataFrame, db_path: str = DB_PATH) -> None:
    """
    Persist a close-price DataFrame to the prices table.

    Computes log returns x100 (return_pct) on insert.
    Existing rows (same date + ticker) are ignored.

    Parameters
    ----------
    prices : DataFrame — close prices, index=date, columns=tickers
    """
    rows = []
    for dt in prices.index:
        for ticker in prices.columns:
            close = prices.loc[dt, ticker]
            if pd.isna(close):
                continue
            rows.append((str(dt.date()), ticker, float(close)))

    with sqlite3.connect(db_path) as con:
        con.executemany(
            "INSERT OR IGNORE INTO prices (date, ticker, close) VALUES (?, ?, ?)",
            rows,
        )

    # Recompute return_pct for all newly inserted rows
    _recompute_returns(db_path)


def _recompute_returns(db_path: str) -> None:
    """Recompute log return x100 for all rows where return_pct is NULL."""
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(
            "SELECT date, ticker, close FROM prices ORDER BY ticker, date",
            con,
        )
    if df.empty:
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    df['return_pct'] = df.groupby('ticker')['close'].transform(
        lambda x: np.log(x / x.shift(1)) * 100
    )

    rows = [
        (float(r['return_pct']), str(r['date'].date()), r['ticker'])
        for _, r in df.iterrows()
        if not pd.isna(r['return_pct'])
    ]
    with sqlite3.connect(db_path) as con:
        con.executemany(
            "UPDATE prices SET return_pct = ? WHERE date = ? AND ticker = ?",
            rows,
        )


# ── Price / returns loaders ───────────────────────────────────────────────────

def load_all_prices(tickers: list = TICKERS, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Load full close-price history as a DataFrame (date x ticker).
    """
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(
            "SELECT date, ticker, close FROM prices ORDER BY date",
            con,
            parse_dates=['date'],
        )
    if df.empty:
        return pd.DataFrame(columns=['date'] + tickers).set_index('date')
    pivot = df.pivot(index='date', columns='ticker', values='close')
    return pivot.reindex(columns=tickers)


def load_returns(tickers: list = TICKERS,
                 last_n: Optional[int] = None,
                 db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Load log returns (x100) as a DataFrame (date x ticker).

    Parameters
    ----------
    last_n : int or None — if given, return only the last N rows
    """
    query = "SELECT date, ticker, return_pct FROM prices WHERE return_pct IS NOT NULL ORDER BY date"
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(query, con, parse_dates=['date'])
    if df.empty:
        return pd.DataFrame(columns=['date'] + tickers).set_index('date')
    pivot = df.pivot(index='date', columns='ticker', values='return_pct')
    pivot = pivot.reindex(columns=tickers).dropna(how='any')
    if last_n is not None:
        pivot = pivot.iloc[-last_n:]
    return pivot


# ── GARCH params ──────────────────────────────────────────────────────────────

def save_garch_params(params_list: list, fitted_date: str,
                      tickers: list = TICKERS, db_path: str = DB_PATH) -> None:
    """
    Persist fitted GJR-GARCH parameter Series list to the DB.

    Parameters
    ----------
    params_list : list of N pandas Series (output of fit_multivariate_gjr)
    fitted_date : str — ISO date string (e.g. '2026-03-20')
    """
    rows = []
    for ticker, params in zip(tickers, params_list):
        p = params
        rows.append((
            fitted_date, ticker,
            float(p.get('Const', p.get('mu', 0.0))),
            float(p['omega']),
            float(p['alpha[1]']),
            float(p['gamma[1]']),
            float(p['beta[1]']),
            float(p['nu']),
        ))
    with sqlite3.connect(db_path) as con:
        con.executemany(
            """INSERT OR REPLACE INTO garch_params
               (fitted_date, ticker, mu, omega, alpha, gamma, beta, nu)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )


def load_garch_params(tickers: list = TICKERS,
                      db_path: str = DB_PATH) -> Optional[list]:
    """
    Load the most recently fitted GJR-GARCH params as a list of pandas Series.

    Returns None if no params are stored.
    """
    with sqlite3.connect(db_path) as con:
        latest = con.execute(
            "SELECT MAX(fitted_date) FROM garch_params"
        ).fetchone()[0]
        if latest is None:
            return None
        df = pd.read_sql(
            "SELECT * FROM garch_params WHERE fitted_date = ? ORDER BY ticker",
            con, params=(latest,),
        )

    result = []
    for ticker in tickers:
        row = df[df['ticker'] == ticker].iloc[0]
        s = pd.Series({
            'Const':      row['mu'],
            'omega':      row['omega'],
            'alpha[1]':   row['alpha'],
            'gamma[1]':   row['gamma'],
            'beta[1]':    row['beta'],
            'nu':         row['nu'],
        })
        result.append(s)
    return result


# ── DCC params ────────────────────────────────────────────────────────────────

def save_dcc_params(params: np.ndarray, fitted_date: str,
                    model: str, delta: Optional[float] = None,
                    db_path: str = DB_PATH) -> None:
    a = float(params[0])
    b = float(params[1])
    g = float(params[2]) if len(params) > 2 else None
    with sqlite3.connect(db_path) as con:
        con.execute(
            """INSERT OR REPLACE INTO dcc_params
               (fitted_date, model, a, b, g, delta) VALUES (?, ?, ?, ?, ?, ?)""",
            (fitted_date, model, a, b, g, delta),
        )


def load_dcc_params(model: str = 'DCC',
                    db_path: str = DB_PATH) -> Optional[dict]:
    """Load the most recently fitted DCC/ADCC params. Returns None if not stored."""
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT * FROM dcc_params WHERE model = ? ORDER BY fitted_date DESC LIMIT 1",
            (model,),
        ).fetchone()
    if row is None:
        return None
    cols = ['fitted_date', 'model', 'a', 'b', 'g', 'delta']
    return dict(zip(cols, row))


# ── DCC filter state ──────────────────────────────────────────────────────────

def save_dcc_state(dt: str, model: str,
                   q_state: np.ndarray, q_bar: np.ndarray,
                   n_bar: Optional[np.ndarray] = None,
                   db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            """INSERT OR REPLACE INTO dcc_state
               (date, model, q_state, q_bar, n_bar) VALUES (?, ?, ?, ?, ?)""",
            (dt, model,
             pickle.dumps(q_state),
             pickle.dumps(q_bar),
             pickle.dumps(n_bar) if n_bar is not None else None),
        )


def load_dcc_state(model: str = 'DCC',
                   db_path: str = DB_PATH) -> Optional[dict]:
    """Load the most recent DCC state. Returns None if not stored."""
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT date, q_state, q_bar, n_bar FROM dcc_state WHERE model = ? ORDER BY date DESC LIMIT 1",
            (model,),
        ).fetchone()
    if row is None:
        return None
    dt, q_state_b, q_bar_b, n_bar_b = row
    return {
        'date':    dt,
        'q_state': pickle.loads(q_state_b),
        'q_bar':   pickle.loads(q_bar_b),
        'n_bar':   pickle.loads(n_bar_b) if n_bar_b is not None else None,
    }


# ── Forecasts ────────────────────────────────────────────────────────────────

def save_forecast(forecast_date: str, model: str,
                  h_matrix: np.ndarray, sigma_vec: np.ndarray,
                  r_vec: Optional[np.ndarray] = None,
                  db_path: str = DB_PATH) -> None:
    """
    Save H_{t+1|t} (the forecast for `forecast_date`'s next business day).

    `r_vec` is today's return vector (% daily) — stored alongside the forecast
    for convenient QLIKE evaluation tomorrow.
    """
    with sqlite3.connect(db_path) as con:
        con.execute(
            """INSERT OR REPLACE INTO forecasts
               (forecast_date, model, h_matrix, sigma_vec, r_vec)
               VALUES (?, ?, ?, ?, ?)""",
            (forecast_date, model,
             pickle.dumps(h_matrix),
             pickle.dumps(sigma_vec),
             pickle.dumps(r_vec) if r_vec is not None else None),
        )


def bulk_save_forecasts(rows: list, db_path: str = DB_PATH) -> None:
    """
    Bulk insert forecasts in a single transaction (used by bootstrap).

    `rows` is a list of (forecast_date, model, h_matrix, sigma_vec, r_vec) tuples.
    """
    with sqlite3.connect(db_path) as con:
        con.executemany(
            """INSERT OR REPLACE INTO forecasts
               (forecast_date, model, h_matrix, sigma_vec, r_vec)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (dt, mdl,
                 pickle.dumps(H),
                 pickle.dumps(sig),
                 pickle.dumps(r) if r is not None else None)
                for dt, mdl, H, sig, r in rows
            ],
        )


def load_forecast(forecast_date: str, model: str = 'DCC',
                  db_path: str = DB_PATH) -> Optional[dict]:
    """Load the forecast stored on `forecast_date`."""
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT h_matrix, sigma_vec, r_vec FROM forecasts WHERE forecast_date = ? AND model = ?",
            (forecast_date, model),
        ).fetchone()
    if row is None:
        return None
    h_b, s_b, r_b = row
    return {
        'H':     pickle.loads(h_b),
        'sigma': pickle.loads(s_b),
        'r':     pickle.loads(r_b) if r_b is not None else None,
    }


def load_all_forecasts(model: str = 'DCC',
                       db_path: str = DB_PATH) -> pd.DataFrame:
    """Load all forecast dates and their QLIKE evaluations (joined)."""
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(
            """SELECT f.forecast_date, e.qlike
               FROM forecasts f
               LEFT JOIN eval_results e
                 ON f.forecast_date = e.eval_date AND f.model = e.model
               WHERE f.model = ?
               ORDER BY f.forecast_date""",
            con, params=(model,), parse_dates=['forecast_date'],
        )
    return df


# ── Evaluation results ────────────────────────────────────────────────────────

def save_eval(eval_date: str, model: str, qlike: float,
              db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT OR REPLACE INTO eval_results (eval_date, model, qlike) VALUES (?, ?, ?)",
            (eval_date, model, qlike),
        )


def load_eval_history(model: str = 'DCC',
                      db_path: str = DB_PATH) -> pd.DataFrame:
    """Return full QLIKE history as a DataFrame with columns [eval_date, qlike]."""
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(
            "SELECT eval_date, qlike FROM eval_results WHERE model = ? ORDER BY eval_date",
            con, params=(model,), parse_dates=['eval_date'],
        )
    return df


# ── Latest state query ────────────────────────────────────────────────────────

def get_latest_date(db_path: str = DB_PATH) -> Optional[str]:
    """Return the most recent date for which a forecast exists."""
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT MAX(forecast_date) FROM forecasts"
        ).fetchone()
    return row[0] if row else None


def db_has_been_initialized(db_path: str = DB_PATH) -> bool:
    """True if the DB exists and contains at least one forecast row."""
    if not os.path.exists(db_path):
        return False
    with sqlite3.connect(db_path) as con:
        n = con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    return n > 0
