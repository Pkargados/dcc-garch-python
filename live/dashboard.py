"""
dashboard.py — Streamlit dashboard for the live DCC-GARCH system.

Run with:
    streamlit run live/dashboard.py

Reads ONLY from data/live.db — no model code here.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from live.data_layer import (
    DB_PATH,
    TICKERS,
    db_has_been_initialized,
    get_latest_date,
    load_dcc_params,
    load_eval_history,
    load_forecast,
    load_garch_params,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='DCC-GARCH Live',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── Helper functions (must be defined before use) ─────────────────────────────

@st.cache_data(ttl=300)
def _load_r_and_sigma_history(model: str):
    """Load per-day R, H, sigma, and realized returns from stored forecasts. Cached 5 min."""
    import sqlite3
    with sqlite3.connect(DB_PATH) as con:
        rows = con.execute(
            "SELECT forecast_date, h_matrix, sigma_vec, r_vec FROM forecasts "
            "WHERE model = ? ORDER BY forecast_date",
            (model,),
        ).fetchall()
    dates, R_list, H_list, sigma_list, r_list = [], [], [], [], []
    for dt, h_b, s_b, r_b in rows:
        H   = pickle.loads(h_b)
        sig = pickle.loads(s_b)
        r   = pickle.loads(r_b) if r_b is not None else None
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.where(
                np.outer(sig, sig) > 0,
                H / np.outer(sig, sig),
                0.0,
            )
        np.fill_diagonal(R, 1.0)
        dates.append(pd.to_datetime(dt))
        R_list.append(R)
        H_list.append(H)
        sigma_list.append(sig)
        r_list.append(r)
    return dates, R_list, H_list, sigma_list, r_list


def _qlike_delta_str(eval_hist: pd.DataFrame) -> str:
    last = eval_hist['qlike'].iloc[-1]
    mean = eval_hist['qlike'].tail(60).mean()
    return f"{last - mean:+.4f} vs 60d mean"


def _corr_heatmap(R: np.ndarray) -> go.Figure:
    fig = px.imshow(
        R,
        x=TICKERS,
        y=TICKERS,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        text_auto='.2f',
        aspect='auto',
    )
    fig.update_layout(height=480, coloraxis_colorbar_title='Correlation')
    return fig


def _pair_history_chart(pair: str, model: str) -> go.Figure:
    a_ticker, b_ticker = pair.split('-')
    i, j = TICKERS.index(a_ticker), TICKERS.index(b_ticker)
    dates, R_list, _, _, _ = _load_r_and_sigma_history(model)
    corr_series = [R[i, j] for R in R_list]
    fig = go.Figure(go.Scatter(
        x=dates,
        y=corr_series,
        mode='lines',
        line=dict(color='steelblue', width=1.2),
    ))
    fig.update_layout(
        yaxis_title=f'Correlation {pair}',
        yaxis_range=[-1, 1],
        height=300,
        margin=dict(t=20),
    )
    return fig


def _avg_corr_history_chart(model: str) -> go.Figure:
    """Average pairwise correlation over time."""
    dates, R_list, _, _, _ = _load_r_and_sigma_history(model)
    n = len(TICKERS)
    avg_corr = [(R.sum() - n) / (n * (n - 1)) for R in R_list]
    fig = go.Figure(go.Scatter(
        x=dates, y=avg_corr, mode='lines',
        line=dict(color='firebrick', width=1.2),
    ))
    fig.update_layout(
        yaxis_title='Average Pairwise Correlation',
        height=280, margin=dict(t=20),
    )
    return fig


def _sigma_history_chart(selected_tickers: list, model: str) -> go.Figure:
    dates, _, _, sigma_list, _ = _load_r_and_sigma_history(model)
    sigma_arr = np.array(sigma_list) * np.sqrt(252)   # annualize
    fig = go.Figure()
    for ticker in selected_tickers:
        i = TICKERS.index(ticker)
        fig.add_trace(go.Scatter(
            x=dates,
            y=sigma_arr[:, i],
            mode='lines',
            name=ticker,
        ))
    fig.update_layout(yaxis_title='Annualized Vol (%)', height=350, margin=dict(t=20))
    return fig


def _portfolio_vol_chart(model: str) -> go.Figure:
    """Equal-weight portfolio annualized volatility over time."""
    dates, _, H_list, _, _ = _load_r_and_sigma_history(model)
    n = len(TICKERS)
    w = np.ones(n) / n
    port_vol = [np.sqrt(max(w @ H @ w, 0.0)) * np.sqrt(252) for H in H_list]
    fig = go.Figure(go.Scatter(
        x=dates, y=port_vol, mode='lines',
        line=dict(color='darkorange', width=1.5),
        fill='tozeroy', fillcolor='rgba(255,165,0,0.08)',
    ))
    fig.update_layout(
        yaxis_title='Equal-Weight Portfolio Vol (% ann.)',
        height=300, margin=dict(t=20),
    )
    return fig


def _forecast_vs_realized_chart(ticker: str, model: str) -> go.Figure:
    """Forecast sigma vs |realized return| for a single asset."""
    dates, _, _, sigma_list, r_list = _load_r_and_sigma_history(model)
    i = TICKERS.index(ticker)
    forecast_sig = [s[i] for s in sigma_list]
    realized_abs = [abs(r[i]) if r is not None else np.nan for r in r_list]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=realized_abs, mode='lines', name='|Realized return|',
        line=dict(color='steelblue', width=0.8),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=forecast_sig, mode='lines', name='Forecast σ',
        line=dict(color='firebrick', width=1.5),
    ))
    fig.update_layout(
        yaxis_title='% daily', height=320, margin=dict(t=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def _qlike_chart(eval_hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eval_hist['eval_date'],
        y=eval_hist['qlike'],
        mode='lines',
        name='QLIKE',
        line=dict(color='steelblue', width=1),
    ))
    roll = eval_hist['qlike'].rolling(60, min_periods=10).mean()
    fig.add_trace(go.Scatter(
        x=eval_hist['eval_date'],
        y=roll,
        mode='lines',
        name='60-day rolling mean',
        line=dict(color='orange', width=2),
    ))
    fig.update_layout(
        yaxis_title='QLIKE',
        height=380,
        margin=dict(t=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title('DCC-GARCH Live')
    model = st.selectbox('Model', ['DCC', 'ADCC'], index=0)
    st.markdown('---')
    st.markdown('**Universe:** 10 US Sector ETFs')
    st.markdown('**GARCH:** GJR-GARCH(1,1,1)-t')
    st.markdown('**Correlation:** DCC / ADCC')
    st.markdown('---')
    if st.button('Clear cache'):
        st.cache_data.clear()
        st.rerun()

# ── Guard: not initialized ────────────────────────────────────────────────────

if not db_has_been_initialized(DB_PATH):
    st.error(
        '**Database not initialized.**\n\n'
        'Run the bootstrap first from the project root:\n\n'
        '```bash\npython live/daily_run.py --bootstrap\n```'
    )
    st.stop()

# ── Load state ────────────────────────────────────────────────────────────────

latest_date = get_latest_date(DB_PATH)
forecast    = load_forecast(latest_date, model, DB_PATH) if latest_date else None
eval_hist   = load_eval_history(model, DB_PATH)
dcc_p       = load_dcc_params(model, DB_PATH)

# ── Page header ───────────────────────────────────────────────────────────────

st.title('DCC-GARCH Live Forecasting System')

c1, c2, c3, c4 = st.columns(4)
c1.metric('Last update', latest_date or 'N/A')
c2.metric('Model', model)

if not eval_hist.empty:
    last_qlike = eval_hist['qlike'].iloc[-1]
    c3.metric(
        'QLIKE (latest)',
        f'{last_qlike:.4f}',
        delta=_qlike_delta_str(eval_hist) if len(eval_hist) > 1 else None,
        delta_color='inverse',
    )
else:
    c3.metric('QLIKE', 'N/A')

if forecast is not None and dcc_p:
    H   = forecast['H']
    sig = forecast['sigma']
    R   = H / np.outer(sig, sig)
    np.fill_diagonal(R, 1.0)
    n   = len(TICKERS)
    avg = (R.sum() - n) / (n * (n - 1))
    c4.metric('Avg pairwise correlation', f'{avg:.4f}')
else:
    c4.metric('Avg pairwise correlation', 'N/A')

st.markdown('---')

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_corr, tab_vol, tab_accuracy, tab_params = st.tabs([
    'Correlations', 'Volatility', 'Forecast Accuracy', 'Parameters'
])


# ── Tab 1: Correlations ───────────────────────────────────────────────────────

with tab_corr:
    if forecast is None:
        st.warning('No forecast available. Run `python live/daily_run.py` to generate one.')
    else:
        H   = forecast['H']
        sig = forecast['sigma']
        R   = H / np.outer(sig, sig)
        np.fill_diagonal(R, 1.0)

        st.subheader(f'Current Correlation Matrix — {latest_date}')
        st.plotly_chart(_corr_heatmap(R), use_container_width=True)

        st.subheader('Average Pairwise Correlation — History')
        st.caption('Spikes indicate market stress periods where sector correlations converge.')
        st.plotly_chart(_avg_corr_history_chart(model), use_container_width=True)

        st.subheader('Pair Correlation History')
        pair_options = [
            f'{TICKERS[i]}-{TICKERS[j]}'
            for i in range(len(TICKERS))
            for j in range(i + 1, len(TICKERS))
        ]
        selected_pair = st.selectbox('Select pair', pair_options, index=0)
        st.plotly_chart(_pair_history_chart(selected_pair, model), use_container_width=True)


# ── Tab 2: Volatility ─────────────────────────────────────────────────────────

with tab_vol:
    if forecast is None:
        st.warning('No forecast available.')
    else:
        sig        = forecast['sigma']
        sig_annual = sig * np.sqrt(252)

        st.subheader(f'Conditional Volatility — {latest_date}')
        fig = go.Figure(go.Bar(
            x=TICKERS,
            y=sig_annual,
            marker_color=px.colors.qualitative.Plotly[:len(TICKERS)],
            text=[f'{v:.1f}%' for v in sig_annual],
            textposition='outside',
        ))
        fig.update_layout(
            yaxis_title='Annualized Volatility (%)',
            height=350,
            margin=dict(t=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Equal-Weight Portfolio Volatility — History')
        st.caption('Annualized vol of a 1/N portfolio across all 10 sector ETFs.')
        st.plotly_chart(_portfolio_vol_chart(model), use_container_width=True)

        st.subheader('Per-Asset Volatility History')
        sel = st.multiselect('Assets to plot', TICKERS, default=['SPY', 'XLE', 'XLF'])
        if sel:
            st.plotly_chart(_sigma_history_chart(sel, model), use_container_width=True)

        st.subheader('Forecast σ vs |Realized Return|')
        sel_ticker = st.selectbox('Asset', TICKERS, index=0)
        st.caption('Red = model forecast σ. Blue = realized |return|. When blue spikes above red, the model underestimated risk.')
        st.plotly_chart(_forecast_vs_realized_chart(sel_ticker, model), use_container_width=True)


# ── Tab 3: Forecast Accuracy ──────────────────────────────────────────────────

with tab_accuracy:
    st.subheader('QLIKE Forecast Accuracy')

    with st.expander('What is QLIKE?', expanded=False):
        st.markdown("""
**QLIKE** = log|H_t| + r_t′ H_t⁻¹ r_t

It measures how well yesterday's covariance forecast H_t predicted today's realized returns r_t.

**Lower = better.**

| Condition | Interpretation |
|-----------|---------------|
| QLIKE falling / stable | Model is tracking risk well |
| QLIKE spike | Model underestimated risk that day (returns were large relative to forecast) |
| Persistent rise | Regime change — model may need refit |

**Reference levels (10-asset portfolio, % daily units):**
- Calm markets: ~10–15
- Moderate volatility: ~15–25
- Stress / crash days: 40+

The 60-day rolling mean is the most informative signal. A single spike is normal.
A rising rolling mean over weeks suggests the model is losing calibration.
        """)

    if eval_hist.empty:
        st.info(
            'No evaluations yet. QLIKE is computed the next trading day '
            'after each forecast is made.'
        )
    else:
        st.plotly_chart(_qlike_chart(eval_hist), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric('Mean QLIKE (all)', f"{eval_hist['qlike'].mean():.4f}")
        c2.metric('Mean QLIKE (last 60d)', f"{eval_hist['qlike'].tail(60).mean():.4f}")
        c3.metric('N evaluations', len(eval_hist))

        with st.expander('Full QLIKE history'):
            st.dataframe(
                eval_hist.rename(columns={'eval_date': 'Date', 'qlike': 'QLIKE'})
                         .set_index('Date')
                         .sort_index(ascending=False),
                use_container_width=True,
            )


# ── Tab 4: Parameters ─────────────────────────────────────────────────────────

with tab_params:
    st.subheader('DCC Parameters')
    if dcc_p:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('a (ARCH)', f"{dcc_p['a']:.6f}")
        c2.metric('b (GARCH)', f"{dcc_p['b']:.6f}")
        c3.metric('a + b', f"{dcc_p['a'] + dcc_p['b']:.6f}")
        if dcc_p.get('g') is not None:
            c4.metric('g (asymmetry)', f"{dcc_p['g']:.6f}")
        st.caption(f"Last fitted: {dcc_p['fitted_date']}")
    else:
        st.info('No DCC params found.')

    st.subheader('GJR-GARCH Parameters')
    garch_p = load_garch_params(TICKERS, DB_PATH)
    if garch_p:
        rows = []
        for ticker, p in zip(TICKERS, garch_p):
            rows.append({
                'Ticker':       ticker,
                'omega':        round(float(p['omega']), 6),
                'alpha':        round(float(p['alpha[1]']), 6),
                'gamma':        round(float(p['gamma[1]']), 6),
                'beta':         round(float(p['beta[1]']), 6),
                'alpha+beta':   round(float(p['alpha[1]']) + float(p['beta[1]']), 4),
                'nu (df)':      round(float(p['nu']), 3),
            })
        st.dataframe(
            pd.DataFrame(rows).set_index('Ticker'),
            use_container_width=True,
        )
    else:
        st.info('No GARCH params found.')

    st.subheader('Re-estimation')
    st.markdown(
        'To re-fit all parameters on a rolling 5-year window (recommended monthly):\n\n'
        '```bash\npython live/daily_run.py --refit --model DCC\n```'
    )
