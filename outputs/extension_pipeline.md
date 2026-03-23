# DCC-GARCH Live System -- Extension Pipeline

Complete specification for a daily forecasting, evaluation, and monitoring system
built on top of the existing `python/dcc` library and the GJR-GARCH univariate layer.

---

## Daily Pipeline (fully specified)

```
1. Download new close prices          yfinance
        |
2. Compute log returns x 100          pandas
        |
3. Run GJR-GARCH filter (fixed params) arch_model.filter(params)
   --> sigma_t (%) and z_t per asset
        |
4. One-step DCC/ADCC update           _update_Q  (python/dcc, microseconds)
   Q_t = AUQ + a*outer(z,z) + b*Q_prev
        |
5. Normalize --> R_t, H_t             compute_R  (python/dcc)
        |
6. Multi-horizon forecasts            analytic (DCC) / Monte Carlo (ADCC)
   h = 1, 5, 21 days
        |
7. Store forecast + compute QLIKE     database (SQLite / PostgreSQL)
        |
8. Streamlit dashboard refreshes      reads from DB
```

---

## Univariate GARCH Layer

**Model:** GJR-GARCH(1,1,1) with Student-t innovations
**Library:** `arch` (`arch_model` with `vol='GARCH', p=1, o=1, q=1, dist='t'`)
**Return scaling:** log returns x 100 (sigmas are in % daily)
**Reference:** `notebooks/Data_Exploration_Univariate_Models.ipynb`

The GJR asymmetric term (`o=1`, parameter `gamma`) captures each asset's own
univariate leverage effect. The ADCC `g` parameter captures an additional multivariate
asymmetry -- correlation reacting to simultaneous negative shocks -- on top of this.
These are two distinct mechanisms at two different levels.

### Setup (run once, store params)

```python
from arch import arch_model

model = arch_model(returns_history * 100, mean='Constant',
                   vol='GARCH', p=1, o=1, q=1, dist='t')
res    = model.fit(disp='off')
params = res.params   # persist to DB: omega, alpha[1], gamma[1], beta[1], nu
```

### Daily update (no optimization, seconds)

```python
model_new = arch_model(returns_updated * 100, mean='Constant',
                       vol='GARCH', p=1, o=1, q=1, dist='t')
filtered  = model_new.filter(params)
sigma_t   = filtered.conditional_volatility.iloc[-1]   # % daily
z_t       = filtered.std_resid.iloc[-1]
```

---

## DCC/ADCC State Update

```python
from python.dcc.model import _update_Q
from python.dcc.utils import make_N_matrix

# Load from DB
Q_prev, a, b, g, Q_bar, N_bar = db.load_dcc_state()

# Asymmetric innovation (ADCC only)
n_t  = z_t * (z_t < 0)

# Pre-compute intercept
AUQ  = (1 - a - b) * Q_bar - g * N_bar   # ADCC
# AUQ = (1 - a - b) * Q_bar              # DCC

# One-step update (microseconds)
Q_t  = _update_Q(Q_prev, z_t, n_t, AUQ, a, b, g, model='ADCC')

# Normalize
sqrt_diag = np.sqrt(np.diag(Q_t))
R_t  = Q_t / np.outer(sqrt_diag, sqrt_diag)
H_t  = sigma_t[:, None] * R_t * sigma_t[None, :]

# Persist state for tomorrow
db.save_dcc_state(Q_t, a, b, g, Q_bar, N_bar)
db.save_forecast(date=today, H=H_t, horizon=1, model='DCC')
```

---

## Multi-Horizon Forecasts

### DCC -- analytic (exact, all horizons)

```python
# h-step ahead: mean reversion to Q_bar at geometric rate (a+b)^h
# Half-life = log(0.5) / log(a+b) ~ 157 trading days for this dataset

for h in [1, 5, 21]:
    Q_h      = Q_bar + (a + b)**h * (Q_t - Q_bar)
    sqrt_d   = np.sqrt(np.diag(Q_h))
    R_h      = Q_h / np.outer(sqrt_d, sqrt_d)
    sigma_h  = garch_forecast_h(params, h=h)    # arch h-step vol forecast
    H_h      = sigma_h[:, None] * R_h * sigma_h[None, :]
    db.save_forecast(date=today, H=H_h, horizon=h, model='DCC')
```

### ADCC -- Monte Carlo simulation (h > 1)

```python
# Required because asymmetric indicator 1[z<0] is not analytically tractable for h>1
M = 1000

for h in [5, 21]:
    paths = []
    for _ in range(M):
        Q_sim = Q_t.copy()
        for step in range(h):
            sqrt_d   = np.sqrt(np.diag(Q_sim))
            R_sim    = Q_sim / np.outer(sqrt_d, sqrt_d)
            L        = np.linalg.cholesky(R_sim)
            z_sim    = L @ np.random.randn(N)
            n_sim    = z_sim * (z_sim < 0)
            AUQ_sim  = (1 - a - b) * Q_bar - g * N_bar
            Q_sim    = _update_Q(Q_sim, z_sim, n_sim, AUQ_sim, a, b, g, 'ADCC')
        paths.append(Q_sim)
    Q_h_adcc = np.mean(paths, axis=0)
    # normalize and store as above
```

---

## Parameter Re-estimation

Parameters are stable day-to-day (identified from thousands of observations).
Re-estimation should use a **rolling window** to allow slow adaptation to regime changes.

```python
# Monthly, rolling 5-year window (1260 trading days)
Z_window      = db.load_Z(last_n_days=1260)
sigmas_window = db.load_sigmas(last_n_days=1260)

# GARCH: re-fit per asset
for asset in assets:
    ret_window = db.load_returns(asset, last_n_days=1260)
    res = arch_model(ret_window * 100, mean='Constant',
                     vol='GARCH', p=1, o=1, q=1, dist='t').fit(disp='off')
    db.save_garch_params(asset, res.params)

# DCC/ADCC: re-fit
from python.dcc import fit
result = fit(Z_window, sigmas_window, model='ADCC')
db.save_dcc_params(result['params'], result['delta'])

# Re-initialize filter from scratch with new params
```

Rolling window ensures g adapts downward when crisis episodes roll out of the window --
the key fix for the OOS over-fitting identified in evaluation.

---

## Evaluation (daily, automatic)

```python
# Each day: compare yesterday's h=1 forecast to today's realized r_t r_t'
H_forecast = db.load_forecast(date=yesterday, horizon=1, model='DCC')
r_today    = db.load_returns(date=today)

# QLIKE loss
L      = np.linalg.cholesky(H_forecast)
logdet = 2 * np.sum(np.log(np.diag(L)))
quad   = r_today @ np.linalg.solve(H_forecast, r_today)
qlike  = logdet + quad

db.save_eval(date=today, model='DCC', horizon=1, qlike=qlike)
# Repeat for ADCC, h=5, h=21
```

Diebold-Mariano test runs on the expanding window of stored eval records.

---

## Technology Stack

| Component          | Tool                          | Notes |
|--------------------|-------------------------------|-------|
| Price download     | `yfinance`                    | Free; covers ETFs, equities |
| Data storage       | `SQLite` (local) / `PostgreSQL` (cloud) | |
| GARCH filter       | `arch` library                | Already used in notebook |
| DCC/ADCC filter    | `python/dcc` (this library)   | Already built |
| Scheduling         | `APScheduler` or system cron  | Trigger at ~4:30pm ET daily |
| Multi-horizon DCC  | Analytic formula              | One line per horizon |
| Multi-horizon ADCC | Monte Carlo (~1s for M=1000)  | |
| Dashboard          | `Streamlit` + `plotly`        | Pure Python, browser UI |
| Deployment         | Local → Docker → cloud VM     | Start local |

---

## Dashboard Pages (Streamlit)

| Page               | Content |
|--------------------|---------|
| Overview           | Current correlation heatmap, avg correlation gauge, regime indicator |
| Pair drill-down    | Select pair -> R_t time series + forecast bands for h=1,5,21 |
| Volatility         | GJR-GARCH sigma per asset, annualized, time series |
| Forecast accuracy  | Rolling QLIKE DCC vs ADCC, cumulative advantage, DM statistic live |
| VaR                | Input portfolio weights -> 1-day VaR and CVaR from H_t |
| Parameters         | Time series of (a, b, g) over re-estimation history |

---

## Build Order

1. Data layer: `yfinance` fetch + DB schema + daily price store       (~1 day)
2. GARCH daily layer: `arch` filter mode, per-asset state persistence  (~1 day)
3. DCC state update: wire `_update_Q` to DB                            (~0.5 day)
4. Multi-horizon DCC: analytic formula, store forecasts                (~0.5 day)
5. Multi-horizon ADCC: Monte Carlo loop                                (~1 day)
6. Evaluation engine: QLIKE per day per model per horizon              (~1 day)
7. Streamlit dashboard: heatmap + pair view + QLIKE chart              (~2-3 days)
8. Scheduler + glue code + logging                                     (~1 day)
                                                              Total:  ~8-10 days

---

## What is Already Built

- `python/dcc/utils.py`     -- Q_bar, N_bar, delta estimation
- `python/dcc/model.py`     -- _update_Q, compute_Q, compute_R, objective, loglikelihood
- `python/dcc/optimizer.py` -- fit(), constraints
- `project/run_dcc_analysis.py`   -- full in-sample analysis + figures
- `project/run_oos_evaluation.py` -- OOS QLIKE evaluation + DM test

The model core is complete. The live system is an operational wrapper around it.
