"""
run_oos_evaluation.py -- Out-of-sample forecast evaluation for DCC and ADCC.

Design:
  - Train: first 80% of sample (parameters and Q_bar estimated here only)
  - Test:  last 20% of sample  (COVID crash and post-COVID regimes)
  - Forecast: 1-step-ahead H_{t|t-1} using training parameters, filtered on full sample
  - Loss functions: QLIKE and MSE (Frobenius) vs rank-1 realized covariance proxy r_t r_t'
  - Test: Diebold-Mariano for equal predictive accuracy

Run from project root:
    python project/run_oos_evaluation.py
"""

import sys
import os
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python.dcc import fit
from python.dcc.model import compute_Q, compute_R
from python.dcc.utils import (
    estimate_Qbar, validate_Qbar,
    make_N_matrix, estimate_Nbar, compute_delta
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'figure.dpi':       130,
})

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
with open('data/dcc_inputs.pkl', 'rb') as f:
    data = pickle.load(f)

Z       = np.asarray(data['Z'])
sigmas  = np.asarray(data['sigmas'])
returns = np.asarray(data['returns'])
dates   = pd.to_datetime(data['dates'])
assets  = list(data['assets'])
T, N    = Z.shape

# ---------------------------------------------------------------------------
# 2. Train / test split (80 / 20)
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.80
T_train = int(T * TRAIN_FRAC)
T_test  = T - T_train

dates_train = dates[:T_train]
dates_test  = dates[T_train:]

print(f"\nSplit:")
print(f"  Train : {T_train} obs  ({dates_train[0].date()} -- {dates_train[-1].date()})")
print(f"  Test  : {T_test}  obs  ({dates_test[0].date()}  -- {dates_test[-1].date()})")
print(f"  (Test covers: {dates_test[0].year} -- {dates_test[-1].year})")

Z_train      = Z[:T_train]
sigmas_train = sigmas[:T_train]

# ---------------------------------------------------------------------------
# 3. Estimate parameters on training data only (no look-ahead)
# ---------------------------------------------------------------------------
print("\nFitting DCC on training data...")
t0 = time.time()
dcc_train = fit(Z_train, sigmas_train, model='DCC')
print(f"  Converged: {dcc_train['converged']}  "
      f"a={dcc_train['params'][0]:.6f}  b={dcc_train['params'][1]:.6f}  "
      f"time={time.time()-t0:.1f}s")

print("Fitting ADCC on training data...")
t0 = time.time()
adcc_train = fit(Z_train, sigmas_train, model='ADCC')
a_ad, b_ad, g_ad = adcc_train['params']
delta_train = adcc_train['delta']
print(f"  Converged: {adcc_train['converged']}  "
      f"a={a_ad:.6f}  b={b_ad:.6f}  g={g_ad:.6f}  "
      f"time={time.time()-t0:.1f}s")

# Training-set Q_bar and N_bar (no look-ahead)
Q_bar_train = estimate_Qbar(Z_train)
N_mat_train = make_N_matrix(Z_train)
N_bar_train = estimate_Nbar(N_mat_train)

# ---------------------------------------------------------------------------
# 4. Filter on full sample using training parameters
#    Q_bar is set to training-sample Q_bar throughout (no look-ahead)
# ---------------------------------------------------------------------------
print("\nFiltering on full sample with fixed training parameters...")

N_mat_full = make_N_matrix(Z)

# DCC full filter
Q_dcc_full = compute_Q(
    Z, None,
    dcc_train['params'],
    Q_bar_train, None,
    model='DCC'
)
R_dcc_full = compute_R(Q_dcc_full)
H_dcc_full = sigmas[:, :, np.newaxis] * R_dcc_full * sigmas[:, np.newaxis, :]

# ADCC full filter
Q_adcc_full = compute_Q(
    Z, N_mat_full,
    adcc_train['params'],
    Q_bar_train, N_bar_train,
    model='ADCC'
)
R_adcc_full = compute_R(Q_adcc_full)
H_adcc_full = sigmas[:, :, np.newaxis] * R_adcc_full * sigmas[:, np.newaxis, :]

# ---------------------------------------------------------------------------
# 5. Loss functions
#    Realized covariance proxy: Sigma_realized_t = r_t r_t'  (rank-1, unbiased)
#
#    QLIKE: L_t = log|H_t| + r_t' H_t^{-1} r_t
#      (trace(H_t^{-1} r_t r_t') = r_t' H_t^{-1} r_t for rank-1 proxy)
#      Lower is better.
#
#    MSE (Frobenius): L_t = ||H_t - r_t r_t'||_F^2
#      Lower is better.
# ---------------------------------------------------------------------------

def compute_losses(H, ret, idx_start, idx_end):
    """
    Compute QLIKE and MSE losses on the test period.

    H   : (T, N, N) full-sample forecast path
    ret : (T, N)    return vectors
    Returns arrays of length (idx_end - idx_start).
    """
    qlike = np.empty(idx_end - idx_start)
    mse   = np.empty(idx_end - idx_start)

    for k, t in enumerate(range(idx_start, idx_end)):
        H_t = H[t]
        r_t = ret[t]

        # log|H_t| via Cholesky
        try:
            L_chol = np.linalg.cholesky(H_t)
            log_det = 2.0 * np.sum(np.log(np.diag(L_chol)))
            # r_t' H_t^{-1} r_t via forward substitution
            y = np.linalg.solve(L_chol, r_t)
            quad = y @ y
        except np.linalg.LinAlgError:
            log_det = np.nan
            quad    = np.nan

        qlike[k] = log_det + quad

        # MSE: ||H_t - r_t r_t'||_F^2
        diff = H_t - np.outer(r_t, r_t)
        mse[k] = np.sum(diff ** 2)

    return qlike, mse


print("Computing losses on test period...")
qlike_dcc,  mse_dcc  = compute_losses(H_dcc_full,  returns, T_train, T)
qlike_adcc, mse_adcc = compute_losses(H_adcc_full, returns, T_train, T)

# Remove NaNs (rare Cholesky failures)
valid = np.isfinite(qlike_dcc) & np.isfinite(qlike_adcc)
qlike_dcc  = qlike_dcc[valid]
qlike_adcc = qlike_adcc[valid]
mse_dcc    = mse_dcc[valid]
mse_adcc   = mse_adcc[valid]
dates_oos  = dates_test[valid]

print(f"  Valid test observations: {valid.sum()} / {T_test}")

# ---------------------------------------------------------------------------
# 6. Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("OUT-OF-SAMPLE FORECAST EVALUATION (h=1)")
print("="*60)
print(f"{'':25s}  {'DCC':>12s}  {'ADCC':>12s}  {'Better':>8s}")
print(f"{'Mean QLIKE':25s}  {qlike_dcc.mean():>12.4f}  {qlike_adcc.mean():>12.4f}  "
      f"{'ADCC' if qlike_adcc.mean() < qlike_dcc.mean() else 'DCC':>8s}")
print(f"{'Mean MSE':25s}  {mse_dcc.mean():>12.2f}  {mse_adcc.mean():>12.2f}  "
      f"{'ADCC' if mse_adcc.mean() < mse_dcc.mean() else 'DCC':>8s}")
print(f"{'QLIKE improvement (%)':25s}  {'--':>12s}  "
      f"{100*(qlike_dcc.mean()-qlike_adcc.mean())/abs(qlike_dcc.mean()):>11.4f}%  {'':>8s}")

# ---------------------------------------------------------------------------
# 7. Diebold-Mariano test
#    H0: equal predictive accuracy (E[d_t] = 0)
#    d_t = L_DCC_t - L_ADCC_t
#    Positive d_t means DCC is worse; negative means ADCC is worse.
#    DM statistic with HAC variance (Newey-West, bandwidth = h-1 = 0 for h=1)
# ---------------------------------------------------------------------------

def diebold_mariano(loss_a, loss_b, h=1):
    """
    Two-sided Diebold-Mariano test.
    d_t = loss_a_t - loss_b_t
    H0: E[d_t] = 0  (equal forecast accuracy)
    Uses HAC variance with bandwidth = h-1.
    Returns: dm_stat, p_value, mean_d
    """
    d = loss_a - loss_b
    T = len(d)
    mean_d = d.mean()

    # HAC variance (Bartlett kernel, bandwidth = h-1, for h=1 this is just sample var)
    bw = max(0, h - 1)
    gamma0 = np.var(d, ddof=1)
    hac_var = gamma0
    for lag in range(1, bw + 1):
        gamma_k = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        hac_var += 2 * (1 - lag / (bw + 1)) * gamma_k

    dm_stat = mean_d / np.sqrt(hac_var / T)
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value, mean_d

dm_stat_q, dm_p_q, mean_d_q = diebold_mariano(qlike_dcc, qlike_adcc, h=1)
dm_stat_m, dm_p_m, mean_d_m = diebold_mariano(mse_dcc,   mse_adcc,   h=1)

print(f"\nDiebold-Mariano Test (H0: equal forecast accuracy)")
print(f"  d_t = loss_DCC - loss_ADCC  (positive = DCC worse)")
print(f"  {'':20s}  {'QLIKE':>10s}  {'MSE':>10s}")
print(f"  {'mean(d_t)':20s}  {mean_d_q:>10.6f}  {mean_d_m:>10.2f}")
print(f"  {'DM statistic':20s}  {dm_stat_q:>10.4f}  {dm_stat_m:>10.4f}")
print(f"  {'p-value (two-sided)':20s}  {dm_p_q:>10.6f}  {dm_p_m:>10.6f}")
verdict_q = "ADCC significantly better" if (dm_p_q < 0.05 and mean_d_q > 0) else \
            "DCC significantly better"  if (dm_p_q < 0.05 and mean_d_q < 0) else \
            "No significant difference"
verdict_m = "ADCC significantly better" if (dm_p_m < 0.05 and mean_d_m > 0) else \
            "DCC significantly better"  if (dm_p_m < 0.05 and mean_d_m < 0) else \
            "No significant difference"
print(f"  {'Verdict (QLIKE)':20s}  {verdict_q}")
print(f"  {'Verdict (MSE)':20s}  {verdict_m}")

# ---------------------------------------------------------------------------
# 8. Sub-period breakdown (calm vs stress)
# ---------------------------------------------------------------------------
covid_start = pd.Timestamp('2020-02-01')
covid_end   = pd.Timestamp('2020-12-31')
covid_mask  = (dates_oos >= covid_start) & (dates_oos <= covid_end)
calm_mask   = ~covid_mask

print(f"\nSub-period QLIKE breakdown:")
print(f"  {'Period':20s}  {'N':>6s}  {'DCC mean':>12s}  {'ADCC mean':>12s}  {'ADCC better?':>14s}")
for label, mask in [('COVID crash', covid_mask), ('Calm (non-COVID)', calm_mask)]:
    if mask.sum() == 0:
        continue
    dcc_m  = qlike_dcc[mask].mean()
    adcc_m = qlike_adcc[mask].mean()
    better = "Yes" if adcc_m < dcc_m else "No"
    print(f"  {label:20s}  {mask.sum():>6d}  {dcc_m:>12.4f}  {adcc_m:>12.4f}  {better:>14s}")

# ---------------------------------------------------------------------------
# FIGURE 8: Rolling QLIKE loss (DCC vs ADCC) + difference
# ---------------------------------------------------------------------------
ROLL = 63   # ~3 months

s_qlike_dcc  = pd.Series(qlike_dcc,  index=dates_oos)
s_qlike_adcc = pd.Series(qlike_adcc, index=dates_oos)
s_diff_q     = s_qlike_dcc - s_qlike_adcc   # positive = DCC worse

roll_dcc  = s_qlike_dcc.rolling(ROLL).mean()
roll_adcc = s_qlike_adcc.rolling(ROLL).mean()
roll_diff = s_diff_q.rolling(ROLL).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig.suptitle('Out-of-Sample Forecast Evaluation (h=1): QLIKE Loss', fontsize=12, fontweight='bold')

ax1.plot(dates_oos, roll_dcc,  lw=0.9, color='steelblue', label=f'DCC  ({ROLL}d MA)')
ax1.plot(dates_oos, roll_adcc, lw=0.9, color='tomato',    label=f'ADCC ({ROLL}d MA)', linestyle='--')
ax1.set_ylabel('QLIKE loss (rolling avg)')
ax1.legend(fontsize=9)
ax1.axvspan(covid_start, covid_end, alpha=0.10, color='gray', label='COVID')
ax1.set_title(f'Rolling {ROLL}-day QLIKE -- lower is better', fontsize=10)

ax2.plot(dates_oos, roll_diff, lw=0.9, color='purple', label='DCC - ADCC QLIKE')
ax2.axhline(0, color='black', lw=0.8, linestyle='--')
ax2.fill_between(dates_oos, roll_diff, 0,
                 where=(roll_diff > 0), alpha=0.3, color='tomato',
                 label='ADCC better (DCC - ADCC > 0)')
ax2.fill_between(dates_oos, roll_diff, 0,
                 where=(roll_diff < 0), alpha=0.3, color='steelblue',
                 label='DCC better (DCC - ADCC < 0)')
ax2.axvspan(covid_start, covid_end, alpha=0.10, color='gray')
ax2.set_ylabel('QLIKE difference (DCC - ADCC)')
ax2.set_xlabel('Date')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator(1))
ax2.legend(fontsize=8)

# Annotate DM result
dm_text = f"DM stat={dm_stat_q:.2f}, p={dm_p_q:.4f}\n{verdict_q}"
ax2.text(0.02, 0.05, dm_text, transform=ax2.transAxes,
         fontsize=8.5, verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'fig8_oos_qlike.png')
fig.savefig(out, bbox_inches='tight')
print(f"\nSaved: {os.path.basename(out)}")
plt.close()

# ---------------------------------------------------------------------------
# FIGURE 9: Cumulative QLIKE advantage of ADCC over DCC
# ---------------------------------------------------------------------------
cum_advantage = np.cumsum(qlike_dcc - qlike_adcc)   # positive = ADCC cumulative better

fig, ax = plt.subplots(figsize=(13, 4))
fig.suptitle('Cumulative QLIKE Advantage of ADCC over DCC (OOS)', fontsize=12, fontweight='bold')

ax.plot(dates_oos, cum_advantage, lw=1.0, color='purple')
ax.axhline(0, color='black', lw=0.7, linestyle='--')
ax.fill_between(dates_oos, cum_advantage, 0,
                where=(cum_advantage > 0), alpha=0.25, color='tomato',
                label='ADCC ahead')
ax.fill_between(dates_oos, cum_advantage, 0,
                where=(cum_advantage < 0), alpha=0.25, color='steelblue',
                label='DCC ahead')
ax.axvspan(covid_start, covid_end, alpha=0.10, color='gray', label='COVID crash')
ax.set_ylabel('Cumulative QLIKE(DCC) - QLIKE(ADCC)')
ax.set_xlabel('Date')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.legend(fontsize=9)
ax.set_title('Positive = ADCC cumulatively better', fontsize=10)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'fig9_cumulative_qlike.png')
fig.savefig(out, bbox_inches='tight')
print(f"Saved: {os.path.basename(out)}")
plt.close()

# ---------------------------------------------------------------------------
# FIGURE 10: Forecast vs realized -- scatter for a representative pair
# ---------------------------------------------------------------------------
# Pick SPY-XLF (high correlation, financially interesting)
i_spy = assets.index('SPY')
i_xlf = assets.index('XLF')

# Forecast correlations from DCC and ADCC on test period
r_dcc_pair  = R_dcc_full[T_train:,  i_spy, i_xlf]
r_adcc_pair = R_adcc_full[T_train:, i_spy, i_xlf]

# Realized: sign(r_spy * r_xlf) is a rough measure; use 21-day rolling realized correlation
ret_spy = pd.Series(returns[:, i_spy], index=dates)
ret_xlf = pd.Series(returns[:, i_xlf], index=dates)
roll_real_corr = ret_spy.rolling(21).corr(ret_xlf)
real_oos = roll_real_corr.iloc[T_train:].values

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle('Forecast vs Realized Correlation: SPY -- XLF (OOS)', fontsize=12, fontweight='bold')

# Time series
ax = axes[0]
ax.plot(dates_oos, r_dcc_pair,  lw=0.8, color='steelblue', label='DCC forecast',  alpha=0.85)
ax.plot(dates_oos, r_adcc_pair, lw=0.8, color='tomato',    label='ADCC forecast', alpha=0.85, linestyle='--')
ax.plot(dates_oos, real_oos,    lw=0.8, color='black',      label='Realized (21d rolling)', alpha=0.6)
ax.axvspan(covid_start, covid_end, alpha=0.10, color='gray')
ax.set_ylabel('Correlation')
ax.set_xlabel('Date')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.legend(fontsize=8)
ax.set_title('Forecast vs realized over time', fontsize=10)

# Scatter: DCC forecast vs realized
ax = axes[1]
valid_r = np.isfinite(real_oos)
ax.scatter(r_dcc_pair[valid_r],  real_oos[valid_r],
           s=2, alpha=0.3, color='steelblue', label='DCC')
ax.scatter(r_adcc_pair[valid_r], real_oos[valid_r],
           s=2, alpha=0.3, color='tomato',    label='ADCC')
ax.plot([-0.2, 1.0], [-0.2, 1.0], 'k--', lw=0.8, label='45-degree line')
ax.set_xlabel('Forecast correlation')
ax.set_ylabel('Realized correlation (21d)')
ax.set_title('Forecast accuracy (SPY-XLF)', fontsize=10)
ax.legend(fontsize=8, markerscale=4)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'fig10_forecast_vs_realized.png')
fig.savefig(out, bbox_inches='tight')
print(f"Saved: {os.path.basename(out)}")
plt.close()

# ---------------------------------------------------------------------------
# Print parameter comparison: in-sample vs OOS training
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PARAMETER STABILITY CHECK")
print("="*60)
print(f"  {'':8s}  {'Full sample':>15s}  {'Train only':>15s}  {'Difference':>12s}")

# Run full-sample fit for comparison (reuse from earlier if available, else note)
print(f"  {'DCC a':8s}  {'(run run_dcc_analysis.py)':>15s}  "
      f"{dcc_train['params'][0]:>15.6f}  {'--':>12s}")
print(f"  {'DCC b':8s}  {'(run run_dcc_analysis.py)':>15s}  "
      f"{dcc_train['params'][1]:>15.6f}  {'--':>12s}")
print(f"  {'ADCC a':8s}  {'(run run_dcc_analysis.py)':>15s}  "
      f"{adcc_train['params'][0]:>15.6f}  {'--':>12s}")
print(f"  {'ADCC b':8s}  {'(run run_dcc_analysis.py)':>15s}  "
      f"{adcc_train['params'][1]:>15.6f}  {'--':>12s}")
print(f"  {'ADCC g':8s}  {'(run run_dcc_analysis.py)':>15s}  "
      f"{adcc_train['params'][2]:>15.6f}  {'--':>12s}")

print("\nDone.")
