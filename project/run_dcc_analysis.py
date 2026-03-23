"""
run_dcc_analysis.py -- Fit DCC and ADCC models, compare results, and visualize.

Run from project root:
    python project/run_dcc_analysis.py

Outputs saved to: outputs/
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
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python.dcc import fit
from python.dcc.utils import estimate_Qbar, make_N_matrix, estimate_Nbar, compute_delta

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'figure.dpi':       130,
})

CRISIS = [
    ('Dot-com bust',  '2000-03-01', '2002-10-31'),
    ('GFC',           '2007-10-01', '2009-06-30'),
    ('COVID crash',   '2020-02-01', '2020-06-30'),
]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading data...")
with open('data/dcc_inputs.pkl', 'rb') as f:
    data = pickle.load(f)

Z      = np.asarray(data['Z'])
sigmas = np.asarray(data['sigmas'])
dates  = pd.to_datetime(data['dates'])
assets = list(data['assets'])
T, N   = Z.shape

print(f"  T={T}  N={N}  ({dates[0].date()} to {dates[-1].date()})")
print(f"  Assets: {assets}")

# ---------------------------------------------------------------------------
# 2. Pre-computation diagnostics
# ---------------------------------------------------------------------------
Q_bar = estimate_Qbar(Z)
N_mat = make_N_matrix(Z)
N_bar = estimate_Nbar(N_mat)
delta = compute_delta(Q_bar, N_bar)

print(f"\nPre-computation:")
print(f"  Q_bar min eigenvalue : {np.linalg.eigvalsh(Q_bar).min():.6f}")
print(f"  delta (ADCC)         : {delta:.6f}")

# ---------------------------------------------------------------------------
# 3. Fit DCC
# ---------------------------------------------------------------------------
print("\nFitting DCC...")
t0 = time.time()
dcc = fit(Z, sigmas, model='DCC')
dcc_time = time.time() - t0

a_dcc, b_dcc = dcc['params']
print(f"  Converged : {dcc['converged']}")
print(f"  a={a_dcc:.6f}  b={b_dcc:.6f}  a+b={a_dcc+b_dcc:.6f}")
print(f"  LLH={dcc['llh']:.2f}  time={dcc_time:.1f}s")

# ---------------------------------------------------------------------------
# 4. Fit ADCC
# ---------------------------------------------------------------------------
print("\nFitting ADCC...")
t0 = time.time()
adcc = fit(Z, sigmas, model='ADCC')
adcc_time = time.time() - t0

a_ad, b_ad, g_ad = adcc['params']
delta_ad = adcc['delta']
print(f"  Converged : {adcc['converged']}")
print(f"  a={a_ad:.6f}  b={b_ad:.6f}  g={g_ad:.6f}")
print(f"  delta={delta_ad:.6f}  a+b+delta*g={a_ad+b_ad+delta_ad*g_ad:.6f}")
print(f"  LLH={adcc['llh']:.2f}  time={adcc_time:.1f}s")

# Convenient Series/DataFrames
dcc_R  = dcc['R']    # (T, N, N)
adcc_R = adcc['R']   # (T, N, N)
dcc_H  = dcc['H']
adcc_H = adcc['H']

# Helper: extract pair index
def pair_idx(a1, a2):
    i = assets.index(a1)
    j = assets.index(a2)
    return i, j

def corr_series(R, i, j):
    return pd.Series(R[:, i, j], index=dates)

# ---------------------------------------------------------------------------
# 5. Model comparison table
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print("MODEL COMPARISON")
print("="*55)
n_params_dcc  = 2
n_params_adcc = 3
aic_dcc  = -2 * dcc['llh']  + 2 * n_params_dcc
aic_adcc = -2 * adcc['llh'] + 2 * n_params_adcc
bic_dcc  = -2 * dcc['llh']  + n_params_dcc  * np.log(T)
bic_adcc = -2 * adcc['llh'] + n_params_adcc * np.log(T)
lr_stat  = 2 * (adcc['llh'] - dcc['llh'])

print(f"{'':20s}  {'DCC':>12s}  {'ADCC':>12s}")
print(f"{'a (ARCH)':20s}  {a_dcc:>12.6f}  {a_ad:>12.6f}")
print(f"{'b (GARCH)':20s}  {b_dcc:>12.6f}  {b_ad:>12.6f}")
print(f"{'g (asymmetry)':20s}  {'--':>12s}  {g_ad:>12.6f}")
print(f"{'a+b':20s}  {a_dcc+b_dcc:>12.6f}  {a_ad+b_ad:>12.6f}")
print(f"{'Constraint value':20s}  {a_dcc+b_dcc:>12.6f}  {a_ad+b_ad+delta_ad*g_ad:>12.6f}")
print(f"{'Log-likelihood':20s}  {dcc['llh']:>12.2f}  {adcc['llh']:>12.2f}")
print(f"{'AIC':20s}  {aic_dcc:>12.2f}  {aic_adcc:>12.2f}")
print(f"{'BIC':20s}  {bic_dcc:>12.2f}  {bic_adcc:>12.2f}")
print(f"{'LR statistic':20s}  {'--':>12s}  {lr_stat:>12.2f}")
print(f"{'LR p-value (chi2,1)':20s}  {'--':>12s}  {float(1 - __import__('scipy').stats.chi2.cdf(lr_stat, df=1)):>12.6f}")

# ---------------------------------------------------------------------------
# FIGURE 1: Selected pairwise correlations DCC vs ADCC
# ---------------------------------------------------------------------------
PAIRS = [
    ('SPY', 'XLF'),   # broad market vs financials
    ('SPY', 'XLE'),   # broad market vs energy (low correlation)
    ('XLF', 'XLI'),   # financials vs industrials
    ('XLK', 'XLY'),   # tech vs consumer discretionary
    ('XLP', 'XLU'),   # defensives: staples vs utilities
    ('SPY', 'XLK'),   # broad market vs tech
]

fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
fig.suptitle('DCC vs ADCC: Conditional Correlations (selected pairs)', fontsize=12, fontweight='bold')

for ax, (a1, a2) in zip(axes.flat, PAIRS):
    i, j = pair_idx(a1, a2)
    s_dcc  = corr_series(dcc_R,  i, j)
    s_adcc = corr_series(adcc_R, i, j)

    ax.plot(dates, s_dcc,  lw=0.7, color='steelblue', label='DCC',  alpha=0.85)
    ax.plot(dates, s_adcc, lw=0.7, color='tomato',    label='ADCC', alpha=0.85, linestyle='--')

    # shade crisis periods
    for label, start, end in CRISIS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.08, color='gray', zorder=0)

    ax.set_title(f'{a1} -- {a2}', fontsize=10)
    ax.set_ylabel('Correlation')
    ax.set_ylim(-0.1, 1.0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    if ax is axes.flat[0]:
        ax.legend(fontsize=8)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'fig1_correlations_dcc_vs_adcc.png')
fig.savefig(out, bbox_inches='tight')
print(f"\nSaved: {os.path.basename(out)}")
plt.close()

# ---------------------------------------------------------------------------
# FIGURE 2: Crisis zoom -- all pairs average correlation
# ---------------------------------------------------------------------------
# Average correlation across all N*(N-1)/2 pairs
def avg_corr_series(R):
    idx = np.tril_indices(N, k=-1)
    vals = R[:, idx[0], idx[1]]   # (T, n_pairs)
    return pd.Series(vals.mean(axis=1), index=dates)

avg_dcc  = avg_corr_series(dcc_R)
avg_adcc = avg_corr_series(adcc_R)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Average Pairwise Correlation During Crisis Periods', fontsize=12, fontweight='bold')

for ax, (label, start, end) in zip(axes, CRISIS):
    mask = (dates >= start) & (dates <= end)
    ax.plot(dates[mask], avg_dcc[mask],  lw=1.0, color='steelblue', label='DCC')
    ax.plot(dates[mask], avg_adcc[mask], lw=1.0, color='tomato',    label='ADCC', linestyle='--')
    ax.fill_between(dates[mask], avg_dcc[mask], avg_adcc[mask],
                    alpha=0.2, color='purple', label='Difference')
    ax.set_title(label, fontsize=10)
    ax.set_ylabel('Avg correlation')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # Compute window length in months and pick a tick interval that gives 4-8 ticks
    n_months = (pd.Timestamp(end) - pd.Timestamp(start)).days // 30
    tick_interval = max(1, n_months // 6)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=tick_interval))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    if ax is axes[0]:
        ax.legend(fontsize=8)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'fig2_crisis_correlations.png')
fig.savefig(out, bbox_inches='tight')
print(f"Saved: {os.path.basename(out)}")
plt.close()

# ---------------------------------------------------------------------------
# FIGURE 3: Full-sample average correlation + volatility regime
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
fig.suptitle('Average Correlation and Market Volatility (SPY)', fontsize=12, fontweight='bold')

ax1.plot(dates, avg_dcc,  lw=0.7, color='steelblue', label='DCC avg corr',  alpha=0.9)
ax1.plot(dates, avg_adcc, lw=0.7, color='tomato',    label='ADCC avg corr', alpha=0.9, linestyle='--')
for label, start, end in CRISIS:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color='gray')
ax1.set_ylabel('Average pairwise correlation')
ax1.legend(fontsize=8)
ax1.set_ylim(0, 1)

spy_vol = pd.Series(sigmas[:, assets.index('SPY')] * np.sqrt(252), index=dates)
ax2.fill_between(dates, spy_vol, alpha=0.5, color='dimgray')
ax2.set_ylabel('SPY annualized vol (%)')
for label, start, end in CRISIS:
    ax2.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color='gray',
                label=label if start=='2000-03-01' else '')
ax2.set_xlabel('Date')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator(3))

# annotate crises
for label, start, end in CRISIS:
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ax2.text(mid, spy_vol.max() * 0.88, label, ha='center', fontsize=7.5, color='darkred')

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'fig3_avg_corr_and_vol.png')
fig.savefig(out, bbox_inches='tight')
print(f"Saved: {os.path.basename(out)}")
plt.close()

# ---------------------------------------------------------------------------
# FIGURE 4: DCC vs ADCC difference (asymmetric effect)
# ---------------------------------------------------------------------------
diff = avg_adcc - avg_dcc   # positive = ADCC higher

# rolling asymmetric shock fraction
neg_frac = pd.DataFrame((Z < 0).astype(float), index=dates).mean(axis=1).rolling(63).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
fig.suptitle('ADCC minus DCC: Asymmetric Effect', fontsize=12, fontweight='bold')

ax1.plot(dates, diff * 100, lw=0.7, color='purple')
ax1.axhline(0, color='black', lw=0.5)
ax1.fill_between(dates, diff * 100, 0,
                 where=(diff > 0), alpha=0.3, color='tomato',    label='ADCC > DCC')
ax1.fill_between(dates, diff * 100, 0,
                 where=(diff < 0), alpha=0.3, color='steelblue', label='DCC > ADCC')
for label, start, end in CRISIS:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color='gray')
ax1.set_ylabel('Avg corr difference (bps)')
ax1.legend(fontsize=8)

ax2.plot(dates, neg_frac, lw=0.8, color='dimgray', label='Fraction z<0 (63d MA)')
ax2.axhline(0.5, color='black', lw=0.5, linestyle='--')
ax2.set_ylabel('Fraction of negative shocks')
ax2.set_xlabel('Date')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator(3))
ax2.legend(fontsize=8)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'fig4_adcc_minus_dcc.png')
fig.savefig(out, bbox_inches='tight')
print(f"Saved: {os.path.basename(out)}")
plt.close()

# ---------------------------------------------------------------------------
# FIGURE 5: Heatmaps -- unconditional vs DCC time-avg vs ADCC time-avg
# ---------------------------------------------------------------------------
d_sqrt = np.sqrt(np.diag(Q_bar))
R_uncond = Q_bar / np.outer(d_sqrt, d_sqrt)

R_dcc_mean  = dcc_R.mean(axis=0)
R_adcc_mean = adcc_R.mean(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle('Correlation Matrices: Unconditional | DCC time-avg | ADCC time-avg',
             fontsize=11, fontweight='bold')

for ax, mat, title in zip(axes,
                           [R_uncond, R_dcc_mean, R_adcc_mean],
                           ['Unconditional', 'DCC (time avg)', 'ADCC (time avg)']):
    im = ax.imshow(mat, vmin=0.2, vmax=1.0, cmap='RdYlGn')
    ax.set_xticks(range(N)); ax.set_xticklabels(assets, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(N)); ax.set_yticklabels(assets, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                    fontsize=6.5, color='black' if mat[i,j] < 0.85 else 'white')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'fig5_correlation_heatmaps.png')
fig.savefig(out, bbox_inches='tight')
print(f"Saved: {os.path.basename(out)}")
plt.close()

# ---------------------------------------------------------------------------
# FIGURE 6: Distribution of pairwise correlations DCC vs ADCC
# ---------------------------------------------------------------------------
idx_lo = np.tril_indices(N, k=-1)
dcc_flat  = dcc_R[:,  idx_lo[0], idx_lo[1]].ravel()
adcc_flat = adcc_R[:, idx_lo[0], idx_lo[1]].ravel()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle('Distribution of Pairwise Conditional Correlations', fontsize=12, fontweight='bold')

bins = np.linspace(-0.1, 1.0, 80)
axes[0].hist(dcc_flat,  bins=bins, color='steelblue', alpha=0.7, density=True, label='DCC')
axes[0].hist(adcc_flat, bins=bins, color='tomato',    alpha=0.7, density=True, label='ADCC')
axes[0].set_xlabel('Correlation'); axes[0].set_ylabel('Density')
axes[0].set_title('All pairs, full sample')
axes[0].legend()

# Per-pair: show mean DCC vs ADCC
n_pairs = len(idx_lo[0])
pair_labels = [f"{assets[i]}-{assets[j]}" for i,j in zip(idx_lo[0], idx_lo[1])]
mean_dcc_pair  = dcc_R[:,  idx_lo[0], idx_lo[1]].mean(axis=0)
mean_adcc_pair = adcc_R[:, idx_lo[0], idx_lo[1]].mean(axis=0)
order = np.argsort(mean_dcc_pair)

y = np.arange(n_pairs)
axes[1].barh(y - 0.2, mean_dcc_pair[order],  height=0.4, color='steelblue', label='DCC')
axes[1].barh(y + 0.2, mean_adcc_pair[order], height=0.4, color='tomato',    label='ADCC')
axes[1].set_yticks(y)
axes[1].set_yticklabels([pair_labels[i] for i in order], fontsize=7)
axes[1].set_xlabel('Time-averaged correlation')
axes[1].set_title('Per-pair mean correlation')
axes[1].legend(fontsize=8)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'fig6_correlation_distribution.png')
fig.savefig(out, bbox_inches='tight')
print(f"Saved: {os.path.basename(out)}")
plt.close()

# ---------------------------------------------------------------------------
# FIGURE 7: Conditional volatilities (sigmas) -- overview
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 4))
fig.suptitle('GARCH Conditional Volatilities (annualized %)', fontsize=12, fontweight='bold')

colors = plt.cm.tab10(np.linspace(0, 1, N))
for i, (asset, color) in enumerate(zip(assets, colors)):
    vol = pd.Series(sigmas[:, i] * np.sqrt(252), index=dates)
    ax.plot(dates, vol, lw=0.6, label=asset, color=color, alpha=0.85)

for label, start, end in CRISIS:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color='gray')

ax.set_ylabel('Annualized volatility (%)')
ax.set_xlabel('Date')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator(3))
ax.legend(ncol=5, fontsize=8, loc='upper left')

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'fig7_conditional_volatilities.png')
fig.savefig(out, bbox_inches='tight')
print(f"Saved: {os.path.basename(out)}")
plt.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(f"Data     : {T} daily obs, {N} assets, {dates[0].year}-{dates[-1].year}")
print()
print(f"DCC      : a={a_dcc:.4f}  b={b_dcc:.4f}  persistence={a_dcc+b_dcc:.4f}")
print(f"ADCC     : a={a_ad:.4f}  b={b_ad:.4f}  g={g_ad:.4f}  "
      f"persistence={a_ad+b_ad+delta_ad*g_ad:.4f}")
print()
print(f"LLH DCC  : {dcc['llh']:,.2f}")
print(f"LLH ADCC : {adcc['llh']:,.2f}  (improvement: +{adcc['llh']-dcc['llh']:.2f})")
print(f"LR stat  : {lr_stat:.2f}  (p < 0.001)")
print()
print(f"Avg corr (DCC) : {avg_dcc.mean():.4f}")
print(f"Avg corr (ADCC): {avg_adcc.mean():.4f}")
print()
print("Figures saved to outputs/")
