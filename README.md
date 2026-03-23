# DCC-GARCH Python Library

A mathematically faithful Python implementation of the Dynamic Conditional Correlation (DCC) and Asymmetric DCC (ADCC) GARCH models, reverse-engineered from the R package `rmgarch` and validated against the original academic literature.

## References

- Engle, R. (2002). Dynamic Conditional Correlations: A Simple Class of Multivariate GARCH Models. *Journal of Business & Economic Statistics*, 20(3), 339-350.
- Cappiello, L., Engle, R., & Sheppard, K. (2006). Asymmetric Dynamics in the Correlations of Global Equity and Bond Returns. *Journal of Financial Econometrics*, 4(4), 537-572.

---

## Model

### DCC (Engle 2002)

The model operates in two stages. Stage 1 (univariate GARCH) is assumed pre-completed. This library handles Stage 2 only.

**Standardized residuals:**

```
z_{i,t} = eps_{i,t} / sigma_{i,t}
```

**Q_t recursion:**

```
Q_t = (1 - a - b) * Q_bar  +  a * z_{t-1} z_{t-1}'  +  b * Q_{t-1}
```

**Correlation matrix:**

```
R_t[i,j] = Q_t[i,j] / sqrt(Q_t[i,i] * Q_t[j,j])
```

**Conditional covariance:**

```
H_t = Sigma_t * R_t * Sigma_t
```

where `Sigma_t = diag(sigma_{1,t}, ..., sigma_{N,t})`.

**Log-likelihood (Stage 2 objective):**

```
L = -0.5 * sum_t [ log|R_t| + z_t' R_t^{-1} z_t ]
```

### ADCC (Cappiello et al. 2006)

Adds an asymmetric term capturing the leverage effect in correlations (negative shocks increase correlations more than positive shocks of equal magnitude).

**Asymmetric innovations:**

```
n_t = z_t * 1[z_t < 0]    (element-wise)
```

**Q_t recursion (full Cappiello intercept):**

```
Q_t = (1 - a - b) * Q_bar  -  g * N_bar
    +  a * z_{t-1} z_{t-1}'
    +  g * n_{t-1} n_{t-1}'
    +  b * Q_{t-1}
```

where `N_bar = E[n_t n_t']`, estimated as the uncentered second moment.

**Stationarity constraint:**

```
a + b + delta * g < 1
```

where `delta` = max eigenvalue of `Q_bar^{-1/2} N_bar Q_bar^{-1/2}`.

---

## Project Structure

```
python/dcc/
    __init__.py     -- public API
    utils.py        -- pure pre-computation (Q_bar, N_bar, delta)
    model.py        -- _update_Q, compute_Q, compute_R, objective, loglikelihood
    optimizer.py    -- constraints, fit() entry point
    validate.py     -- five-level validation suite
```

The univariate GARCH layer lives separately in `python/garch/` and is not part of this library.

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy>=1.24`, `scipy>=1.10`. No other dependencies.

---

## Usage

### Minimal

```python
import pickle
import numpy as np
from python.dcc import fit

# Load pre-computed GARCH outputs
with open('data/dcc_inputs.pkl', 'rb') as f:
    data = pickle.load(f)

Z      = data['Z']       # (T, N) standardized residuals
sigmas = data['sigmas']  # (T, N) conditional volatilities

# Fit DCC
result = fit(Z, sigmas, model='DCC')

print(result['params'])     # array([a, b])
print(result['llh'])        # full log-likelihood
print(result['converged'])  # True/False
print(result['R'].shape)    # (T, N, N) -- conditional correlations
print(result['H'].shape)    # (T, N, N) -- conditional covariances
```

### ADCC

```python
result = fit(Z, sigmas, model='ADCC')

a, b, g = result['params']
delta   = result['delta']
print(f"a={a:.4f}  b={b:.4f}  g={g:.4f}  delta={delta:.4f}")
print(f"Stationarity: a+b+delta*g = {a + b + delta*g:.6f}")
```

### Custom starting values

```python
result = fit(Z, sigmas, model='DCC', x0=np.array([0.03, 0.96]))
```

### Return value

`fit()` returns a dictionary:

| Key | Type | Description |
|-----|------|-------------|
| `params` | ndarray | Estimated parameters: `(a, b)` for DCC; `(a, b, g)` for ADCC |
| `Q` | ndarray (T, N, N) | Pseudo-correlation matrix path |
| `R` | ndarray (T, N, N) | Conditional correlation matrix path |
| `H` | ndarray (T, N, N) | Conditional covariance matrix path |
| `llh` | float | Full Gaussian log-likelihood |
| `converged` | bool | Optimizer convergence flag |
| `delta` | float or None | ADCC stationarity scalar; `None` for DCC |

### Individual functions

```python
from python.dcc.utils import estimate_Qbar, make_N_matrix, estimate_Nbar, compute_delta
from python.dcc.model import compute_Q, compute_R, loglikelihood

# Pre-compute
Q_bar = estimate_Qbar(Z)          # (N, N)
N_mat = make_N_matrix(Z)          # (T, N) -- for ADCC
N_bar = estimate_Nbar(N_mat)      # (N, N)
delta = compute_delta(Q_bar, N_bar)

# Post-convergence path reconstruction
params = (0.02, 0.97)
Q_path = compute_Q(Z, None, params, Q_bar, None, model='DCC')   # (T, N, N)
R_path = compute_R(Q_path)                                        # (T, N, N)
```

---

## Inputs

| Variable | Shape | Description |
|----------|-------|-------------|
| `Z` | (T, N) | Standardized residuals from Stage 1 GARCH: `z_{i,t} = eps_{i,t} / sigma_{i,t}` |
| `sigmas` | (T, N) | Conditional volatilities from Stage 1 GARCH (not variances) |

`Q_bar` is always re-estimated internally from `Z` via `np.cov(Z.T, ddof=1)`. The `Q_bar` field in `dcc_inputs.pkl` is ignored.

---

## Implementation Notes

### Deviations from rmgarch (intentional corrections)

| ID | Item | rmgarch | This library |
|----|------|---------|-------------|
| DEV-02/03 | Q_t initialization | Inconsistent (zero-pad vs ones-pad) | Q[0] = Q_bar (clean) |
| DEV-04 | Q_t symmetry | Not enforced | `(Q + Q.T)/2` at each step |
| DEV-05 | R_t^{-1} z_t | LU inversion | Cholesky forward substitution |
| DEV-06 | log\|R_t\| | `log(det())` | `2 * sum(log(diag(L)))` via Cholesky |
| DEV-07 | Cholesky failure | No guard | Returns stateless `PENALTY=1e6` |
| DEV-08 | Constraint enforcement | Soft penalty (nlminb path) | Hard inequality (SLSQP) |

### Deviations from rmgarch (deliberate)

| ID | Item | rmgarch | This library | Reason |
|----|------|---------|-------------|--------|
| DEV-01b | N_bar estimator | `cov()` -- centered | `(N.T @ N) / (T-1)` -- uncentered | Preserves E[Q_t] = Q_bar unconditionally |

### Design invariants

- `_update_Q` is the sole implementation of the Q_t recursion. It is called identically by the optimizer objective and the post-convergence path function. No duplication.
- `_update_Q` takes only `(Q_prev, z_prev, n_prev, AUQ, a, b, g, model)`. It does not access `Q_bar`, `N_bar`, or any global state. `AUQ` is always pre-computed once per call, outside the time loop.
- No in-recursion regularization of Q_t. If Cholesky of R_t fails, the objective returns a stateless constant `PENALTY = 1e6`.

---

## Validation

Run the full five-level validation suite:

```bash
python python/dcc/validate.py
```

Results on real data (T=7099, N=10 US equity sector ETFs):

| Model | a | b | g | Constraint | Log-likelihood |
|-------|---|---|---|-----------|---------------|
| DCC | 0.0209 | 0.9748 | -- | a+b = 0.9956 | -71,756.8 |
| ADCC | 0.0183 | 0.9699 | 0.0138 | a+b+delta*g = 0.9999 | -71,559.6 |

ADCC improves log-likelihood by +197 units, confirming significant asymmetric correlation dynamics in equity sector returns.

---

## Scope

This library implements Normal (Gaussian) QML estimation of DCC(1,1) and ADCC(1,1). The following are explicitly out of scope:

- Univariate GARCH estimation (pre-completed externally)
- Student-t or Laplace distributions
- Multi-lag DCC (order > (1,1))
- VAR mean model
- Standard errors / Hessian
- Forecasting and simulation
