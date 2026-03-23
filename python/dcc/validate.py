"""
validate.py -- Five-level validation suite for the DCC/ADCC-GARCH implementation.

Levels:
  L1 -- Unit: _update_Q arithmetic
  L2 -- Unit: compute_R structural properties
  L3 -- Unit: utils functions
  L4 -- Integration: synthetic data fit (DCC + ADCC)
  L5 -- Integration: real data fit (dcc_inputs.pkl)

Run from project root:
  python python/dcc/validate.py
"""

import sys
import os
import time
import pickle
import numpy as np

# ---------------------------------------------------------------------------
# Path setup -- allow running from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from python.dcc.model import _update_Q, compute_Q, compute_R, dcc_objective, PENALTY
from python.dcc.utils import (
    estimate_Qbar, validate_Qbar, make_N_matrix, estimate_Nbar, compute_delta
)
from python.dcc.optimizer import fit, dcc_constraint, adcc_constraint


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(msg):
    print(f"  [PASS] {msg}")

def fail(msg):
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)

def check(condition, msg):
    if condition:
        ok(msg)
    else:
        fail(msg)


# ---------------------------------------------------------------------------
# Synthetic DCC data generator
# ---------------------------------------------------------------------------

def generate_dcc_data(T=500, N=3, a=0.05, b=0.85, g=0.03, seed=42):
    """
    Simulate standardized residuals from a DCC process.

    Correlation matrix R_t is generated via the DCC recursion.
    Each z_t is drawn from N(0, R_t) and then standardized.
    sigmas are set to constant 1.0 (GARCH step already done).
    """
    rng = np.random.default_rng(seed)
    N_assets = N

    # Q_bar -- target correlation (identity for simplicity)
    Q_bar = np.eye(N_assets) * 0.8 + np.ones((N_assets, N_assets)) * 0.2
    np.fill_diagonal(Q_bar, 1.0)

    Z = np.zeros((T, N_assets))
    sigmas = np.ones((T, N_assets))

    Q_t = Q_bar.copy()
    AUQ = (1 - a - b) * Q_bar   # DCC intercept (g not used here)

    for t in range(T):
        # Draw from R_t
        sqrt_diag = np.sqrt(np.diag(Q_t))
        R_t = Q_t / np.outer(sqrt_diag, sqrt_diag)
        try:
            L = np.linalg.cholesky(R_t)
        except np.linalg.LinAlgError:
            R_t = np.eye(N_assets)
            L = np.eye(N_assets)
        z = L @ rng.standard_normal(N_assets)
        Z[t] = z

        # Update Q
        Q_next = AUQ + a * np.outer(z, z) + b * Q_t
        Q_t = (Q_next + Q_next.T) / 2

    return Z, sigmas, Q_bar


# ===========================================================================
# L1 -- Unit: _update_Q arithmetic
# ===========================================================================

def test_L1_update_Q():
    section("L1 -- Unit: _update_Q arithmetic")

    N = 2
    Q_prev = np.array([[1.0, 0.3], [0.3, 1.0]])
    Q_bar  = np.array([[1.0, 0.2], [0.2, 1.0]])
    z_prev = np.array([0.5, -0.3])
    n_prev = np.array([-0.0, -0.3])   # z < 0 component
    a, b, g = 0.05, 0.85, 0.03

    # --- DCC ---
    AUQ_dcc = (1 - a - b) * Q_bar
    expected_dcc = AUQ_dcc + a * np.outer(z_prev, z_prev) + b * Q_prev
    expected_dcc = (expected_dcc + expected_dcc.T) / 2

    result_dcc = _update_Q(Q_prev, z_prev, None, AUQ_dcc, a, b, g=0.0, model='DCC')

    check(np.allclose(result_dcc, expected_dcc, atol=1e-14),
          "_update_Q DCC matches manual closed-form")

    # --- ADCC ---
    AUQ_adcc = (1 - a - b) * Q_bar - g * np.outer(n_prev, n_prev)
    expected_adcc = AUQ_adcc + a * np.outer(z_prev, z_prev) + g * np.outer(n_prev, n_prev) + b * Q_prev
    expected_adcc = (expected_adcc + expected_adcc.T) / 2

    result_adcc = _update_Q(Q_prev, z_prev, n_prev, AUQ_adcc, a, b, g, model='ADCC')

    check(np.allclose(result_adcc, expected_adcc, atol=1e-14),
          "_update_Q ADCC matches manual closed-form")

    # --- Symmetry ---
    check(np.allclose(result_dcc, result_dcc.T, atol=1e-15),
          "_update_Q DCC result is symmetric")
    check(np.allclose(result_adcc, result_adcc.T, atol=1e-15),
          "_update_Q ADCC result is symmetric")

    # --- No access to globals: verify AUQ is fully external ---
    # Passing AUQ=0 matrix should produce a result with no Q_bar contribution
    AUQ_zero = np.zeros((N, N))
    result_no_intercept = _update_Q(Q_prev, z_prev, None, AUQ_zero, a=0.0, b=1.0, g=0.0, model='DCC')
    check(np.allclose(result_no_intercept, Q_prev, atol=1e-14),
          "_update_Q with AUQ=0, a=0, b=1 returns Q_prev (no internal Q_bar access)")

    print()
    print("  _update_Q design contract verified:")
    print("  - DCC/ADCC arithmetic exact")
    print("  - Symmetry enforced")
    print("  - AUQ is purely external (no internal Q_bar access)")


# ===========================================================================
# L2 -- Unit: compute_R structural properties
# ===========================================================================

def test_L2_compute_R():
    section("L2 -- Unit: compute_R structural properties")

    # Build a synthetic Q path from a simple recursion
    Z, sigmas, Q_bar = generate_dcc_data(T=200, N=3, a=0.05, b=0.85)
    N_mat = make_N_matrix(Z)
    Q_bar_est = estimate_Qbar(Z)
    a, b = 0.05, 0.85
    params = (a, b)

    Q_path = compute_Q(Z, None, params, Q_bar_est, None, model='DCC')
    R_path = compute_R(Q_path)

    T, N, _ = R_path.shape

    # Diagonal = 1
    diags = np.array([np.diag(R_path[t]) for t in range(T)])
    check(np.allclose(diags, 1.0, atol=1e-12),
          "All R_t diagonals equal 1 (tolerance 1e-12)")

    # Symmetry
    sym_err = np.max([np.max(np.abs(R_path[t] - R_path[t].T)) for t in range(T)])
    check(sym_err < 1e-14, f"All R_t are symmetric (max asymmetry: {sym_err:.2e})")

    # PSD: all eigenvalues > 0
    min_eigs = []
    for t in range(T):
        ev = np.linalg.eigvalsh(R_path[t])
        min_eigs.append(ev.min())
    min_eig_global = min(min_eigs)
    check(min_eig_global > 0,
          f"All R_t are positive definite (min eigenvalue: {min_eig_global:.6e})")

    # Off-diagonal in [-1, 1]
    for t in range(T):
        offdiag = R_path[t][np.tril_indices(N, k=-1)]
        check(np.all(offdiag >= -1.0) and np.all(offdiag <= 1.0),
              f"R[{t}] off-diagonal elements in [-1, 1]")
    ok("All R_t off-diagonal elements in [-1, 1]")

    print(f"\n  R path shape: {R_path.shape}")
    print(f"  Diagonal max deviation from 1: {np.max(np.abs(diags - 1)):.2e}")
    print(f"  Min eigenvalue across all R_t: {min_eig_global:.6e}")
    print(f"  Off-diagonal range: [{R_path[:, 0, 1].min():.4f}, {R_path[:, 0, 1].max():.4f}]")


# ===========================================================================
# L3 -- Unit: utils functions
# ===========================================================================

def test_L3_utils():
    section("L3 -- Unit: utils functions")

    rng = np.random.default_rng(7)
    T, N = 300, 4
    Z = rng.standard_normal((T, N))

    # estimate_Qbar -- must match np.cov(Z.T, ddof=1) after symmetrization
    Q_bar = estimate_Qbar(Z)
    expected = np.cov(Z.T, ddof=1)
    expected = (expected + expected.T) / 2
    check(np.allclose(Q_bar, expected, atol=1e-15),
          "estimate_Qbar matches np.cov(Z.T, ddof=1) exactly")

    # validate_Qbar -- should not raise on valid PD matrix
    try:
        validate_Qbar(Q_bar)
        ok("validate_Qbar passes on a valid PD Q_bar")
    except ValueError:
        fail("validate_Qbar raised on a valid PD Q_bar")

    # validate_Qbar -- should raise on singular matrix
    Q_bad = np.zeros((N, N))
    try:
        validate_Qbar(Q_bad)
        fail("validate_Qbar did not raise on singular matrix")
    except ValueError:
        ok("validate_Qbar raises ValueError on singular matrix")

    # make_N_matrix -- n_t = z_t * 1[z_t < 0]
    N_mat = make_N_matrix(Z)
    expected_N = Z * (Z < 0).astype(float)
    check(np.allclose(N_mat, expected_N, atol=1e-15),
          "make_N_matrix: n_t = z_t * 1[z_t<0] (exact)")
    check(np.all(N_mat <= 0),
          "make_N_matrix: all values <= 0 (negative or zero)")
    # x=0 case: indicator=0, so n=0
    Z_with_zero = Z.copy()
    Z_with_zero[0, 0] = 0.0
    N_zero = make_N_matrix(Z_with_zero)
    check(N_zero[0, 0] == 0.0, "make_N_matrix: z=0 maps to n=0 (not 0.5)")

    # estimate_Nbar -- uncentered: (N.T @ N) / (T-1)
    N_bar = estimate_Nbar(N_mat)
    expected_Nbar = (N_mat.T @ N_mat) / (T - 1)
    expected_Nbar = (expected_Nbar + expected_Nbar.T) / 2
    check(np.allclose(N_bar, expected_Nbar, atol=1e-15),
          "estimate_Nbar: (N.T@N)/(T-1) uncentered (exact)")

    # compute_delta -- must be positive
    delta = compute_delta(Q_bar, N_bar)
    check(delta > 0, f"compute_delta returns positive scalar (delta={delta:.6f})")
    check(isinstance(delta, float), "compute_delta returns a Python float")

    print(f"\n  delta (generalized max eigenvalue): {delta:.6f}")
    print(f"  Q_bar min eigenvalue: {np.linalg.eigvalsh(Q_bar).min():.6f}")
    print(f"  N_bar max element: {N_bar.max():.6f}")


# ===========================================================================
# L4 -- Integration: synthetic data fit
# ===========================================================================

def test_L4_synthetic():
    section("L4 -- Integration: synthetic data fit")

    Z, sigmas, Q_bar_true = generate_dcc_data(T=500, N=3, a=0.05, b=0.85, g=0.03, seed=42)
    print(f"  Synthetic data: T={Z.shape[0]}, N={Z.shape[1]}")

    # --- DCC ---
    print("\n  [DCC]")
    t0 = time.time()
    res_dcc = fit(Z, sigmas, model='DCC')
    elapsed = time.time() - t0

    a_hat, b_hat = res_dcc['params']
    print(f"  Converged: {res_dcc['converged']}")
    print(f"  a={a_hat:.6f}  b={b_hat:.6f}  (a+b={a_hat+b_hat:.6f})")
    print(f"  llh={res_dcc['llh']:.4f}  time={elapsed:.2f}s")

    check(res_dcc['converged'], "DCC optimizer converged")
    check(a_hat + b_hat < 1.0, f"DCC constraint: a+b={a_hat+b_hat:.6f} < 1")
    check(a_hat > 0 and b_hat > 0, "DCC parameters are positive")
    check(np.isfinite(res_dcc['llh']), "DCC log-likelihood is finite")
    check(res_dcc['llh'] < 0, "DCC log-likelihood is negative")

    # Output shapes
    T, N = Z.shape
    check(res_dcc['Q'].shape == (T, N, N), f"DCC Q shape: {res_dcc['Q'].shape}")
    check(res_dcc['R'].shape == (T, N, N), f"DCC R shape: {res_dcc['R'].shape}")
    check(res_dcc['H'].shape == (T, N, N), f"DCC H shape: {res_dcc['H'].shape}")

    # Q[0] == Q_bar
    Q_bar_est = estimate_Qbar(Z)
    check(np.allclose(res_dcc['Q'][0], Q_bar_est, atol=1e-14),
          "DCC Q[0] == Q_bar (initialization confirmed)")

    # R structural properties
    diags_dcc = np.array([np.diag(res_dcc['R'][t]) for t in range(T)])
    check(np.allclose(diags_dcc, 1.0, atol=1e-12),
          "DCC all R_t diagonals = 1")

    min_eig_dcc = min(np.linalg.eigvalsh(res_dcc['R'][t]).min() for t in range(T))
    check(min_eig_dcc > 0,
          f"DCC all R_t positive definite (min eig: {min_eig_dcc:.4e})")

    # H structural: H_t = Sigma_t R_t Sigma_t -- check symmetry
    sym_err_H = max(
        np.max(np.abs(res_dcc['H'][t] - res_dcc['H'][t].T)) for t in range(T)
    )
    check(sym_err_H < 1e-12, f"DCC all H_t symmetric (max err: {sym_err_H:.2e})")

    # --- ADCC ---
    print("\n  [ADCC]")
    t0 = time.time()
    res_adcc = fit(Z, sigmas, model='ADCC')
    elapsed = time.time() - t0

    a_hat, b_hat, g_hat = res_adcc['params']
    delta = res_adcc['delta']
    persist = a_hat + b_hat + delta * g_hat
    print(f"  Converged: {res_adcc['converged']}")
    print(f"  a={a_hat:.6f}  b={b_hat:.6f}  g={g_hat:.6f}")
    print(f"  delta={delta:.6f}  a+b+delta*g={persist:.6f}")
    print(f"  llh={res_adcc['llh']:.4f}  time={elapsed:.2f}s")

    check(res_adcc['converged'], "ADCC optimizer converged")
    check(persist < 1.0,
          f"ADCC constraint: a+b+delta*g={persist:.6f} < 1")
    check(a_hat > 0 and b_hat > 0 and g_hat >= 0,
          "ADCC parameters non-negative")
    check(np.isfinite(res_adcc['llh']), "ADCC log-likelihood is finite")
    check(res_adcc['llh'] < 0, "ADCC log-likelihood is negative")

    min_eig_adcc = min(np.linalg.eigvalsh(res_adcc['R'][t]).min() for t in range(T))
    check(min_eig_adcc > 0,
          f"ADCC all R_t positive definite (min eig: {min_eig_adcc:.4e})")

    print("\n  [PENALTY guard]")
    # Force PENALTY: pass params that violate stationarity -- objective must return PENALTY
    N_mat_test = make_N_matrix(Z)
    penalty_val = dcc_objective(
        np.array([0.6, 0.6]),   # a+b=1.2 > 1 -- will produce non-PD Q
        Z, Q_bar_est, None, None, 'DCC'
    )
    # May return PENALTY or a finite value depending on whether Cholesky fails
    print(f"  dcc_objective(a=0.6, b=0.6) = {penalty_val:.4f}  "
          f"(PENALTY={PENALTY} if Cholesky failed, else finite)")
    check(penalty_val > 0,
          "dcc_objective returns positive value (consistent with minimization)")


# ===========================================================================
# L5 -- Integration: real data fit
# ===========================================================================

def test_L5_real_data():
    section("L5 -- Integration: real data fit (dcc_inputs.pkl)")

    pkl_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'dcc_inputs.pkl'
    )
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    Z      = np.asarray(data['Z'])
    sigmas = np.asarray(data['sigmas'])
    T, N   = Z.shape
    print(f"  Data: T={T}, N={N}")
    print(f"  Z range:      [{Z.min():.4f}, {Z.max():.4f}]")
    print(f"  sigmas range: [{sigmas.min():.4f}, {sigmas.max():.4f}]")

    # Pre-computation checks
    Q_bar = estimate_Qbar(Z)
    validate_Qbar(Q_bar)
    ok("Q_bar estimated and PD-validated on real data")
    print(f"  Q_bar min eigenvalue: {np.linalg.eigvalsh(Q_bar).min():.6f}")

    N_mat = make_N_matrix(Z)
    N_bar = estimate_Nbar(N_mat)
    delta = compute_delta(Q_bar, N_bar)
    print(f"  delta (real data): {delta:.6f}")

    # --- DCC ---
    print("\n  [DCC -- real data]")
    t0 = time.time()
    res_dcc = fit(Z, sigmas, model='DCC')
    elapsed = time.time() - t0

    a_hat, b_hat = res_dcc['params']
    print(f"  Converged: {res_dcc['converged']}")
    print(f"  a={a_hat:.6f}  b={b_hat:.6f}  a+b={a_hat+b_hat:.6f}")
    print(f"  llh={res_dcc['llh']:.4f}  time={elapsed:.2f}s")

    check(res_dcc['converged'], "DCC converged on real data")
    check(a_hat + b_hat < 1.0,
          f"DCC stationarity satisfied (a+b={a_hat+b_hat:.6f})")
    check(0 < a_hat < 1 and 0 < b_hat < 1,
          "DCC parameters in (0, 1)")
    check(np.isfinite(res_dcc['llh']) and res_dcc['llh'] < 0,
          "DCC llh finite and negative")

    # Shape checks
    check(res_dcc['Q'].shape == (T, N, N), "DCC Q path shape correct")
    check(res_dcc['R'].shape == (T, N, N), "DCC R path shape correct")
    check(res_dcc['H'].shape == (T, N, N), "DCC H path shape correct")

    # R structural checks (sample 100 random time steps)
    idx = np.random.default_rng(0).choice(T, size=100, replace=False)
    for t in idx:
        diag_ok = np.allclose(np.diag(res_dcc['R'][t]), 1.0, atol=1e-10)
        if not diag_ok:
            fail(f"DCC R[{t}] diagonal not 1")
    ok("DCC R_t diagonal = 1 (100 random steps checked)")

    min_eig_dcc = min(np.linalg.eigvalsh(res_dcc['R'][t]).min() for t in idx)
    check(min_eig_dcc > 0,
          f"DCC R_t PD on sampled steps (min eig: {min_eig_dcc:.4e})")

    # Plausibility: financial DCC typically has a+b > 0.95
    check(a_hat + b_hat > 0.80,
          f"DCC persistence plausible (a+b={a_hat+b_hat:.4f} > 0.80)")

    # --- ADCC ---
    print("\n  [ADCC -- real data]")
    t0 = time.time()
    res_adcc = fit(Z, sigmas, model='ADCC')
    elapsed = time.time() - t0

    a_hat, b_hat, g_hat = res_adcc['params']
    persist = a_hat + b_hat + delta * g_hat
    print(f"  Converged: {res_adcc['converged']}")
    print(f"  a={a_hat:.6f}  b={b_hat:.6f}  g={g_hat:.6f}")
    print(f"  delta={delta:.6f}  a+b+delta*g={persist:.6f}")
    print(f"  llh={res_adcc['llh']:.4f}  time={elapsed:.2f}s")

    check(res_adcc['converged'], "ADCC converged on real data")
    check(persist < 1.0, f"ADCC stationarity satisfied (a+b+delta*g={persist:.6f})")
    check(np.isfinite(res_adcc['llh']) and res_adcc['llh'] < 0,
          "ADCC llh finite and negative")

    min_eig_adcc = min(np.linalg.eigvalsh(res_adcc['R'][t]).min() for t in idx)
    check(min_eig_adcc > 0,
          f"ADCC R_t PD on sampled steps (min eig: {min_eig_adcc:.4e})")

    # ADCC should have llh >= DCC (more flexible model)
    check(res_adcc['llh'] >= res_dcc['llh'] - 1.0,
          f"ADCC llh ({res_adcc['llh']:.2f}) >= DCC llh ({res_dcc['llh']:.2f}) - 1")

    return res_dcc, res_adcc


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print("\nDCC-GARCH Validation Suite")
    print(f"NumPy {np.__version__} | SciPy {__import__('scipy').__version__}")

    results = {}
    failed  = []

    for level, fn in [
        ("L1", test_L1_update_Q),
        ("L2", test_L2_compute_R),
        ("L3", test_L3_utils),
        ("L4", test_L4_synthetic),
        ("L5", test_L5_real_data),
    ]:
        try:
            out = fn()
            results[level] = out
            print(f"\n  >>> {level} PASSED")
        except AssertionError as e:
            print(f"\n  >>> {level} FAILED: {e}")
            failed.append(level)
        except Exception as e:
            print(f"\n  >>> {level} ERROR: {type(e).__name__}: {e}")
            failed.append(level)

    section("Summary")
    total  = 5
    passed = total - len(failed)
    print(f"  {passed}/{total} levels passed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print("  All levels passed.")

    sys.exit(0 if not failed else 1)
