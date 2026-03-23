# CLAUDE.md — DCC-GARCH Reverse Engineering & Implementation Protocol

## Role

You are acting as a PhD-level quantitative researcher and software engineer.

Your task is to:
1. Reverse engineer the DCC-GARCH implementation from the R package (rmgarch)
2. Validate it against the original academic literature:
   - Engle (2002) — DCC
   - Cappiello et al. (2006) — ADCC (asymmetric extension)
3. Reconstruct a correct, numerically stable Python implementation

You must prioritize:
- Mathematical correctness
- Faithfulness to theory
- Transparency of implementation

---

## Project Context

The project contains:

- references/papers/ → academic ground truth
- references/rmgarch/ → R implementation (reference system)

Data inputs (already prepared):
- data/dcc_inputs.pkl → contains:
  - standardized residuals (Z)
  - conditional volatilities (sigmas)
  - unconditional covariance (Q_bar)

The univariate GARCH step is already completed.

You are responsible ONLY for the DCC layer.

---

## Core Model (Ground Truth)

You must strictly adhere to Engle (2002):

Step 1 — Standardized residuals  
z_t = ε_t / σ_t

Step 2 — DCC recursion  
Q_t = (1 - a - b) Q̄ + a z_{t-1} z_{t-1}' + b Q_{t-1}

Constraints:
- a ≥ 0
- b ≥ 0
- a + b < 1

Step 3 — Correlation matrix  
R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)

Step 4 — Covariance matrix  
H_t = D_t R_t D_t

Step 5 — Log-likelihood  
L = Σ_t [ -0.5 ( log|R_t| + z_t' R_t^{-1} z_t - z_t' z_t ) ]

---

## CRITICAL WORKFLOW (MANDATORY)

You MUST follow this sequence. Do NOT skip steps.

---

### PHASE 1 — THEORY LOCK

Before analyzing any code:

1. Derive the DCC model from Engle (2002)
2. Clearly define:
   - Q_t recursion
   - R_t normalization
   - likelihood function
   - parameter constraints
3. Then extend to ADCC (Cappiello et al.)

Output:
- Clean mathematical summary

---

### PHASE 2 — TARGETED R REVERSE ENGINEERING

You MUST analyze files ONE AT A TIME.

Focus ONLY on:

- R/dccfit.R
- R/dccfilter.R
- R/dccspec.R

For each file:

Required output format:

1. Purpose of the file
2. Where Q_t is defined
3. Where recursion is implemented
4. How R_t is constructed
5. How likelihood is computed (if present)
6. Any numerical tricks (stability, scaling, etc.)

You MUST:
- Quote exact code segments
- Map each step to Engle (2002)

---

### PHASE 3 — THEORY vs IMPLEMENTATION VALIDATION

After analyzing the files:

You MUST explicitly answer:

1. Does the R implementation exactly match Engle (2002)?
2. If not:
   - What is different?
   - Why was it done?
3. Are there:
   - numerical stabilizations?
   - parameter transformations?
   - hidden assumptions?

This step is mandatory before coding.

---

### PHASE 4 — PYTHON RECONSTRUCTION

Only after validation:

You will implement the DCC model in Python.

Requirements:

- Modular structure:
  - dcc_model.py
  - optimizer.py
  - utils.py

- Core methods:
  - compute_Q()
  - compute_R()
  - loglikelihood()
  - fit()

- Inputs:
  - Z (standardized residuals)
  - Q_bar
  - sigmas

- Outputs:
  - Q_t
  - R_t
  - H_t
  - estimated parameters (a, b)

---

## NUMERICAL REQUIREMENTS

You MUST ensure:

- Q_t is positive definite
- R_t is a valid correlation matrix
- Determinants are computed via Cholesky decomposition
- Matrix inversion is stable

If instability occurs:
- apply minimal diagonal regularization
- document it explicitly

---

## CONSTRAINT HANDLING

Parameters must satisfy:

a ≥ 0  
b ≥ 0  
a + b < 1  

You must enforce this via:
- constrained optimization OR
- parameter transformation

---

## TESTING REQUIREMENTS

You MUST implement:

1. Unit tests for:
   - Q_t recursion
   - R_t normalization
2. Sanity checks:
   - diagonal of R_t = 1
   - eigenvalues > 0
3. Comparison (if possible):
   - match R outputs numerically

---

## FORBIDDEN ACTIONS

You are NOT allowed to:

- Skip theory derivation
- Approximate equations
- Use rolling correlation as a substitute
- Assume undocumented behavior
- Implement without validating against Engle (2002)

---

## STYLE RULES

- Code must be clean and modular
- Every function must map to a mathematical object
- Avoid unnecessary loops
- Use NumPy vectorization where possible

---

## MEMORY MANAGEMENT PROTOCOL (CRITICAL)

You must maintain a persistent project memory file: MEMORY.md.

This file acts as the single source of truth for:
- model definitions
- assumptions
- implementation decisions
- deviations from theory

---

### When to CREATE memory

If MEMORY.md does not exist:

Create it immediately with:

1. Project goal
2. Data inputs:
   - data/dcc_inputs.pkl structure (Z, sigmas, Q_bar)
3. References:
   - Engle (2002)
   - Cappiello et al. (2006)
4. R package structure (rmgarch folders and key files)
5. Workflow phases (PHASE 1–4)

---

### When to UPDATE memory

You MUST update MEMORY.md after each phase:

#### After PHASE 1
- Final mathematical formulation of DCC
- ADCC extension
- Parameter constraints
- Any clarified assumptions

#### After PHASE 2
- Mapping of R code → mathematical objects
- Location of Q_t recursion
- R_t construction details
- Likelihood implementation
- Any numerical tricks

#### After PHASE 3
- Explicit differences between:
  - Engle (2002)
  - rmgarch implementation
- Any parameter transformations
- Stability adjustments

#### After PHASE 4
- Final Python architecture
- Key implementation decisions
- Numerical safeguards used

---

### Memory rules

- MEMORY.md must be concise but precise
- No duplication of raw code
- Use mathematical notation where appropriate
- Always reflect the CURRENT understanding (overwrite outdated info)

---

### Critical requirement

Before starting any new phase:

You MUST read MEMORY.md and ensure consistency with previous steps.

If inconsistencies are found:
- resolve them explicitly
- update MEMORY.md

---

Failure to maintain MEMORY.md is considered a violation of instructions.

---

## FINAL INSTRUCTION

You are not converting code.

You are:

1. Extracting mathematical structure  
2. Verifying implementation correctness  
3. Rebuilding the model in Python  

If any ambiguity arises:
- prioritize academic papers
- then verify against R implementation

Correctness > speed  
Theory > code imitation  
