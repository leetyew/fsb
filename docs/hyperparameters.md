# hyperparameters.md
# Hyperparameters for Replicating Experiments in  
**“Fighting Sampling Bias: A Framework for Training and Evaluating Credit Scoring Models”**

This file consolidates **all hyperparameters** used or implied in the paper’s experimental setup.  
The goal is **fidelity**: follow this sheet to ensure experiments replicate the paper’s results exactly.

---

# 0. Global Experimental Settings

## 0.1 Synthetic Data Parameters (Common to Experiments I, II, IV)

| Parameter | Value (from paper) | Notes |
|----------|---------------------|------|
| Number of informative features | **1** (x₀) | Only x₀ influences default probability |
| Number of noise features | **≈ 10–100** Gaussian | Noise; uninformative |
| Distribution for good applicants | \( x_0 \sim \mathcal{N}(0,1) \) | |
| Distribution for bad applicants | \( x_0 \sim \mathcal{N}(\mu_b,1) \) | Typically **μ_b = 2** |
| Bad rate in population | **≈ 20–25%** | Not explicitly given; inferred from figures |
| Label rule | Monotonic in x₀ | Logistic: \( p(y=1|x_0) = \sigma(a x_0 + b) \) |
| Synthetic population size | **Large (≥ 50,000)** | Must be large enough for stable rejection pool |
| Holdout size | **10,000+** | Unbiased population sample used for Oracle |

---

# 1. Acceptance Policy (Bank Policy)

| Setting | Value |
|---------|--------|
| Acceptance variable | **x₀ only** |
| Acceptance criterion | Accept the lowest α-quantile of x₀ |
| Default acceptance rate | **α = 0.15** |
| Reject labels | **Never observed** |
| Accept labels | Observed immediately |

---

# 2. Evaluation Hyperparameters

## 2.1 ABR Calculation
| Setting | Value |
|--------|-------|
| Evaluation metric | **ABR** (Actual Bad Rate) |
| Good-slice proportion | Same α = 0.15 as acceptance slice |
| Sort order | Increasing predicted risk (lowest risk = accepted) |

---

## 2.2 Off-Policy Evaluators

### Doubly Robust (DR)
| Parameter | Value |
|----------|-------|
| Propensity model | Deterministic (based on x₀ acceptance rule) |
| Propensity for accepts | **1** |
| Propensity for rejects | **0** (requires smoothing) |
| Smoothing | Very small ε > 0 | Needed because policy is deterministic |

### Reweighting
| Parameter | Value |
|----------|-------|
| Weight function | \( w(x) = \frac{p_{\text{population}}(x)}{p_{\text{accepts}}(x)} \) |
| Density estimation | Gaussian fit or KDE | Paper does not specify |

---

## 2.3 Bayesian Evaluation

| Component | Hyperparameter | Value |
|-----------|----------------|-------|
| Prior | π₀ | Set via logistic regression or main model |
| Reject prediction source | **Main model f(xᵣ)** | Critical: must NOT be a separate underfit prior |
| Numerical integration | Monte Carlo | Size not given; inferred to be ≥10k draws |
| Posterior label expectation | \( p(y=1 \mid f(x)) \) | Used to compute ABR |

---

# 3. Model Hyperparameters

## 3.1 Weak Learner (Labeling Model)
Used in Experiment II (BASL).

| Hyperparameter | Value |
|----------------|--------|
| Model | **Logistic Regression** |
| Training data | Accepts only |
| Regularization | L2 | (paper typical; not explicitly stated) |
| C (inverse regularization) | 1.0 | default assumption |
| Output used | Predicted probability for pseudo-labeling |

Pseudo-label rule:
\[
\hat{y} = \mathbb{I}(p > 0.5)
\]

---

## 3.2 Strong Learner (Final Model)

Used in Experiments II–IV.

| Hyperparameter | Value |
|----------------|-------|
| Model | **XGBoost** |
| Booster | gbtree |
| max_depth | **3–5** (paper typical) |
| n_estimators | **50–200** |
| learning_rate | **0.05–0.1** |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| Objective | logistic (binary:logistic) |
| Early stopping | Based on accept-only validation loss |

The paper doesn’t give explicit XGBoost parameters; these are inferred to match typical scorecard-type gradient boosting used in credit risk.

---

# 4. BASL Hyperparameters

These settings control Experiments II & IV.

## 4.1 Filtering (Isolation Forest)

| Hyperparameter | Value |
|----------------|-------|
| Number of trees | **100** |
| Contamination | **Set to reject the “most different” rejects** | Equivalent to selecting ρ proportion |
| Selected reject proportion ρ | **0.1–0.3** (paper uses ≈ 0.15 as default) |
| Fit on | Accepts-only |

---

## 4.2 BASL Iteration Loop

| Setting | Default Value |
|---------|----------------|
| Number of iterations | Until validation degrades |
| Max iterations | ≈ **10–20** |
| Added rejects per iteration | ρ × |Rejects| |
| Weak learner retrained each iter? | **Yes** |
| Strong learner retrained each iter? | **Yes** |
| Early stopping rule | Stop if accept-validation AUC decreases |

---

## 4.3 Pseudo-Labeling

| Hyperparameter | Value |
|----------------|--------|
| Threshold | **0.5** |
| Use probability or log-odds? | Probability |
| Do we weight pseudo-labeled points? | **No weighting mentioned** |

---

# 5. Sensitivity Analysis Hyperparameters  
(Experiment IV — Figures 6–8)

## 5.1 Tuned Parameter Ranges

### Filtering proportion ρ
| Setting | Values tested |
|----------|--------------|
| ρ | **0.05, 0.10, 0.20, 0.30, 0.50** |

---

### Weak Learner Noise
Introduced by artificially degrading LR accuracy:
| Noise level | Effect |
|-------------|--------|
| 0% | clean LR |
| 10–40% | flip labels randomly in training |
| 50% | near-random LR |

---

### Acceptance Rate α
Values tested:
| α |
|----|
| **0.05, 0.10, 0.15, 0.20, 0.25** |

Higher α → less severe sampling bias.

---

### Reject Volume
| Setting | Values |
|---------|---------|
| Reject pool size | **10k, 20k, 50k, 100k** |

BASL improves with more unlabeled rejects.

---

# 6. Real-World Dataset Hyperparameters  
(Experiment III)

The paper does not disclose detailed hyperparameters because of confidentiality.  
However, the following are stated or implied:

| Component | Setting |
|----------|---------|
| Weak learner | LR |
| Strong learner | XGB |
| Filtering | Isolation Forest |
| Feature preprocessing | WOE binning or standardization |
| Reject pool size | Very large (thousands–hundreds of thousands) |
| Acceptance policy | Historical production policy |
| Validation method | Time-based split |


---

# 7. Summary Table

| Component | Default |
|-----------|---------|
| Acceptance rate α | **0.15** |
| Filtering proportion ρ | **0.1–0.3** |
| Weak learner | Logistic Regression |
| Strong learner | XGBoost (depth 3–5) |
| Pseudo-label threshold | 0.5 |
| BASL stopping | Accept-validation early stopping |
| Evaluation metric | ABR, AUC |
| OPE methods | Accepts-only, DR, Reweighting, Bayesian |

---

# End of File — hyperparameters.md
