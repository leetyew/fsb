# Fighting Sampling Bias – Implementation Plan

This document specifies a **code architecture, directory structure, and algorithms** for implementing the full workflow from the paper *“Fighting Sampling Bias: A Framework for Training and Evaluating Credit Scoring Models”*.

The implementation must:

- Support **synthetic data experiments** (Gaussian mixture + acceptance loop).
- Support **real company data** (accepts + rejects, with an optional representative holdout).
- Implement **Bias-Aware Self-Learning (BASL)** for reject inference.
- Implement the **Bayesian evaluation framework** for performance estimation under sampling bias.
- Be **modular** enough to swap:
  - Synthetic vs real data.
  - XGBoost vs any company-internal model (via adapter interface).

The architecture is written in `.md` format for direct use by code generators.

---

## 1. High-Level Workflow

The pipeline consists of:

1. **Data Source**
   - Synthetic generator or real data loader.

2. **Preprocessing**
   - Feature engineering & pipelines.

3. **Baseline Scorecard Training**
   - Model trained only on accepts.

4. **Bias-Aware Self-Learning (BASL)**
   - Reject filtering.
   - Weak-learner labeling of rejects.
   - Augmented training of final model.

5. **Evaluation**
   - Standard biased metrics (accepts-only).
   - Bayesian evaluation on accepts + rejects.
   - Synthetic experiments additionally compare against oracle holdout.

6. **Reporting**
   - Compare baseline vs BASL vs oracle (synthetic).

---

## 2. Repository Layout

```text
fighting_sampling_bias/
├── README.md
├── pyproject.toml / setup.py
├── requirements.txt
├── docs/
│   └── architecture.md
├── configs/
│   ├── synthetic_data.yaml       # [DONE]
│   ├── model_xgboost.yaml        # [DONE]
│   ├── basl.yaml                 # [DONE]
│   └── acceptance_loop.yaml      # [DONE]
├── data/
│   ├── raw/
│   │   ├── synthetic/
│   │   └── real/
│   ├── interim/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── config.py                 # [DONE] Pydantic config classes
│   ├── io/
│   │   ├── synthetic_generator.py       # [DONE]
│   │   ├── synthetic_acceptance_loop.py # [DONE]
│   │   ├── real_data_loader.py          # [DEFERRED] Format not confirmed
│   │   └── dataset_splits.py            # [DEFERRED] Format not confirmed
│   ├── preprocessing/
│   │   └── feature_pipeline.py   # [DONE] Pass-through for synthetic
│   ├── models/
│   │   ├── xgboost_model.py         # [DONE]
│   │   ├── logistic_regression.py   # [DONE] Weak learner for BASL
│   │   └── internal_model.py        # [DEFERRED] Format not confirmed
│   ├── basl/
│   │   ├── filtering.py    # [DONE] Stage 1: Novelty detection filtering
│   │   ├── labeling.py     # [DONE] Stage 2: Pseudo-labeling iteration
│   │   └── trainer.py      # [DONE] Orchestrates stages
│   └── evaluation/
│       ├── metrics.py       # [DONE] AUC, PAUC, Brier, ABR
│       ├── bayesian_eval.py # [DONE] Monte Carlo evaluation
│       └── profit.py        # [DONE] Profit metrics
├── scripts/
│   └── run_experiment.py    # [DONE] Main entry point
└── notebooks/
    └── (future)
```

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Synthetic Generator | Done | GMM-based data generation |
| Acceptance Loop | Done | Baseline + BASL modes |
| XGBoost Model | Done | Paper hyperparameters |
| Logistic Regression | Done | L1-regularized weak learner |
| BASL (filtering, labeling, trainer) | Done | Full implementation |
| Evaluation Metrics | Done | AUC, PAUC, Brier, ABR |
| Bayesian Evaluation | Done | Monte Carlo sampling |
| Profit Metrics | Done | Expected/realized profit |
| Feature Pipeline | Done | Pass-through for synthetic |
| Experiment Script | Done | End-to-end runner |
| Real Data Loader | Deferred | Format not confirmed |
| Internal Model | Deferred | Format not confirmed |

---

## 3. Core Configuration Objects

Example YAML config:

```yaml
data:
  source: "synthetic"
  synthetic:
    random_seed: 42
    n_features: 2  # Paper uses 2D for visualization; can increase
    n_components: 2  # C: number of Gaussian mixture components
    bad_rate: 0.70  # b: proportion of bad applicants in population
    n_holdout: 3000  # h: representative holdout size
    # Gaussian mixture means (paper Table E.9):
    # μ_g1 = (0, 0), μ_b1 = (2, 1) for component 1
    # μ_g2 = μ_g1 + 1, μ_b2 = μ_b1 + 1 for component 2
    gaussian_mixture:
      mu_good_base: [0.0, 0.0]  # μ_g1
      mu_bad_base: [2.0, 1.0]   # μ_b1
      component_offset: 1.0     # Added to each dimension for subsequent components
      sigma_max: 1.0            # Covariance entries sampled from U(0, σ_max)
    missingness:
      type: "MAR"  # MCAR, MAR, or MNAR (see Section 4.1 synthetic data generation)
      strength: 0.5  # Controls severity of selection bias in acceptance policy
  real:
    accepts_path: "data/raw/real/accepts.parquet"
    rejects_path: "data/raw/real/rejects.parquet"
    holdout_path: null
  splits:
    train_fraction: 0.6
    val_fraction: 0.2
    test_fraction: 0.2

model:
  base_model_type: "xgboost"
  params_file: "configs/model_xgboost.yaml"

basl:
  enabled: true
  max_iterations: 5  # j_max from paper; early stopping may terminate sooner
  filtering:
    novelty_detector: "isolation_forest"
    beta_lower: 0.05  # Remove bottom 5% outliers (most dissimilar to accepts)
    beta_upper: 1.0   # No upper filtering (paper default)
  labeling:
    subsample_ratio: 0.8  # ρ: 0.8 for synthetic, 0.3 for real data
    weak_learner_type: "logreg_l1"
    gamma: 0.01       # Percentile threshold for confident labels
    theta: 2.0        # Imbalance multiplier: label γ as good, θ*γ as bad
    fix_thresholds_after_first_iter: true  # Fix absolute threshold values after iteration 1

evaluation:
  metrics: ["auc", "pauc", "brier", "abr", "profit"]  # PAUC = partial AUC (high-specificity region)
  bayesian:
    enabled: true
    n_samples: 5000
    n_score_bands: 10  # K: number of score bands for stratified sampling
    prior:
      type: "beta"
      alpha: 1.0  # Uninformative prior
      beta: 1.0
    seed: 123

simulation:
  acceptance_loop:
    enabled: true
    n_periods: 500       # Number of lending iterations
    batch_size: 100      # n: applicants per period (paper: 100)
    target_accept_rate: 0.15  # α: acceptance rate (paper: 0.15)
    threshold_policy: "quantile"  # Accept top α quantile by score
    initial_labeled_size: 100  # Initial batch with observed labels to bootstrap model
```

`fsb.config` would provide typed classes (e.g. via pydantic) for these config sections.

---

## 4. Data Layer

### 4.1 Synthetic Data Generation

Implements Gaussian mixture generation (per paper Appendix E.1):

**Gaussian Mixture Model:**

For C components, each class (good/bad) has component-specific parameters:

```
For component c = 1, ..., C:
  Good applicants: X_good ~ N(μ_gc, Σ_gc)
  Bad applicants:  X_bad  ~ N(μ_bc, Σ_bc)

Default means (paper Table E.9):
  μ_g1 = (0, 0),  μ_b1 = (2, 1)
  μ_gc = μ_g1 + (c-1),  μ_bc = μ_b1 + (c-1)  for c > 1

Covariance:
  Σ entries sampled from U(0, σ_max), σ_max = 1.0
  Make symmetric positive-definite via Σ = A @ A.T + εI
```

**Population Generation:**

1. Sample component assignment uniformly.
2. Sample label y from Bernoulli(bad_rate).
3. Sample features from N(μ, Σ) for the assigned component and class.
4. Optionally add noise features from N(0, 1).

Suggested interface:

```python
class SyntheticGenerator:
    def __init__(self, cfg: SyntheticConfig): ...
    def generate_population(self) -> pd.DataFrame: ...
    def generate_holdout(self) -> pd.DataFrame: ...
```

**Acceptance Policy (Missingness Types):**

The acceptance decision determines which labels are observed. Let s(x) be the scorecard score:

- **MCAR** (Missing Completely At Random):
  ```
  P(accept | x, y) = α  (constant)
  ```
  Selection is independent of both features and outcome.

- **MAR** (Missing At Random):
  ```
  P(accept | x, y) = P(accept | s(x))
  ```
  Selection depends on score (features), not directly on outcome.
  Accept the top α quantile by score.

- **MNAR** (Missing Not At Random):
  ```
  P(accept | x, y) = P(accept | s(x), y)
  ```
  Selection depends on both score and outcome.
  Example: Accept if score > threshold OR if y=0 (good applicants have higher chance of acceptance beyond score).

The `missingness.strength` parameter controls the degree of outcome-dependence in MNAR.

### 4.2 Acceptance Loop Simulation

Simulates lender behavior over multiple periods to create selection-biased datasets.

**Key Paper Detail (Appendix C.1, page A8):**

Initial acceptance uses `x_v` - the feature with the largest difference in mean values between good and bad applicants. This is used for rule-based ranking before any model exists.

> "x_v refers to the feature with the largest difference in mean values between good and bad applicants and represents a powerful attribute, such as a bureau score, which can be used to perform a rule-based application ranking."

**Implementation Note:** The paper auto-detects `x_v` from data. Our implementation uses `x_v_feature` in config instead for explicit control and reproducibility.

**Algorithm:**

1. Generate `initial_batch_size` applicants from population.
2. **Initial acceptance (no model yet):**
   - Use `x_v_feature` from config (paper: auto-detect feature with max |mean_good - mean_bad|).
   - Accept top α quantile by `x_v` (lower `x_v` = lower risk).
   - These accepted applicants form initial `D_a`.
3. Train initial XGBoost model on `D_a`.
4. For each period j = 1..n_periods (with tqdm progress bar):
   - Generate `batch_size` new applicants from population.
   - Score applicants with current model.
   - Accept top α quantile by score (lower score = lower risk).
   - Reveal labels for accepts, add to `D_a`.
   - Add rejects (features only, no y) to `D_r`.
   - Retrain model on accumulated `D_a`.
5. Generate holdout `H` independently.
6. Return `(D_a, D_r, H)`.

**Interface:**

```python
@dataclass
class AcceptanceLoopConfig:
    n_periods: int = 500
    batch_size: int = 100
    target_accept_rate: float = 0.15
    initial_batch_size: int = 100
    x_v_feature: str = "x0"  # Feature for initial acceptance (before model exists)
    random_seed: int = 42

class AcceptanceLoop:
    def __init__(
        self,
        generator: SyntheticGenerator,
        model_cfg: XGBoostConfig,
        cfg: AcceptanceLoopConfig,
        basl_cfg: Optional[BASLConfig] = None,  # None = baseline mode
    ): ...

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns (D_a, D_r, H).

        Modes:
        - Baseline (basl_cfg=None): Retrain on D_a only each iteration
        - BASL (basl_cfg provided): Apply BASL each iteration (per paper A9)
        """
```

**Key Design Decisions:**
- Model is retrained each iteration on accumulated `D_a`.
- `D_r` contains only features (no y column) - labels are never observed for rejects.
- Progress bar (tqdm) for visibility during 500 iterations.

### 4.3 Real Data Loader

Loads accepts, rejects, and optional holdout from company datasets:

```python
class RealDataLoader:
    def __init__(self, cfg: RealDataConfig): ...
    def load_accepts(self) -> pd.DataFrame: ...
    def load_rejects(self) -> pd.DataFrame: ...
    def load_holdout(self) -> pd.DataFrame | None: ...
```

### 4.4 Dataset Splits Utilities

Provide helpers to create deterministic splits:

```python
class DatasetSplits(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def make_splits(df: pd.DataFrame, cfg: SplitConfig) -> DatasetSplits:
    ...
```

Splits can be:

- Time-based (recommended),
- Or random by customer/application ID.

---

## 5. Preprocessing

Provide a shared feature pipeline for both synthetic and real data:

- Impute missing values.
- Scale numerical features.
- Encode categorical features.
- Optionally select features.

Interface:

```python
class FeaturePipeline:
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None): ...
    def transform(self, df: pd.DataFrame) -> np.ndarray: ...
    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray: ...
    def save(self, path: str): ...
    @classmethod
    def load(cls, path: str) -> "FeaturePipeline": ...
```

All models and BASL components operate on the transformed `np.ndarray`.

---

## 6. Model Abstractions

### 6.1 XGBoost Implementation

No BaseModel abstract class (YAGNI). Direct XGBoost implementation:

```python
@dataclass
class XGBoostConfig:
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_seed: int = 42

class XGBoostModel:
    def __init__(self, cfg: XGBoostConfig): ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return PD estimates for class y=1."""
```

**Note:** Add BaseModel abstraction later if needed for company-internal models.

**Default hyperparameters (from paper Table E.9):**

```yaml
# configs/model_xgboost.yaml
xgboost:
  # Synthetic experiments
  synthetic:
    n_estimators: 100        # max_trees (paper uses 100 for synthetic)
    max_depth: 3
    learning_rate: 0.1       # eta
    subsample: 0.8           # bagging_ratio
    colsample_bytree: 0.8    # feature_ratio
    objective: "binary:logistic"
    eval_metric: "auc"
    early_stopping_rounds: 10

  # Real data experiments (longer training)
  real:
    n_estimators: 10000      # max_trees (paper uses 10000 for real)
    max_depth: 3
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    objective: "binary:logistic"
    eval_metric: "auc"
    early_stopping_rounds: 50
```

### 6.2 Logistic Regression (Weak Learner)

Used by BASL for labeling. L1 regularization encourages sparse solutions:

```python
class LogisticRegressionModel(BaseModel):
    def __init__(self, penalty: str = "l1", C: float = 1.0, solver: str = "saga"): ...
```

**Default parameters:**

```python
# sklearn.linear_model.LogisticRegression
weak_learner_params = {
    "penalty": "l1",
    "C": 1.0,            # Inverse regularization strength
    "solver": "saga",    # Required for L1 penalty
    "max_iter": 1000,
    "random_state": seed,
}
```

### 6.3 Internal Model (Future)

Adapter for company-specific models. When needed, add a BaseModel interface and implement it:

```python
class InternalModel(BaseModel):
    """Adapter for company-internal model. Implement as needed."""
    def __init__(self, model_config: dict): ...
    def fit(...): ...
    def predict_proba(...): ...
    def save(...): ...
    @classmethod
    def load(...): ...
```

---

## 7. BASL – Bias-Aware Self-Learning

BASL has four stages:

1. **Filtering**: Remove outlier rejects via novelty detection.
2. **Labeling**: Iteratively pseudo-label rejects using a weak learner.
3. **Training**: Train final scorecard on accepts + pseudo-labeled rejects.
4. **Early stopping** (optional): Use Bayesian evaluation to detect overfitting to mislabeled rejects.

### 7.1 Reject Filtering

Train a novelty detector (Isolation Forest) on accepts, then score rejects:

- Rejects with the lowest similarity (most extreme outliers) are dropped.
- Rejects with the highest similarity (too similar to accepts) are also dropped.
- Only the "middle region" is retained for subsequent labeling.

**Isolation Forest Parameters:**

```python
# sklearn.ensemble.IsolationForest defaults used in paper
isolation_forest_params = {
    "n_estimators": 100,
    "contamination": "auto",  # or set to expected outlier fraction
    "random_state": seed,
}
```

**Filtering Logic:**

```python
# 1. Fit Isolation Forest on accepts
iforest.fit(X_accepts)

# 2. Score rejects (lower = more outlier-like)
outlier_scores = iforest.decision_function(X_rejects)

# 3. Keep rejects in [beta_lower, beta_upper] percentile range
lower_thresh = np.percentile(outlier_scores, beta_lower * 100)
upper_thresh = np.percentile(outlier_scores, beta_upper * 100)
keep_mask = (outlier_scores >= lower_thresh) & (outlier_scores <= upper_thresh)
```

Interface:

```python
def filter_rejects(
    X_accepts: np.ndarray,
    X_rejects: np.ndarray,
    beta_lower: float,
    beta_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X_rejects_filtered, indices_kept)."""
```

### 7.2 Weak Learner Labeling

Per iteration:

1. Subsample a fraction `ρ` from the current reject pool.
2. Fit weak learner (L1 logistic regression) on current labeled data `(L_X, L_y)`.
3. Score the subsampled rejects with the weak learner.
4. Define confidence thresholds:
   - **Iteration 1**: Compute `τ_good` as `γ`-percentile, `τ_bad` as `(1 - θ*γ)`-percentile of scores.
   - **Iteration 2+**: Reuse the absolute threshold values from iteration 1 (fixed thresholds).
5. Label rejects:
   - Score ≤ `τ_good` → label as "good" (y=0).
   - Score ≥ `τ_bad` → label as "bad" (y=1).
   - Middle scores remain unlabeled.
6. Add newly labeled rejects to the labeled set.
7. Remove labeled rejects from the pool.

The threshold fixing ensures consistent labeling criteria as the weak learner improves.

Interface:

```python
def label_rejects_iteration(
    weak_learner: BaseModel,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_rejects_pool: np.ndarray,
    gamma: float,
    theta: float,
    subsample_ratio: float,
    rng: np.random.Generator,
    fixed_thresholds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Return (X_new, y_new, remaining_pool_indices, thresholds).

    On first call, pass fixed_thresholds=None to compute thresholds.
    On subsequent calls, pass the returned thresholds to fix them.
    """
```

### 7.3 BASL Trainer

Orchestrates filtering and labeling stages, returning augmented data for external training:

```python
class BASLTrainer:
    def __init__(
        self,
        cfg: BASLConfig,
        feature_pipeline: FeaturePipeline,
    ): ...

    def run(
        self,
        X_a: np.ndarray,
        y_a: np.ndarray,
        X_r: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (X_augmented, y_augmented) for external model training.

        The caller handles model training with their own pipeline.
        """
        ...
```

Algorithm outline:

1. **Filter rejects:**

   - `X_r_filtered, idx_kept = filter_rejects(X_a, X_r, beta_lower, beta_upper)`.

2. **Initialize labeled and unlabeled sets:**

   - Labeled: `L_X = X_a`, `L_y = y_a`.
   - Pool: `U_X = X_r_filtered`.

3. **Iterative labeling (up to `max_iterations`):**

   - `fixed_thresholds = None`

   For each iteration `j` in `1..max_iterations`:

   - Fit weak learner (L1 LogReg) on `(L_X, L_y)`.
   - `(Δ_X, Δ_y, new_pool_indices, thresholds) = label_rejects_iteration(...)`.
   - If `j == 1`: `fixed_thresholds = thresholds` (fix for subsequent iterations).
   - Append `Δ_X, Δ_y` to `(L_X, L_y)`.
   - Update `U_X` to remaining pool.
   - **Early stopping** (optional): Run Bayesian evaluation; stop if metric degrades.

4. **Return:**

   - `(L_X, L_y)` — augmented dataset (accepts + pseudo-labeled rejects).
   - Caller trains final model externally using their own training pipeline.

---

## 8. Evaluation & Metrics

### 8.1 Standard Metrics

Implement:

- **ROC AUC**: Area under the ROC curve.
- **PAUC**: Partial AUC measured over FNR ∈ [0, 0.2] (false negative rate range), normalized to [0, 1]. This focuses on the high-recall region where missing true positives (bad loans) is costly.
- **Brier score**: Mean squared error of probability predictions.
- **ABR** (Average Bad Rate): Average true default rate among accepted applicants.
- Other business metrics as required (e.g., profit).

Interface:

```python
def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metrics: list[str],
) -> dict[str, float]:
    ...
```

### 8.2 Profit / Business Value

Given:

- PD estimates,
- LGD,
- Exposure/loan amount,
- Interest margin,

compute expected profit per account and aggregate.

---

## 9. Bayesian Evaluation Framework

The Bayesian evaluation uses:

- Observed labels for accepts.
- Model scores for both accepts and rejects.
- Priors for reject default rates.

It produces posterior distributions for metrics like AUC, Brier, ABR, and profit.

### 9.1 Interface

```python
@dataclass
class BayesianEvalConfig:
    n_samples: int
    n_score_bands: int  # K: number of score bands for stratified posterior sampling
    prior_type: str
    prior_params: dict[str, float]  # e.g., {"alpha": 1.0, "beta": 1.0}
    metrics: list[str]
    seed: int

def bayesian_evaluate(
    y_accepts: np.ndarray,
    scores_accepts: np.ndarray,
    scores_rejects: np.ndarray,
    cfg: BayesianEvalConfig,
) -> dict[str, Any]:
    ...
```

### 9.2 Algorithm (Paper Algorithm 1)

**Input:**
- `y_a`: True labels for accepts.
- `s_a`: Scores for accepts.
- `s_r`: Scores for rejects.
- `N`: Number of Monte Carlo samples.
- Prior parameters (α, β for Beta distribution).

**Procedure:**

1. **Bin scores** into K score bands (e.g., deciles) based on combined score distribution.

2. **For each score band k:**
   - Count accepts: `n_a[k]`, observe bad count: `d_a[k]`.
   - Count rejects: `n_r[k]`.

3. **For each Monte Carlo sample i in 1..N:**

   a. **Sample band-specific bad rates** from posterior:
      - `p_k ~ Beta(α + d_a[k], β + n_a[k] - d_a[k])` for each band k.

   b. **Pseudo-label rejects** by sampling from Binomial:
      - For each reject in band k: `y_r ~ Bernoulli(p_k)`.

   c. **Combine datasets:**
      - `y_combined = concat(y_a, y_r_sampled)`.
      - `s_combined = concat(s_a, s_r)`.

   d. **Compute metrics** on combined data:
      - AUC, PAUC, Brier, ABR, etc.

   e. **Store metrics** for sample i.

4. **Aggregate** across samples to get posterior statistics (mean, median, credible intervals).

**Output:**

```python
{
  "metrics": {
    "auc": {"mean": ..., "median": ..., "q2.5": ..., "q97.5": ...},
    "pauc": {...},
    "brier": {...},
    "abr": {...},
  }
}
```

The band-specific posterior sampling respects local calibration: rejects in high-score bands get lower sampled bad rates than those in low-score bands.

---

## 10. Simulation Experiments

Wrap the synthetic pipeline in a class:

```python
class AcceptanceLoopSim:
    def __init__(self, cfg: SimulationConfig, data_cfg: SyntheticConfig, model_cfg: ModelConfig): ...

    def run(self) -> dict[str, Any]:
        """
        - Generate population & holdout
        - Run acceptance loop to get D_a, D_r
        - Train baseline on D_a
        - Run BASL to get corrected model
        - Run Bayesian evaluation
        - Compute oracle metrics on holdout
        - Return metrics and artifacts
        """
```

The results can be easily plotted in notebooks.

---

## 11. Runners & CLI

### 11.1 Main Experiment Script

`scripts/run_experiment.py`:

```python
def main(config_path: str):
    cfg = load_experiment_config(config_path)

    # 1. Data source
    if cfg.data.source == "synthetic":
        generator = SyntheticGenerator(cfg.data.synthetic)
        model_for_sim = build_model_from_config(cfg.model)
        loop = AcceptanceLoop(generator, model_for_sim, cfg.simulation.acceptance_loop)
        D_a, D_r, H = loop.run()
    else:
        loader = RealDataLoader(cfg.data.real)
        D_a = loader.load_accepts()
        D_r = loader.load_rejects()
        H = loader.load_holdout()

    # 2. Preprocessing
    feature_pipeline = FeaturePipeline(...)
    X_a = feature_pipeline.fit_transform(D_a.drop(columns=["y"]))
    y_a = D_a["y"].to_numpy()
    X_r = feature_pipeline.transform(D_r)

    X_H = None
    y_H = None
    if H is not None:
        X_H = feature_pipeline.transform(H.drop(columns=["y"]))
        y_H = H["y"].to_numpy()

    # 3. Baseline model
    baseline_model = build_model_from_config(cfg.model)
    baseline_model.fit(X_a, y_a)

    # 4. BASL
    basl_model = None
    augmented_data = None
    if cfg.basl.enabled:
        basl_trainer = BASLTrainer(
            base_model_factory=lambda: build_model_from_config(cfg.model),
            weak_learner_factory=lambda: LogisticRegressionModel(penalty="l1"),
            cfg=cfg.basl,
            feature_pipeline=feature_pipeline,
        )
        basl_model, augmented_data = basl_trainer.run_basl(D_a, D_r)

    # 5. Evaluation
    results = {}

    if cfg.evaluation.bayesian.enabled:
        scores_a = baseline_model.predict_proba(X_a)
        scores_r = baseline_model.predict_proba(X_r)
        bayes_res = bayesian_evaluate(
            y_accepts=y_a,
            scores_accepts=scores_a,
            scores_rejects=scores_r,
            cfg=cfg.evaluation.bayesian,
        )
        results["bayesian_baseline"] = bayes_res

        if basl_model is not None:
            scores_a_basl = basl_model.predict_proba(X_a)
            scores_r_basl = basl_model.predict_proba(X_r)
            bayes_res_basl = bayesian_evaluate(
                y_accepts=y_a,
                scores_accepts=scores_a_basl,
                scores_rejects=scores_r_basl,
                cfg=cfg.evaluation.bayesian,
            )
            results["bayesian_basl"] = bayes_res_basl

    if H is not None:
        scores_H_base = baseline_model.predict_proba(X_H)
        results["oracle_baseline"] = compute_metrics(y_H, scores_H_base, cfg.evaluation.metrics)

        if basl_model is not None:
            scores_H_basl = basl_model.predict_proba(X_H)
            results["oracle_basl"] = compute_metrics(y_H, scores_H_basl, cfg.evaluation.metrics)

    save_results(results, output_dir=cfg.output_dir)
```

### 11.2 Specialized Runners

Additional entry points can be added:

- `train_baseline.py` – train and save a model.
- `run_basl.py` – only run BASL on pre-split data.
- `run_bayesian_eval.py` – run Bayesian evaluation on an existing model and dataset.

---

## 12. Extensibility Notes

- **Synthetic vs Real Data:**
  - Both produce standardized `(D_a, D_r, H)` objects.
  - BASL and Bayesian evaluation are data-source agnostic.

- **Model Swapping:**
  - Any `BaseModel` implementation (XGBoost, sklearn, company model) can be used.
  - For production, swap in the `CompanyModelAdapter` with minimal code changes.

- **Adding New Bias-Correction Methods:**
  - Implement additional strategies (e.g., reweighting, Heckman, mixture models) using the same interfaces and dataset abstractions.

- **Adding New Metrics:**
  - Add functions to `evaluation/metrics.py` and list them in the config.
  - Bayesian evaluation will automatically support them.

This `.md` file is intended to be fed directly into a code-writing assistant to generate the full implementation of the framework, while keeping the design modular, testable, and extendable.

---

## 13. Paper Comparison Tables

Reference tables from the paper for comparing experimental results.

### 13.1 Experiment I: Evaluation Accuracy (Table C.3, Page A9)

Performance prediction accuracy comparing Accepts-based vs Bayesian evaluation methods.

| Evaluation Method | Metric | Bias   | Variance | RMSE   |
|-------------------|--------|--------|----------|--------|
| Accepts-based     | AUC    | 0.1923 | 0.0461   | 0.2205 |
| Bayesian          | AUC    | 0.0910 | 0.0001   | 0.1000 |
| Accepts-based     | BS     | 0.0748 | 0.0006   | 0.0828 |
| Bayesian          | BS     | 0.0038 | 0.0009   | 0.0566 |
| Accepts-based     | PAUC   | 0.2683 | 0.0401   | 0.2803 |
| Bayesian          | PAUC   | 0.1102 | 0.0002   | 0.1187 |
| Accepts-based     | ABR    | 0.1956 | 0.0004   | 0.2010 |
| Bayesian          | ABR    | 0.0039 | 0.0040   | 0.0929 |

**Interpretation:**
- Bias = |Estimated Metric - Oracle Metric| (systematic error)
- Variance = Var(Estimated Metric) (estimation variability)
- RMSE = sqrt(Bias² + Variance) (total error)
- Bayesian evaluation significantly reduces bias for all metrics

### 13.2 Experiment II: Training Bias Impact (Table C.4, Page A10)

Loss due to sampling bias and gains from BASL on synthetic data.

| Metric | Loss due to Bias | Gain from BASL |
|--------|------------------|----------------|
| AUC    | 0.0591           | 35.72%         |
| BS     | 0.0432           | 29.29%         |
| PAUC   | 0.0535           | 22.42%         |
| ABR    | 0.0598           | 24.82%         |
| MMD    | 0.5737           | 3.74%          |

**Definitions:**
- **Loss due to Bias** = Oracle(Full Data Model) - Oracle(Accepts-only Model)
  - Measures performance degradation from training on biased accepts-only data
- **Gain from BASL** = [BASL Improvement / Loss due to Bias] × 100%
  - Percentage of lost performance recovered by BASL
  - BASL Improvement = Oracle(BASL Model) - Oracle(Accepts-only Model)

### 13.3 Key Experimental Parameters (Table E.9, Page A20)

| Parameter | Value | Description |
|-----------|-------|-------------|
| n_periods | 500 | Number of acceptance loop iterations |
| batch_size | 100 | Applicants per period |
| accept_rate (α) | 0.15 | Fraction of applicants accepted |
| bad_rate (b) | 0.70 | True default rate in population |
| holdout_size | 3,000 | Unbiased holdout sample size |
| n_features | 2 | Number of features |
| n_components | 2 | Gaussian mixture components |
| beta_lower | 0.05 | Filtering: remove bottom 5% outliers |
| beta_upper | 1.0 | Filtering: no upper filtering |
| gamma (γ) | 0.01 | Labeling: percentile threshold |
| theta (θ) | 2.0 | Labeling: imbalance multiplier |
| rho (ρ) | 0.8 | Labeling: subsample ratio |
| max_iterations | 5 | Maximum BASL iterations |

### 13.4 Figure 2 Panel (e): ABR Over Acceptance Loop

Shows ABR on holdout sample across 500 acceptance loop iterations:

- **Oracle ABR:** Stabilizes around ~0.15 after 100 iterations
- **Accepts-based ABR:** Consistently higher (~0.20-0.22), demonstrating training bias
- **BASL ABR:** Recovers approximately 25% of the gap between accepts-based and oracle

**Key insight:** The gap between Oracle and Accepts-based represents the "loss due to sampling bias" that BASL aims to recover.

### 13.5 Expected Metric Ranges

Based on paper experiments, typical values for synthetic data with 70% bad rate:

| Metric | Oracle Range | Biased Range | Notes |
|--------|--------------|--------------|-------|
| AUC    | 0.85-0.95    | 0.80-0.90    | Higher is better |
| PAUC   | 0.70-0.90    | 0.60-0.80    | Higher is better |
| Brier  | 0.10-0.20    | 0.15-0.25    | Lower is better |
| ABR    | 0.10-0.20    | 0.15-0.25    | Lower is better |

**Note:** Actual values depend on data separability and bias severity. Our holdout bad rate (~0.70) should produce ABR values near the population bad rate when acceptance is poorly calibrated.
