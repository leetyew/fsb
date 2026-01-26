# Fighting Sampling Bias

Implementation of "Fighting Sampling Bias: A Framework for Training and Evaluating Credit Scoring Models".

## Quick Start

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate fsb

# 2. Generate Figure 2 data (10 seeds, ~30 min)
python scripts/run_figure2_unified.py --config configs/figure2_unified.yaml

# 3. Plot Figure 2
jupyter notebook notebooks/figure_2.ipynb
```

---

## Experiments I & II (Tables 2 & 3)

### Step 1: Generate Synthetic Data

Generate realistic synthetic credit data with 50 features and run the acceptance loop simulation.

```bash
python scripts/generate_synthetic_like_real.py --config configs/synthetic_generator.yaml
```

**Output structure:**
```
data/synthetic/<run_id>/
├── Da.csv          # Accepts (labeled, y ∈ {0,1})
├── Dr.csv          # Rejects (unlabeled, no y column)
├── H.csv           # Holdout (labeled, representative)
├── meta.json       # Parameters, achieved rates, schema
└── snapshots/      # Iteration diagnostics
```

### Step 2: Run Experiment I (Table 2 - Evaluation Accuracy)

Compares biased vs Bayesian evaluation methods across 100 replicates (4 folds × 25 bootstraps).

```bash
python scripts/run_experiment_1.py --data-dir data/synthetic/<run_id>
```

**Output:** `experiments/exp1_<timestamp>/exp1_results.csv`

### Step 3: Run Experiment II (Table 3 - Training Comparison)

Compares accepts-only vs BASL training methods across 100 replicates (4 folds × 25 bootstraps).

```bash
python scripts/run_experiment_2.py --data-dir data/synthetic/<run_id>
```

**Output:** `experiments/exp2_<timestamp>/exp2_results.csv`

### Step 4: Verify & Analyze

```bash
# Verify results
python scripts/verify_experiments.py

# Generate Tables 2 & 3
jupyter notebook notebooks/table2_table3.ipynb
```

---

## Customizing Synthetic Data

Modify `configs/synthetic_generator.yaml` to customize data generation.

### Feature Customization

```yaml
features:
  n_continuous: 60    # Skewed + bounded ratio features (default: 30)
  n_count: 20         # Count/integer features (default: 10)
  n_binary: 10        # Binary features (default: 5)
  n_categorical: 10   # Categorical with 4-8 levels (default: 5)
  total: 100          # Must equal sum of above
```

### Size Customization

```yaml
sizes:
  n_population: 200000     # Total population pool
  n_holdout: 20000         # Representative holdout size
  n_accepts_target: 40000  # Target number of accepts
  # n_rejects will be remaining pool (~140000)
```

### Acceptance Loop Customization

```yaml
acceptance_loop:
  n_periods: 500           # T = number of iterations
  batch_size: 80           # Applicants accepted per period
  sigma_policy: 0.25       # Stochastic noise in [0.15, 0.35]
  acceptance_mode: "stochastic_topk"  # or "topk"
  initial_seed_size: 2000  # Da0 via noisy policy
```

### XGBoost Hyperparameters

```yaml
xgboost:
  max_depth: 3
  n_estimators: 200
  learning_rate: 0.07
  subsample: 0.8
  colsample_bytree: 0.8
```

---

## Using Your Own Data

You can use your own credit data by providing CSV files in the expected format.

### Required Files

```
data/your_dataset/
├── Da.csv    # Accepts (with y column, binary 0/1)
├── Dr.csv    # Rejects (NO y column)
└── H.csv     # Holdout (with y column, binary 0/1)
```

### Format Requirements

| File | y column | Description |
|------|----------|-------------|
| `Da.csv` | Required | Accepted applicants with observed outcomes |
| `Dr.csv` | Not present | Rejected applicants (labels unknown) |
| `H.csv` | Required | Representative holdout for evaluation |

**Important:**
- All files must have identical feature columns (excluding `y`)
- The `y` column must be binary: 0 = good (non-default), 1 = bad (default)
- `Dr.csv` must NOT have a `y` column (labels are unknown for rejects)
- Sanity check: `bad_rate(Da) < bad_rate(H)` (accepts should have lower bad rate than population)

### Usage

```bash
python scripts/run_experiment_1.py --data-dir data/your_dataset
python scripts/run_experiment_2.py --data-dir data/your_dataset
```

---

## Using Your Own Model

You can integrate custom models by implementing the required interface.

### Required Interface

```python
class MyModel:
    def __init__(self, cfg):
        """Initialize model with configuration object."""
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on labeled data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary labels of shape (n_samples,), where 1=bad, 0=good.
        """
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of default (P(y=1)) for each sample.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1D array of shape (n_samples,) with P(y=1|X) estimates.
        """
        ...
```

### Example: Sklearn Wrapper

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestModel:
    def __init__(self, cfg):
        self._model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]
```

---

## Figure 2: Loss Due to Sampling Bias

| Panel | Title | Description |
|-------|-------|-------------|
| (a) | Bias in Data | Feature distributions (Population vs Accepts vs Rejects) |
| (b) | Bias in Model | LR surrogate coefficients on XGB predictions |
| (c) | Bias in Predictions | P(BAD) score distributions |
| (d) | Impact on Evaluation | ABR over iterations (Bayesian vs Accepts-only) |
| (e) | Impact on Training | ABR over iterations (BASL vs Accepts-only) |

### Run Options

```bash
# Quick test (1 seed, ~3 min)
python scripts/run_figure2_unified.py --config configs/figure2_unified.yaml --n-seeds 1

# Full run (10 seeds, default)
python scripts/run_figure2_unified.py --config configs/figure2_unified.yaml
```

---

## Project Structure

```
├── configs/
│   ├── experiment_1.yaml          # Exp I parameters
│   ├── experiment_2.yaml          # Exp II parameters
│   ├── synthetic_generator.yaml   # Data generation parameters
│   └── figure2_unified.yaml       # Figure 2 parameters
├── data/synthetic/                # Generated data bundles (gitignored)
├── experiments/                   # Experiment outputs (gitignored)
├── notebooks/
│   ├── figure_2.ipynb             # Figure 2 visualization
│   ├── table2_table3.ipynb        # Tables 2 & 3 generation
│   └── *_diagnostics.ipynb        # Analysis notebooks
├── scripts/
│   ├── generate_synthetic_like_real.py  # Data generation
│   ├── run_experiment_1.py        # Exp I runner
│   ├── run_experiment_2.py        # Exp II runner
│   ├── verify_experiments.py      # Results verification
│   └── run_figure2_unified.py     # Figure 2 script
└── src/
    ├── basl/                      # BASL reject inference
    ├── data/                      # Data backends & splitters
    ├── evaluation/                # Metrics & Bayesian evaluation
    ├── experiments/               # Modular experiment runners
    ├── io/                        # Data generation & acceptance loop
    └── models/                    # XGBoost wrapper
```
