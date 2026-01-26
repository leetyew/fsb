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
# Generate data bundle (creates data/synthetic/<run_id>/)
python scripts/generate_synthetic_like_real.py

# Or with custom config/seed
python scripts/generate_synthetic_like_real.py --config configs/synthetic_generator.yaml --seed 42
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

Compares biased vs Bayesian evaluation methods across 100 replicates.

```bash
# Full run (4 folds × 25 bootstraps = 100 replicates)
python scripts/run_experiment_1.py --data-dir data/synthetic/<run_id>

# Quick test (fewer replicates)
python scripts/run_experiment_1.py --data-dir data/synthetic/<run_id> --n-folds 2 --n-bootstraps 2
```

**Output:** `experiments/exp1_<timestamp>/exp1_results.csv`

### Step 3: Run Experiment II (Table 3 - Training Comparison)

Compares accepts-only vs BASL training methods across 100 replicates.

```bash
# Full run (4 folds × 25 bootstraps = 100 replicates)
python scripts/run_experiment_2.py --data-dir data/synthetic/<run_id>

# Quick test (fewer replicates)
python scripts/run_experiment_2.py --data-dir data/synthetic/<run_id> --n-folds 2 --n-bootstraps 2
```

**Output:** `experiments/exp2_<timestamp>/exp2_results.csv`

### Step 4: Verify Results

```bash
python scripts/verify_experiments.py
```

### Step 5: Generate Tables

```bash
jupyter notebook notebooks/table2_table3.ipynb
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
