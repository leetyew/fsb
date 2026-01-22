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

## Project Structure

```
├── configs/figure2_unified.yaml   # Figure 2 parameters
├── environment.yml                # Conda environment
├── experiments/                   # Output data (gitignored)
├── notebooks/figure_2.ipynb       # Figure 2 visualization
├── scripts/run_figure2_unified.py # Main experiment script
└── src/
    ├── basl/                      # BASL reject inference
    ├── evaluation/                # Metrics & Bayesian evaluation
    ├── io/                        # Data generation & acceptance loop
    └── models/                    # XGBoost wrapper
```
