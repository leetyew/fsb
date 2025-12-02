# Fighting Sampling Bias

Implementation of the paper "Fighting Sampling Bias: A Framework for Training and Evaluating Credit Scoring Models".

## Requirements

- Python 3.8.15

## Directory Structure

```
.
├── configs/               # YAML configuration files
│   ├── acceptance_loop.yaml
│   ├── basl.yaml
│   ├── experiment.yaml
│   ├── model_xgboost.yaml
│   └── synthetic_data.yaml
├── docs/                  # Documentation
│   └── architecture.md    # Detailed architecture and paper references
├── experiments/           # Experiment outputs (auto-generated)
├── notebooks/             # Jupyter notebooks for analysis
│   └── result_comparison.ipynb
├── scripts/               # Experiment runners
│   └── run_experiment.py  # Main experiment script
└── src/                   # Source code
    ├── basl/              # BASL implementation
    ├── evaluation/        # Metrics and Bayesian evaluation
    ├── io/                # Data generation and acceptance loop
    ├── models/            # XGBoost and Logistic Regression
    └── preprocessing/     # Feature pipeline
```

## Running Experiments

### Main Experiment (Figure 2, Tables C.3-C.4)

Runs baseline vs BASL comparison across multiple seeds with iteration tracking for Figure 2.

```bash
# Full experiment (100 seeds, as in paper)
python scripts/run_experiment.py

# Quick test (3 seeds)
python scripts/run_experiment.py --n-seeds 3

# Single seed for debugging
python scripts/run_experiment.py --seed 42
```

**CLI Overrides** (all have defaults in `configs/experiment.yaml`):
- `--seed`: Run a single specific seed
- `--n-seeds`: Number of random seeds (default: 100)
- `--start-seed`: Starting seed value (default: 0)
- `--n-periods`: Acceptance loop periods
- `--track-every`: Metric tracking interval (default: 10)
- `--name`: Experiment name suffix

### Output

Results are saved to `experiments/experiment_{timestamp}/`:
- `trial_seed{N}.json`: Per-trial results with baseline and BASL histories
- `aggregated.json`: Aggregated metrics across all trials (mean, std)
- `config.json`: Experiment configuration

Each trial contains iteration history with three evaluation types:
- **Oracle**: Metrics on external holdout with true labels
- **Accepts**: Metrics on accepts only (biased)
- **Bayesian**: Metrics using coin flip pseudo-labeling

## Configuration

All parameters are configured via YAML files in `configs/`. Key files:

| File | Description |
|------|-------------|
| `experiment.yaml` | Experiment settings (n_seeds, track_every) |
| `acceptance_loop.yaml` | Loop parameters (n_periods, batch_size, accept_rate) |
| `basl.yaml` | BASL hyperparameters (filtering, labeling) |
| `bayesian_eval.yaml` | MC sampling params (j_min, j_max, epsilon) |
| `synthetic_data.yaml` | Data generation (bad_rate, n_features) |
| `model_xgboost.yaml` | XGBoost model parameters |

## Plotting Results

Use the notebook `notebooks/result_comparison.ipynb` to visualize:
- Table C.3: Evaluation method comparison (Accepts vs Bayesian)
- Table C.4: BASL improvement over baseline
- Figure 2: Metrics over iterations (all 5 panels)

## Implementation Notes

See `docs/architecture.md` for:
- Paper algorithm details (Algorithm 1, Table E.9)
- Implementation simplifications and trade-offs
- Difference between paper's MC sampling and our approach
