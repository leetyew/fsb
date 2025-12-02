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
│   ├── run_multiseed_experiment.py  # Main experiment (Table C.4)
│   └── run_iteration_tracking.py    # Iteration tracking (Figure 2)
└── src/                   # Source code
    ├── basl/              # BASL implementation
    ├── evaluation/        # Metrics and Bayesian evaluation
    ├── io/                # Data generation and acceptance loop
    ├── models/            # XGBoost and Logistic Regression
    └── preprocessing/     # Feature pipeline
```

## Running Experiments

### Multi-Seed Experiment (Table C.3, C.4)

Runs baseline vs BASL comparison across multiple seeds with oracle, accepts-based, and Bayesian evaluation metrics.

```bash
# Full experiment (100 seeds, as in paper)
python scripts/run_multiseed_experiment.py

# Quick test (3 seeds)
python scripts/run_multiseed_experiment.py --n-seeds 3

# Resume interrupted experiment
python scripts/run_multiseed_experiment.py --resume experiments/experiment_YYYYMMDD_HHMMSS
```

**CLI Overrides** (all have defaults in `configs/experiment.yaml`):
- `--n-seeds`: Number of random seeds
- `--start-seed`: Starting seed value
- `--n-periods`: Acceptance loop periods
- `--n-bayes-samples`: Monte Carlo samples for Bayesian evaluation
- `--name`: Experiment name suffix

### Iteration Tracking (Figure 2)

Tracks metrics per-iteration to generate all 5 panels of Figure 2.

```bash
# Full experiment (100 seeds)
python scripts/run_iteration_tracking.py

# Quick test
python scripts/run_iteration_tracking.py --n-seeds 3 --track-every 50
```

**CLI Overrides**:
- `--n-seeds`, `--start-seed`, `--n-periods`, `--n-bayes-samples`: Same as above
- `--track-every`: Metric tracking interval (default: 10)

### Output

Results are saved to `experiments/{experiment_name}/`:
- `results.json`: Full results with all trials
- `summary.json`: Aggregated metrics
- `config.json`: Experiment configuration
- `trials.jsonl`: Incremental trial results (for resume)

## Configuration

All parameters are configured via YAML files in `configs/`. Key files:

| File | Description |
|------|-------------|
| `experiment.yaml` | Experiment settings (n_seeds, n_bayes_samples, track_every) |
| `acceptance_loop.yaml` | Loop parameters (n_periods, batch_size, accept_rate) |
| `basl.yaml` | BASL hyperparameters (filtering, labeling) |
| `synthetic_data.yaml` | Data generation (bad_rate, n_features) |
| `model_xgboost.yaml` | XGBoost model parameters |

## Plotting Results

Use the notebook `notebooks/result_comparison.ipynb` to visualize:
- Table C.3: Evaluation method comparison (Accepts vs Bayesian)
- Table C.4: BASL improvement over baseline
- Figure 2: Metrics over iterations (all 5 panels)
