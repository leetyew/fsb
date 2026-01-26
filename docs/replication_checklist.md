# replication_checklist.md
# Replication Checklist for  
**“Fighting Sampling Bias: A Framework for Training and Evaluating Credit Scoring Models”**

This is a **strict, step‑by‑step, implementation‑ready** checklist.  
If you follow every step, you will faithfully replicate the paper’s experiments.

---

# 0. Environment Setup

- [ ] Python ≥ 3.9  
- [ ] Install: numpy, scipy, pandas, scikit-learn, xgboost, matplotlib  
- [ ] Fix random seeds for reproducibility  
- [ ] Ensure consistent train/validation splits across all experiments  

---

# 1. Synthetic Data Generation (Experiments I, II, IV)

### 1.1 Generate x₀ (informative feature)
- [ ] Choose population size N ≥ 50,000  
- [ ] Sample:
  - Good borrowers: x₀ ~ N(0, 1)  
  - Bad borrowers: x₀ ~ N(μ_b, 1), usually μ_b = 2  
- [ ] Assign class priors (e.g., 20–25% bad)

### 1.2 Generate labels
- [ ] Use logistic rule:
  - p(y=1 | x₀) = σ(a x₀ + b)
- [ ] Sample y ~ Bernoulli(p)

### 1.3 Add noise features (optional)
- [ ] Add K Gaussian noise features, K ∈ [10, 100]  
- [ ] Ensure noise is independent from labels  

### 1.4 Create unbiased holdout set
- [ ] Sample from the same population distribution  
- [ ] Size ≥ 10,000  
- [ ] Store labels for Oracle evaluation  

---

# 2. Acceptance Policy (Bank Policy Simulation)

### 2.1 Acceptance rule
- [ ] Compute acceptance threshold T = percentile(x₀, α)  
- [ ] Standard α = 0.15  
- [ ] Accept = x₀ ≤ T  
- [ ] Reject = x₀ > T  

### 2.2 Label visibility
- [ ] Accept labels are revealed  
- [ ] Reject labels are **never revealed**  

### 2.3 Store datasets
- [ ] D_a = accepts with labels  
- [ ] D_r = rejects unlabeled  

---

# 3. Experiment I — Reliability of Evaluation Methods (Figure 2)

Goal: Compare **Accepts-only**, **DR**, **Reweighting**, **Bayesian** estimates.

### 3.1 Train accepts-only weak model
- [ ] Model = Logistic Regression  
- [ ] Training data = D_a  

### 3.2 Compute estimators
- [ ] Accepts-only ABR  
- [ ] DR estimate (with smoothed propensities)  
- [ ] Reweighting estimate  
- [ ] Bayesian estimate using main model predictions on rejects  

### 3.3 Compute Oracle ABR
- [ ] Evaluate on unbiased holdout  

### 3.4 Plot (Figure 2)
- [ ] Show gap between estimators and Oracle  

---

# 4. Experiment II — BASL Training Effectiveness (Figures 3, 4, Table 2)

Goal: Compare BASL vs all baselines.

### 4.1 Prepare datasets
- [ ] Use D_a (labeled) + D_r (unlabeled)  
- [ ] Create accept-only validation set  

### 4.2 Implement BASL loop

#### Step A — Filtering
- [ ] Train Isolation Forest on D_a  
- [ ] Select ρ proportion (0.1–0.3) of D_r  

#### Step B — Weak learner labeling
- [ ] Train LR on D_a  
- [ ] Predict p(y=1|x) for filtered rejects  
- [ ] Assign pseudo-label = 1 if p > 0.5  

#### Step C — Strong learner training
- [ ] Train XGBoost on D_a ∪ pseudo-labeled rejects  

#### Step D — Early stopping
- [ ] Evaluate strong learner on accept-validation  
- [ ] Stop if performance decreases  

### 4.3 Baselines to implement
- [ ] Accepts-only LR  
- [ ] Accepts-only XGB  
- [ ] Naive self-learning  
- [ ] DR training  
- [ ] Reweighting  
- [ ] BASL  

### 4.4 Compute metrics
- [ ] AUC on holdout  
- [ ] ABR on holdout  

### 4.5 Figures/Table
- [ ] Figure 3: AUC/ABR performance comparison  
- [ ] Figure 4: Bias across feature range  
- [ ] Table 2: Summary metrics  

---

# 5. Experiment III — Real-World Dataset Evaluation (Figure 5, Table 3)

Cannot be perfectly replicated without the proprietary data, but structure is:

### 5.1 Load real dataset
- [ ] Accepts: labeled  
- [ ] Rejects: unlabeled  
- [ ] Historical outcomes as Oracle set  

### 5.2 Run the same BASL steps as Experiment II
- [ ] Isolation forest  
- [ ] Weak learner LR  
- [ ] Strong learner XGB  
- [ ] Iteration + early stopping  

### 5.3 Compute metrics
- [ ] AUC  
- [ ] ABR  

### 5.4 Generate outputs
- [ ] Table 3: numeric performance  
- [ ] Figure 5: calibration / score distributions  

---

# 6. Experiment IV — Sensitivity Analysis (Figures 6–8)

### 6.1 Vary filtering strength ρ
- [ ] ρ ∈ {0.05, 0.1, 0.2, 0.3, 0.5}  
- [ ] Rerun BASL for each  

### 6.2 Vary weak learner noise
- [ ] Inject random flips in LR labels  
- [ ] Test noise = 0%, 10%, 20%, 40%, 50%  

### 6.3 Vary acceptance rate α
- [ ] α ∈ {0.05, 0.1, 0.15, 0.2, 0.25}  
- [ ] Regenerate D_a, D_r for each  
- [ ] Rerun BASL  

### 6.4 Vary reject pool size
- [ ] Sizes ∈ {10k, 20k, 50k, 100k}  

### 6.5 Produce figures
- [ ] Figure 6: effect of ρ  
- [ ] Figure 7: effect of weak learner noise  
- [ ] Figure 8: effect of α  

---

# 7. Final Verification

### 7.1 Visual alignment
- [ ] Curves and tables should resemble the paper’s figures  
- [ ] BASL should outperform baselines consistently  

### 7.2 Key expected behaviors
- [ ] Accepts-only model is biased  
- [ ] Bayesian evaluation is the most accurate estimator  
- [ ] BASL reduces sampling bias  
- [ ] Filtering (ρ) has strong impact  
- [ ] Weak learner does not need to be perfect  
- [ ] BASL robustness shown across α  

---

# End of File — replication_checklist.md
