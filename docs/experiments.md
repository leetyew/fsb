# Fighting Sampling Bias — Clean Experiment Overview & Replication README

This document provides a **clean, paper-faithful experiment structure**, explicitly distinguishing:

* **Core experiments** (hypotheses tested)
* **Diagnostic figures** (illustrative, not separate experiments)
* **Ablation / sensitivity analyses** (not new experiments)

It is written to be **reviewer-proof**: every figure and table is mapped to exactly one experiment, with no ambiguity.

---

# 1. Canonical Experiment Structure (Paper-Faithful)

The paper contains **two core synthetic experiments**, plus **one real-world validation**, plus **sensitivity analyses**.

No additional standalone experiments are introduced.

| ID                 | Name                                | Hypothesis Tested                                                                            |
| ------------------ | ----------------------------------- | -------------------------------------------------------------------------------------------- |
| **Experiment I**   | Evaluation under Sampling Bias      | Can Bayesian evaluation reduce bias in performance estimation when only accepts are labeled? |
| **Experiment II**  | Training under Sampling Bias (BASL) | Can BASL reduce training bias and improve predictive performance?                            |
| **Experiment III** | Real-World Validation               | Does BASL generalize to real credit data?                                                    |

Sensitivity analyses reuse Experiment II and **do not constitute new experiments**.

---

# 2. Diagnostics vs Core Results (Critical Clarification)

The paper deliberately mixes **diagnostic visualizations** with **core experimental results**.

These must not be misinterpreted as separate experiments.

## 2.1 Diagnostic Figures (Illustrative Only)

These figures **do not test new hypotheses**. They visualize *how sampling bias propagates* through data, model, and predictions.

**Important clarification (paper‑faithful and explicit):** Panels (b) and (c) include **BASL‑trained models**, but they are **not results of Experiment II**. They are **diagnostic bridge panels** whose purpose is to *visually connect Experiment I (bias diagnosis) to Experiment II (bias‑aware training)*.

* They reuse the **training outcome** of BASL from Experiment II
* They do **not** evaluate BASL quantitatively
* They introduce **no new hypothesis**
* They must **not** be counted as Experiment II results

In other words: **BASL appears in panels (b) and (c) only for illustration, not for evaluation**.

| Figure | Panel | Purpose                                    | Models Shown                                | Experiment Context |
| ------ | ----- | ------------------------------------------ | ------------------------------------------- | ------------------ |
| Fig. 2 | (a)   | Bias in data distribution                  | Population / Accepts / Rejects              | Exp I (diagnostic) |
| Fig. 2 | (b)   | Bias in model parameters (LR coefficients) | Accepts-only LR vs Oracle LR vs **BASL-LR** | Exp I → II bridge  |
| Fig. 2 | (c)   | Bias in predictions (LR scores)            | Accepts-only LR vs Oracle LR vs **BASL-LR** | Exp I → II bridge  |

Properties:

* Single snapshot (not averaged)
* No iterative retraining curves
* Logistic regression used for interpretability
* **BASL appears as a diagnostic comparator, not as a core experiment result**

---

## 2.2 Core Evaluation Result

| Figure | Panel | Meaning                | Experiment                     |
| ------ | ----- | ---------------------- | ------------------------------ |
| Fig. 2 | (d)   | Bias in ABR evaluation | **Experiment I (core result)** |

This panel **is** the result of Experiment I.

---

## 2.3 Training Dynamics (Bridge Figure)

| Figure | Panel | Meaning                       | Experiment                            |
| ------ | ----- | ----------------------------- | ------------------------------------- |
| Fig. 2 | (e)   | Training bias over iterations | **Experiment II (training dynamics)** |

Although visually grouped in Figure 2, panel (e):

* Belongs to Experiment II
* Is shown early for narrative continuity
* Must NOT be generated from Experiment I code

---

# 3. Experiment I — Evaluation under Sampling Bias (Core)

## 3.1 Objective

Evaluate **performance estimation bias** when models are trained on accepts-only data.

No reject inference is used for training.

---

## 3.2 Setup

* Synthetic population (Section 4.1 of paper)
* Acceptance policy: lowest (lpha = 0.15) by (x_0)
* Labels observed **only for accepts**

---

## 3.3 Model Used

* **XGBoost scorecard** (non-parametric)
* Trained only on accepts

(Logistic regression is used *only* for diagnostic Figures 2b–2c.)

---

## 3.4 Evaluation Methods Compared

| Method              | Uses Rejects? | Purpose                    |
| ------------------- | ------------- | -------------------------- |
| Accepts-only        | No            | Naïve baseline             |
| Reweighting         | Yes           | Covariate shift correction |
| Doubly Robust       | Yes           | Off-policy estimator       |
| Bayesian Evaluation | Yes           | Proposed method            |

---

## 3.5 Core Outputs (Must Reproduce)

| Artifact  | Status          |
| --------- | --------------- |
| Fig. 2(d) | **Core result** |
| Table 1   | **Core result** |

---

# 4. Experiment II — Training under Sampling Bias (BASL)

## 4.1 Objective

Evaluate whether **BASL reduces training bias** and improves predictive performance.

---

## 4.2 Setup

* Same synthetic population
* Same acceptance policy
* Iterative training loop

---

## 4.3 BASL Algorithm (High-Level)

Each iteration:

1. Filter rejects close to accepts (Isolation Forest)
2. Pseudo-label using weak LR
3. Retrain strong XGBoost model
4. Early stopping on validation accepts

---

## 4.4 Core Outputs

| Artifact | Status          |
| -------- | --------------- |
| Fig. 3   | **Core result** |
| Fig. 4   | **Core result** |
| Table 2  | **Core result** |

---

## 4.5 Training Diagnostics

| Artifact  | Meaning           |
| --------- | ----------------- |
| Fig. 2(e) | Training dynamics |

---

# 5. Sensitivity & Ablation Studies (NOT Experiments)

These analyses reuse Experiment II with parameter perturbations.

| Figure | Parameter Varied        |
| ------ | ----------------------- |
| Fig. 6 | Filtering rate (        |
| ho)    |                         |
| Fig. 7 | Weak learner noise      |
| Fig. 8 | Acceptance rate (lpha) |

No new hypotheses are introduced.

---

# 6. Experiment III — Real-World Validation

* Same BASL pipeline
* Real bank data
* Acceptance policy unknown

Outputs:

* Table 3
* Figure 5

---

# 7. Replication README (Reviewer-Proof)

## What Counts as an Experiment

> An experiment tests **one new hypothesis** under a fixed setup.

Accordingly:

* Evaluation bias → Experiment I
* Training bias → Experiment II
* Real data generalization → Experiment III

---

## What Does NOT Count as an Experiment

* Diagnostic plots
* Training curves
* Sensitivity analyses

These **must not** be split into new experiments.

---

## Hard Replication Rules

1. Do not train on reject labels
2. Do not use XGBoost for Figure 2(b)
3. Do not generate Fig. 2(e) from Experiment I
4. Always log oracle labels **offline only**

---

## Reviewer-Facing Summary (Safe to Quote)

> “The paper contains two core synthetic experiments: one evaluating bias in performance estimation, and one evaluating bias-aware training. All other figures are either diagnostic visualizations or ablation studies built on these experiments.”

---

# End of File
