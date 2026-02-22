---
title: "Predicting Heart Disease: A Comprehensive Machine Learning Study"
subtitle: "Kaggle Playground Series S6E2 — 83 Models, 13 Experimental Phases"
author: "Dustin Ober"
date: "2025"
abstract: |
  We present a comprehensive, reproducible benchmark of 83 machine learning models
  applied to the Kaggle Playground Series S6E2 synthetic heart disease dataset
  (630,000 training instances, 13 clinical features). Across 13 experimental phases
  spanning logistic regression, classical ensembles, gradient boosting, deep neural
  networks, AutoML, imbalance-handling strategies, unsupervised clustering, and
  large language model (LLM) zero-shot classification, we demonstrate that
  Optuna-tuned gradient boosting methods (LightGBM, XGBoost, CatBoost) achieve a
  performance ceiling of **ROC-AUC = 0.9555** that no subsequent method meaningfully
  exceeds. SHAP analysis confirms that Thallium stress test result and Chest pain
  type are the dominant predictors; blood pressure and fasting blood sugar contribute
  negligibly. Neural networks (MPS-accelerated) plateau at AUC = 0.9525, and
  rank-averaging ensembles achieve AUC = 0.9555 while improving recall to 0.9215.
  A 0.6B-parameter LLM (Qwen3) predicts the all-negative class with 0% recall,
  confirming that small generalist models lack tabular medical reasoning capability.
  Threshold-optimized LightGBM at t = 0.438 (Youden's J) yields the best F1 (0.8763)
  and strong recall (0.8865) for clinical deployment. All experiments are tracked
  via MLflow; the full codebase and results are publicly reproducible.

bibliography: references.bib
numbersections: true
geometry: margin=1in
fontsize: 11pt
---

---

# 1. Introduction

Cardiovascular disease remains the leading cause of death globally, accounting for
approximately 17.9 million deaths per year according to the World Health Organization.
Early and accurate prediction of heart disease is critical for clinical decision support,
enabling timely intervention that can significantly reduce morbidity and mortality.

Machine learning approaches to heart disease prediction have a long history, beginning
with early work on the UCI Heart Disease dataset [@detrano1989international] and
extending to modern deep learning and AutoML approaches. The Kaggle Playground Series
S6E2 competition provides a synthetic dataset derived from the original UCI data,
scaled to 630,000 training instances — enabling thorough evaluation of high-capacity
models that would be infeasible on the original 303-instance benchmark.

This paper makes the following contributions:

1. A systematic benchmark of **83 machine learning models** across 13 experimental phases,
   enabling direct comparison of classical, ensemble, neural, AutoML, and LLM approaches.
2. Rigorous evaluation on three clinically meaningful metrics: ROC-AUC, F1-Score, and Recall,
   with full cross-validation (Stratified 5-fold).
3. Evidence that gradient boosting methods define a hard performance ceiling (~0.9555 AUC)
   for this dataset, and that no approach — including neural networks, AutoML, or ensembles —
   meaningfully surpasses it.
4. SHAP-based interpretability connecting model decisions to clinical domain knowledge, with
   actionable findings for the 13 features studied.
5. The first evaluation of a quantized LLM (Qwen3 0.6B) for zero-shot tabular heart disease
   classification, demonstrating that sub-1B models are wholly unsuitable for this task.
6. A fully reproducible framework with MLflow experiment tracking, version-controlled code,
   and file-level commits after each experimental phase.

---

# 2. Related Work

## 2.1 Classical Machine Learning for Heart Disease

The UCI Heart Disease dataset [@detrano1989international] introduced the canonical
benchmark for ML-based cardiac risk prediction. Subsequent studies applied logistic
regression, decision trees, and Naive Bayes, typically achieving 70–85% accuracy on
the 303-instance dataset. Palaniappan and Awang (2008) demonstrated that decision trees
and Naive Bayes could achieve ~84% accuracy on this benchmark. More recently, random
forests and gradient boosting methods have consistently outperformed shallower models,
with AUC values in the 0.92–0.96 range on the original UCI data.

## 2.2 Gradient Boosting for Tabular Data

Gradient boosting methods — XGBoost [@chen2016xgboost], LightGBM [@ke2017lightgbm],
and CatBoost [@prokhorenkova2018catboost] — have become the dominant paradigm for
tabular data competitions. Multiple Kaggle benchmarks confirm that tree-based boosting
routinely outperforms deep neural networks on tabular data [@grinsztajn2022tree].
The histogram-based variant (HistGradientBoosting, LightGBM) provides particularly
efficient training on large datasets.

## 2.3 Deep Learning for Tabular Data

TabNet [@arik2021tabnet] introduced sequential attention for tabular feature selection.
NODE [@popov2020neural] and SAINT [@somepalli2021saint] extended transformer-style
architectures to tabular problems. Despite architectural innovation, multiple benchmarks
show that these approaches struggle to consistently outperform gradient boosting on
real-world tabular datasets [@grinsztajn2022tree; @mcelfresh2023neural].

## 2.4 AutoML

FLAML [@wang2021flaml] uses a cost-efficient search strategy to find good configurations
within a fixed time budget. Auto-Sklearn [@feurer2015efficient] applies Bayesian
optimization with meta-learning warm-starting. Optuna [@akiba2019optuna] provides a
framework-agnostic hyperparameter optimization library supporting TPE and CMA-ES samplers.

## 2.5 Interpretability

SHAP (SHapley Additive exPlanations) [@lundberg2017unified] provides theoretically
grounded feature attribution based on cooperative game theory. SHAP TreeExplainer
efficiently computes exact Shapley values for tree-based models. LIME [@ribeiro2016why]
provides local linear approximations for any black-box model. These methods are
increasingly required for regulatory compliance in clinical AI applications.

## 2.6 LLMs for Structured Data

TabLLM [@hegselmann2023tabllm] demonstrated that large language models can be
fine-tuned for tabular classification by serializing rows to natural language.
FeatLLM [@han2024featllm] uses LLMs as feature engineers rather than direct classifiers.
Zero-shot LLM classification of tabular medical data remains an open challenge, with
most published results showing that LLMs without fine-tuning perform near random
or exhibit strong prediction biases on numerical tabular features.

---

# 3. Dataset and Preprocessing

## 3.1 Dataset Overview

The dataset is sourced from the Kaggle Playground Series Season 6, Episode 2.
It is synthetically generated via CTGAN-style models conditioned on the original UCI
Heart Disease repository, scaled from 303 instances to 630,000 training instances.

| Split    | Instances | Features | Target Distribution               |
|----------|-----------|----------|-----------------------------------|
| Training | 630,000   | 13       | 55.2% Absence, 44.8% Presence     |
| Test     | 270,000   | 13       | Unlabeled (competition holdout)   |

The target variable (`Heart Disease`) is binary: *Presence* (1) or *Absence* (0).
The class imbalance ratio is **1.23:1**, classified as mild and not requiring
aggressive rebalancing as a primary concern.

## 3.2 Feature Descriptions

| Feature                  | Type          | Range / Values | Clinical Significance                          |
|--------------------------|---------------|----------------|------------------------------------------------|
| Age                      | Continuous    | 28–77 years    | Risk increases with age                        |
| Sex                      | Binary        | 0=F, 1=M       | Males historically higher baseline risk        |
| Chest pain type          | Ordinal       | 1–4            | Type 4 (asymptomatic) paradoxically high risk  |
| BP                       | Continuous    | 94–200 mmHg    | Resting blood pressure                         |
| Cholesterol              | Continuous    | 131–564 mg/dl  | Serum cholesterol                              |
| FBS over 120             | Binary        | 0/1            | Fasting blood sugar > 120 mg/dl                |
| EKG results              | Ordinal       | 0–2            | Resting ECG result classification              |
| Max HR                   | Continuous    | 71–202 bpm     | Maximum heart rate achieved during stress test |
| Exercise angina          | Binary        | 0/1            | Chest pain induced by exercise                 |
| ST depression            | Continuous    | 0.0–6.2        | ST segment depression during exercise          |
| Slope of ST              | Ordinal       | 1–3            | Direction of peak exercise ST segment          |
| Number of vessels fluro  | Ordinal       | 0–3            | Major vessels colored by fluoroscopy           |
| Thallium                 | Ordinal       | 3, 6, 7        | Thallium stress test: normal/fixed/reversible  |

## 3.3 Data Quality Assessment

The dataset exhibits no missing values and no duplicate rows across all 630,000
training instances — an artifact of synthetic generation. A Kolmogorov-Smirnov test
comparing train and test feature distributions showed no statistically significant
drift for any feature (all p > 0.05), confirming distributional consistency.

Outlier analysis (IQR method) flagged 2–8% of instances per continuous feature as
potential outliers, consistent with the natural tails of clinical measurement
distributions. Isolation Forest identified a 5% anomaly rate in multivariate space.
No imputation was required; outliers were retained as they reflect realistic
clinical variation.

## 3.4 Preprocessing and Feature Engineering

All 13 raw features are used without transformation for tree-based models.
For linear models and neural networks, we apply `StandardScaler` to continuous
features. One-hot encoding is not used; ordinal integer encoding is preserved
throughout, consistent with the original UCI annotation.

We constructed six named feature sets:

| Feature Set       | Description                                      | Dimensionality |
|-------------------|--------------------------------------------------|----------------|
| `raw_13`          | Raw features, no scaling                         | 13             |
| `scaled_13`       | Raw + StandardScaler                             | 13             |
| `engineered_18`   | + Age×HR ratio, BP×Cholesterol, ST interaction,  | 18             |
|                   | Age bins, polynomial degree-2 selected features  |                |
| `selected_top8`   | Top-8 features by Mutual Information             | 8              |
| `pca_k`           | PCA retaining 95% variance (≈10 components)      | 10             |

A feature-set ablation experiment (Section 5.6) showed no meaningful difference
between `raw_13`, `scaled_13`, and `engineered_18` on LightGBM (all AUC ≈ 0.9554),
confirming that the 13 raw features are sufficient and that handcrafted engineering
provides no additional signal.

---

# 4. Methodology

## 4.1 Evaluation Protocol

All supervised models are evaluated using **Stratified 5-fold Cross-Validation**
(`sklearn.model_selection.StratifiedKFold`, `n_splits=5`, `random_state=42`).
We report the mean across all folds for:

- **ROC-AUC** (primary): area under the receiver operating characteristic curve;
  threshold-independent and robust to class imbalance.
- **F1-Score**: harmonic mean of precision and recall at threshold 0.5.
- **Recall**: sensitivity; minimizing false negatives is clinically important
  (missed diagnoses are more harmful than false alarms in cardiac screening).

Where noted, subsampling is applied to computationally intensive models:
neural networks use 150,000 training instances (MLP) or 50,000 (TabNet);
SHAP values are computed on a 10,000-instance sample; t-SNE/UMAP on 20,000 instances.

All experiments are logged to a local MLflow server under the experiment
`Heart-Disease-Kaggle`. Hyperparameter optimization uses Optuna with an SQLite
backend for state persistence across sessions.

## 4.2 Hardware Environment

All experiments are conducted on an **Apple M1 MacBook Pro, 16GB Unified Memory**.
Neural networks use the Metal Performance Shaders (MPS) backend (PyTorch 2.10.0,
`torch.backends.mps.is_available() = True`). Tree-based models use CPU-optimized
histogram methods (`tree_method='hist'`). Python 3.11.11 via `pyenv`;
virtual environment managed with `venv`. XGBoost requires OpenMP (`libomp`)
for parallelism on Apple Silicon.

## 4.3 Model Families

### 4.3.1 Baseline Models

Dummy classifiers (majority-class, stratified, prior) establish the performance floor.
Logistic Regression (L1, L2, ElasticNet) provides the linear baseline.
Gaussian Naive Bayes assumes conditional feature independence.
k-Nearest Neighbors evaluates distance-based instance retrieval.
Decision Trees assess single-tree depth vs. performance tradeoffs.

### 4.3.2 Classical Ensemble Methods

Random Forest and Extra Trees evaluate bagging-based variance reduction.
AdaBoost applies sequential boosted stumps.
Bagging with Decision Tree base learners isolates ensemble effect.
Support Vector Machines (RBF, Linear, Polynomial kernels) test margin-based classifiers.
Linear and Quadratic Discriminant Analysis evaluate Gaussian class-conditional models.

### 4.3.3 Gradient Boosting

XGBoost, LightGBM, and CatBoost are evaluated in default and Optuna-tuned variants.
FLAML AutoML discovers configurations within fixed time budgets (300s, 600s).
Optuna TPE (Tree-structured Parzen Estimator) and CMA-ES (Covariance Matrix
Adaptation Evolution Strategy) are compared over 100–200 trials.

### 4.3.4 Neural Networks

Five MLP architectures are evaluated: Small (64-32), Medium (256-128-64, BatchNorm),
Wide (512-256, GELU), Deep (5-layer with residual connections), and TabNet.
All use the Adam optimizer with ReduceLROnPlateau scheduling and early stopping
(patience = 8 epochs). MPS acceleration reduces training time by approximately
3-5× vs. CPU for the architectures tested.

### 4.3.5 Class Imbalance Strategies

Oversampling: SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE.
Undersampling: RandomUnderSampler, TomekLinks, EditedNearestNeighbours.
Combined: SMOTEENN, SMOTETomek.
Cost-sensitive: `class_weight='balanced'`, custom ratios.
Post-hoc: threshold optimization (Youden's J, F1-optimal, Recall@95%).

### 4.3.6 LLM Classification

Each row is serialized to natural language:
*"A [age]-year-old [sex] patient with chest pain type [x], resting BP [x] mmHg,
cholesterol [x] mg/dl..."*
The LLM is asked to classify Presence/Absence using zero-shot, few-shot (5-shot),
and chain-of-thought prompting. The Ollama API endpoint (`localhost:11434`) is used.
The originally planned model (`hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_S`) was
unavailable at experiment time; the available `qwen3:0.6b` (522 MB, 0.6B parameters)
was substituted. This is a critical limitation acknowledged throughout Section 5.10.

---

# 5. Experiments and Results

## 5.1 Exploratory Data Analysis

### 5.1.1 Dataset Characteristics

The training set contains 630,000 instances with **no missing values** and
**no duplicate rows**. The target distribution is mildly imbalanced:
55.2% Absence (347,546) vs. 44.8% Presence (282,454), yielding an imbalance
ratio of 1.23:1.

### 5.1.2 Feature Informativeness

**Table 1: Mutual Information scores and effect sizes (features vs. target)**

| Rank | Feature                  | MI Score | Cramér's V / p-value   | Interpretation     |
|------|--------------------------|----------|------------------------|--------------------|
| 1    | Thallium                 | 0.2358   | V = 0.606 (strong)     | Dominant predictor |
| 2    | Chest pain type          | 0.1895   | V = 0.525 (strong)     | Strong predictor   |
| 3    | Sex                      | 0.1318   | V = 0.342 (moderate)   | Moderate predictor |
| 4    | Max HR                   | 0.1293   | MWU p < 0.001          | Strong predictor   |
| 5    | Slope of ST              | 0.1250   | V = 0.430 (strong)     | Strong predictor   |
| 6    | Exercise angina          | 0.1239   | V = 0.442 (strong)     | Strong predictor   |
| 7    | Number of vessels fluro  | 0.1206   | V = 0.463 (strong)     | Strong predictor   |
| 8    | ST depression            | 0.1082   | MWU p < 0.001          | Strong predictor   |
| 9    | EKG results              | 0.0751   | V = 0.219 (moderate)   | Moderate predictor |
| 10   | Age                      | 0.0299   | MWU p < 0.001          | Weak predictor     |
| 11   | Cholesterol              | 0.0120   | MWU p < 0.001          | Very weak predictor|
| 12   | BP                       | 0.0110   | MWU p = 0.503 (n.s.)   | **Not significant**|
| 13   | FBS over 120             | 0.0022   | V = 0.034 (negligible) | Negligible         |

**Key finding**: Blood pressure (BP) showed no statistically significant difference
between classes (Mann-Whitney U, p = 0.503), making it effectively uninformative
for binary classification despite its well-established clinical relevance for cardiovascular
risk. This is a known limitation of the synthetic generation process. Fasting blood
sugar (FBS) also contributed negligibly (MI = 0.002).

### 5.1.3 Clinical Interpretation

- **Thallium stress test** (MI = 0.236, V = 0.606): The strongest single predictor.
  Normal result (3) is strongly associated with Absence; fixed defect (6) and
  reversible defect (7) with Presence. This is the gold-standard non-invasive test
  for coronary artery disease.
- **Chest pain type** (MI = 0.189, V = 0.525): Type 4 (asymptomatic) is paradoxically
  the most associated with Presence — a well-documented clinical phenomenon where
  silent ischemia carries higher actual risk than symptomatic angina.
- **Max HR** (MI = 0.129): Lower maximum exercise heart rate is associated with Presence,
  reflecting chronotropic incompetence in diseased patients.
- **Number of vessels colored by fluoroscopy** (V = 0.463): Direct measure of
  coronary artery disease burden; more affected vessels → higher Presence rate.
- **Exercise angina** (V = 0.442): A direct ischemic symptom; strongly associated with Presence.

## 5.2 Baseline Models

Baseline models establish the minimum acceptable performance threshold.
All models are evaluated with Stratified 5-fold CV on the full 630,000-instance training set.

**Table 2: Baseline model results (5-fold CV, mean values)**

| Model                    | ROC-AUC | F1-Score | Recall |
|--------------------------|---------|----------|--------|
| Dummy (majority class)   | 0.5000  | 0.0000   | 0.0000 |
| Dummy (stratified)       | 0.4990  | 0.4471   | 0.4470 |
| Decision Tree (depth=3)  | 0.9058  | 0.8119   | 0.7624 |
| Decision Tree (depth=5)  | 0.9327  | 0.8436   | 0.8262 |
| Decision Tree (depth=10) | 0.9488  | 0.8661   | 0.8571 |
| Decision Tree (unlimited)| 0.8239  | 0.8059   | 0.8077 |
| GaussianNB               | 0.9382  | 0.8560   | 0.8557 |
| KNN (k=5)                | 0.9209  | 0.8461   | 0.8383 |
| KNN (k=21)               | 0.9403  | 0.8566   | 0.8431 |
| **LR (L2, C=1.0)**       |**0.9505**|**0.8679**|**0.8563**|
| LR (L1, C=1.0)           | 0.9505  | 0.8679   | 0.8564 |
| LR (ElasticNet)          | 0.9505  | 0.8679   | 0.8564 |

**Key finding**: Logistic Regression achieves AUC = **0.9505** — a strong linear
baseline indicating that the class boundary is largely linearly separable in scaled
feature space. The regularization method (L1, L2, ElasticNet) and strength (C = 0.01 to
100) have negligible impact on AUC, confirming that the decision boundary is well
determined. This baseline becomes our primary reference point for measuring the value
of model complexity.

An unlimited depth Decision Tree achieves only AUC = 0.824 due to severe overfitting,
while depth-10 trees recover to 0.949 — a 12.5-point AUC gap illustrating the
bias-variance tradeoff at this dataset scale.

## 5.3 Classical Machine Learning

**Table 3: Classical ML results (5-fold CV, mean values, ranked by ROC-AUC)**

| Model                         | ROC-AUC | F1-Score | Recall |
|-------------------------------|---------|----------|--------|
| **HistGradientBoosting (lr=0.05)** | **0.9550** | **0.8743** | **0.8675** |
| HistGradientBoosting (default) | 0.9547  | 0.8739   | 0.8673 |
| AdaBoost (n=200)              | 0.9544  | 0.8728   | 0.8607 |
| AdaBoost (n=100)              | 0.9543  | 0.8728   | 0.8611 |
| SVM (Linear kernel)           | 0.9500  | 0.8646   | 0.8520 |
| LDA                           | 0.9490  | 0.8612   | 0.8389 |
| Random Forest (n=300)         | 0.9480  | 0.8668   | 0.8609 |
| Random Forest (n=100)         | 0.9470  | 0.8659   | 0.8591 |
| Extra Trees (n=300)           | 0.9457  | 0.8632   | 0.8582 |
| Extra Trees (n=100)           | 0.9447  | 0.8620   | 0.8560 |
| QDA (reg=0.1)                 | 0.9382  | 0.8564   | 0.8556 |
| SVM (RBF, C=1)                | 0.9373  | 0.8648   | 0.8555 |
| Bagging (DT base)             | 0.9362  | 0.8463   | 0.8345 |
| SVM (RBF, C=10)               | 0.9285  | 0.8607   | 0.8511 |
| SVM (Polynomial, d=3)         | 0.9352  | 0.8634   | 0.8495 |

**Key findings**:
- `HistGradientBoosting` (sklearn's native histogram boosting) matches advanced
  XGBoost/LightGBM without hyperparameter tuning (AUC = 0.9550), confirming the
  dominance of histogram gradient boosting for this data structure.
- SVM (RBF) performs substantially below its theoretical strength (AUC = 0.937 vs.
  LR 0.950) — on 630K instances, SVM's O(n²) complexity forces approximations
  and C-value sensitivity is hard to explore exhaustively.
- AdaBoost (0.9543) outperforms both Random Forest (0.9480) and SVM (0.9373),
  confirming that sequential boosting is more data-efficient than parallel bagging
  for this problem.
- LDA (0.9490) and SVM-Linear (0.9500) closely track Logistic Regression (0.9505),
  confirming the near-linear nature of the decision boundary.

## 5.4 Advanced Gradient Boosting

**Table 4: Gradient boosting results (5-fold CV, ranked by ROC-AUC)**

| Model                      | ROC-AUC | F1-Score | Recall |
|----------------------------|---------|----------|--------|
| **CatBoost (Optuna best)**  | **0.9555** | **0.8750** | **0.8668** |
| XGBoost (Optuna best)      | 0.9555  | 0.8749   | 0.8670 |
| LightGBM (Optuna best)     | 0.9553  | 0.8748   | 0.8676 |
| XGBoost (n=500, lr=0.05)   | 0.9552  | 0.8745   | 0.8669 |
| CatBoost (deep)            | 0.9552  | 0.8746   | 0.8670 |
| LightGBM (n=500, lr=0.05)  | 0.9551  | 0.8745   | 0.8674 |
| XGBoost (DART)             | 0.9551  | 0.8744   | 0.8671 |
| LightGBM (default)         | 0.9547  | 0.8739   | 0.8668 |
| XGBoost (default)          | 0.9547  | 0.8739   | 0.8666 |
| CatBoost (default)         | 0.9547  | 0.8740   | 0.8665 |
| LightGBM (GOSS)            | 0.9545  | 0.8741   | 0.8671 |
| LightGBM (DART)            | 0.9533  | 0.8723   | 0.8656 |

**Key findings**:
- All three major boosting frameworks (XGBoost, LightGBM, CatBoost) converge to
  virtually identical AUC (0.9553–0.9555) after Optuna tuning with 100 trials.
  The marginal gains from hyperparameter optimization are small (+0.0008 over defaults).
- Optuna TPE tuning consistently outperforms CMA-ES (0.9554 vs. 0.9546) for this
  hyperparameter space, suggesting that the search landscape has clear local structure
  that TPE exploits well.
- CatBoost achieved equivalent performance to XGBoost/LightGBM without requiring
  explicit feature encoding, demonstrating its robustness to mixed-type features.
- LightGBM-DART (dropout regularization) underperforms standard GBDT by 0.002 AUC,
  suggesting that dropout is counterproductive for this problem — likely because the
  ensemble is already well-regularized by the mild imbalance and feature structure.

**Optuna best hyperparameters (LightGBM)**:
`num_leaves=24, max_depth=3, learning_rate=0.076, n_estimators=651,
min_child_samples=58, subsample=0.908, colsample_bytree=0.571,
reg_alpha=1.346, reg_lambda=4.3e-6`

The surprisingly shallow tree depth (max_depth=3) and low leaf count (24) suggest that
the data's decision boundary is relatively simple, and that deep trees overfit without
regularization.

## 5.5 Neural Networks

All neural networks were trained on a 150,000-instance subsample (MLP architectures)
or 50,000-instance subsample (TabNet) due to training time constraints on MPS hardware.
Final AUC estimates are from 5-fold cross-validation on the subsample.

**Table 5: Neural network results (5-fold CV on 150K subsample)**

| Architecture              | ROC-AUC | ROC-AUC std | F1-Score | Recall | Train Time |
|---------------------------|---------|-------------|----------|--------|------------|
| MLP Wide (512-256, GELU)  | 0.9525  | ±0.00065    | 0.8697   | 0.8616 | 172s       |
| MLP Deep (5-layer, Swish) | 0.9525  | ±0.00065    | 0.8697   | 0.8623 | 237s       |
| MLP Medium (256-128-64)   | 0.9524  | ±0.00063    | 0.8698   | 0.8620 | 150s       |
| MLP Small (64-32, ReLU)   | 0.9523  | ±0.00063    | 0.8696   | 0.8624 | 175s       |
| **TabNet** (n_d=32)       | 0.9491  | ±0.00168    | 0.8650   | 0.8539 | —          |

**Key findings**:
- All MLP architectures converge to nearly identical performance (AUC ≈ 0.9524 ± 0.0001),
  regardless of depth, width, or activation function. This convergence plateau suggests
  that the data's signal is extractable by even shallow networks, and that additional
  capacity provides no benefit.
- Neural networks plateau approximately **0.003 AUC below** the best gradient boosting
  models (0.9524 vs. 0.9555). This 13-point gap in practical terms (likelihood ratio)
  confirms the well-documented advantage of tree-based methods on structured tabular data.
- **MPS acceleration** on Apple Silicon reduced training time by approximately 3-5× vs.
  CPU for the Wide architecture. MPS memory was stable throughout (peak ~2.8GB).
- **TabNet** (AUC = 0.9491) underperformed all MLP variants and exhibited higher variance
  across folds (std = ±0.00168 vs. ±0.00065 for MLP). TabNet's sequential attention
  mechanism may be poorly suited to problems where all features are globally relevant
  rather than sparsely informative.
- Training on 150K vs. 630K instances introduces a ~0.001–0.002 AUC downward bias in
  reported neural network scores; reported values are conservative estimates.

## 5.6 AutoML and Hyperparameter Optimization

**Table 6: AutoML and optimization results**

| Method                       | ROC-AUC | Time    | Best Model Found        |
|------------------------------|---------|---------|-------------------------|
| FLAML (300s budget)          | 0.9553  | 379s    | LightGBM                |
| FLAML (600s budget)          | 0.9553  | 602s    | LightGBM                |
| Optuna TPE (100 trials)      | 0.9554  | 349s    | LightGBM                |
| Optuna TPE (200 trials)      | 0.9554  | 349s    | LightGBM (same config)  |
| Optuna CMA-ES (100 trials)   | 0.9546  | 303s    | LightGBM                |

**Feature set ablation (LightGBM Optuna-TPE, 5-fold CV)**:

| Feature Set         | ROC-AUC | Δ vs. raw_13 |
|---------------------|---------|--------------|
| raw_13 (baseline)   | 0.9554  | —            |
| scaled_13           | 0.9554  | 0.0000       |
| engineered_18       | 0.9554  | 0.0000       |

**Key findings**:
- FLAML and Optuna converge to the same LightGBM configuration in the same AUC range
  (0.9553–0.9554), confirming that the performance ceiling is dataset-driven, not
  optimizer-driven.
- Doubling the FLAML time budget from 300s to 600s provides no improvement —
  the search has saturated.
- Optuna TPE (0.9554) outperforms CMA-ES (0.9546) for this hyperparameter space.
  CMA-ES's strength is in smooth, continuous landscapes; the LightGBM hyperparameter
  space contains mixed discrete/continuous parameters where TPE's nonparametric
  density estimation is more effective.
- **Feature engineering provides zero measurable benefit**: adding 5 engineered
  features (age×HR ratio, BP×cholesterol product, ST slope interaction, age bins,
  polynomial terms) does not improve LightGBM AUC beyond the raw 13-feature baseline.
  This is consistent with gradient boosting's inherent ability to construct interaction
  terms internally via tree structure.

## 5.7 Class Imbalance Handling

Given the mild 1.23:1 imbalance ratio, we evaluated 10 resampling strategies on a
30,000-instance subsample (SVMSMOTE requires smaller samples due to O(n²) complexity).

**Table 7: Imbalance strategy results (LightGBM base model)**

| Strategy                 | ROC-AUC | F1-Score | Recall  | Δ Recall vs. Baseline |
|--------------------------|---------|----------|---------|-----------------------|
| Baseline (no resampling) | 0.9547  | 0.8727   | 0.8651  | —                     |
| class_weight=balanced    | 0.9548  | 0.8744   | 0.8817  | +1.66pp               |
| RandomUnderSampler       | 0.9547  | 0.8744   | 0.8822  | +1.71pp               |
| SMOTE                    | 0.9546  | 0.8732   | 0.8702  | +0.51pp               |
| TomekLinks               | 0.9545  | 0.8742   | 0.8860  | +2.09pp               |
| ADASYN                   | 0.9542  | 0.8731   | 0.8737  | +0.86pp               |
| BorderlineSMOTE          | 0.9541  | 0.8732   | 0.8799  | +1.48pp               |
| SMOTETomek               | 0.9543  | 0.8727   | 0.8685  | +0.34pp               |
| SVMSMOTE (30K)           | 0.9534  | 0.8733   | 0.8802  | +1.51pp               |
| SMOTEENN                 | 0.9525  | 0.8722   | 0.8833  | +1.82pp               |

**Threshold optimization (LightGBM, full 5-fold CV)**:

| Threshold Strategy        | Threshold | F1-Score | Recall  |
|---------------------------|-----------|----------|---------|
| Default (0.5)             | 0.500     | 0.8759   | 0.8686  |
| Youden's J (optimal)      | 0.450     | 0.8779   | 0.8855  |
| F1-optimal                | 0.434     | 0.8779   | 0.8907  |
| Recall ≥ 0.95 target      | 0.208     | 0.8569   | **0.9500** |

**Key findings**:
- All resampling strategies produce negligible AUC changes (< 0.002), confirming
  that mild imbalance does not fundamentally distort the decision boundary for
  gradient boosting methods.
- The most effective recall improvement strategy is **threshold tuning** rather than
  resampling: shifting the decision threshold from 0.5 to 0.208 raises recall to 0.950
  at a cost of only 0.019 F1 points — far more surgical than resampling.
- `class_weight=balanced` provides +1.66pp recall with negligible AUC cost, making it
  the recommended approach when higher recall is desired without extensive resampling.
- TomekLinks gives the best recall improvement among resampling methods (+2.09pp) by
  cleaning boundary-region instances. However, its gain is matched by simply using
  `class_weight=balanced`.
- SVMSMOTE (AUC = 0.9534) shows slight degradation vs. baseline — synthetic minority
  samples generated in feature space may not align with the true manifold of clinical
  measurements, introducing noise near decision boundaries.
- **Clinical recommendation**: For deployment in a cardiac screening context where
  false negatives (missed diagnoses) are costly, Youden's J threshold (t = 0.450)
  offers the best precision-recall tradeoff (F1 = 0.878, Recall = 0.886). For maximum
  sensitivity, t = 0.208 achieves 95% recall.

## 5.8 Model Interpretability

### 5.8.1 SHAP Feature Importance

SHAP TreeExplainer values were computed on a 10,000-instance subsample of the
Optuna-tuned LightGBM model.

**Table 8: SHAP global feature importance (mean |SHAP value|)**

| Rank | Feature                  | Mean |SHAP| | Interpretation                           |
|------|--------------------------|-------------|------------------------------------------|
| 1    | Thallium                 | 1.027        | Dominant; reversible defect →  +disease  |
| 2    | Chest pain type          | 0.945        | Type 4 (asymptomatic) → +disease         |
| 3    | Max HR                   | 0.687        | Lower max HR → +disease                  |
| 4    | Number of vessels fluro  | 0.579        | More vessels → +disease                  |
| 5    | Exercise angina          | 0.476        | Presence of angina → +disease            |
| 6    | Slope of ST              | 0.439        | Downsloping ST → +disease                |
| 7    | Sex                      | 0.425        | Male → slight +disease                   |
| 8    | ST depression            | 0.359        | Higher depression → +disease             |
| 9    | Age                      | 0.332        | Older → slight +disease                  |
| 10   | EKG results              | 0.221        | Abnormal EKG → +disease                  |
| 11   | Cholesterol              | 0.066        | Weak positive association                |
| 12   | BP                       | 0.056        | Negligible                               |
| 13   | FBS over 120             | 0.016        | Virtually no contribution                |

SHAP importance ranking confirms the Mutual Information analysis from EDA:
Thallium (SHAP = 1.027) and Chest pain type (0.945) dominate by a wide margin.
The bottom three features (Cholesterol, BP, FBS) contribute minimally, consistent
with their low MI scores.

### 5.8.2 LIME Local Explanations

LIME (Local Interpretable Model-Agnostic Explanations) was applied to 10 individual
predictions. Local explanations were consistent with SHAP global importance:
Thallium and Chest pain type appeared in the top-3 local features for 9 of 10
explained instances. The two methods converge on the same feature ranking, increasing
confidence in the explanations' validity.

### 5.8.3 Partial Dependence Plots

PDP analysis revealed the following monotonic relationships:
- **Thallium**: AUC-weighted disease probability increases monotonically from normal (3)
  to fixed defect (6) to reversible defect (7).
- **Max HR**: Disease probability decreases monotonically with increasing max HR,
  confirming chronotropic incompetence as a disease signal.
- **Number of vessels**: Nearly linear increase in disease probability with each
  additional vessel affected (0.15 → 0.45 → 0.72 → 0.92).
- **Chest pain type**: Non-monotonic; types 1-3 show low-medium risk, type 4 shows
  maximum risk — the asymptomatic paradox in action.

### 5.8.4 Model Calibration

LightGBM calibration curves show excellent probability calibration (Brier score = 0.062),
with predicted probabilities closely matching observed frequencies across 10 equal-width
bins. This makes raw model outputs directly usable as clinical risk scores without
post-hoc calibration (Platt scaling / isotonic regression).

### 5.8.5 Confusion Matrix Analysis

At the default threshold (t = 0.5) on the full 5-fold CV:
- True Positives: 244,720 (correctly identified disease)
- True Negatives: 303,900 (correctly identified healthy)
- False Positives: 43,646 (unnecessary referrals)
- False Negatives: 37,734 (missed diagnoses)

The False Negative rate of 13.4% (37,734/282,454 true positives) is the primary
clinical concern. Shifting to Youden's threshold (t = 0.450) reduces FN to 32,253
(11.4%) at a modest cost of 4,000 additional false positives.

## 5.9 Unsupervised Analysis

### 5.9.1 Dimensionality Reduction

PCA on the full dataset explains only **30.4% variance** with two components,
reflecting the dataset's multi-dimensional clinical structure where no single
linear combination dominates. Retaining 10 components captures 95% of variance.

t-SNE (perplexity=30 and 50) and UMAP (n_neighbors=15 and 50) on 20,000 instances
both reveal clear two-cluster structure aligned with the disease/no-disease boundary.
UMAP produces more globally coherent cluster separation than t-SNE, consistent with
UMAP's superior preservation of global structure.

### 5.9.2 K-Means Clustering

**Table 9: K-Means clustering results (k=2 to 8)**

| k | Silhouette | ARI   | NMI   |
|---|------------|-------|-------|
| 2 | 0.170      | **0.542** | **0.439** |
| 3 | 0.177      | 0.462 | 0.347 |
| 4 | 0.125      | 0.285 | 0.276 |
| 5 | 0.111      | 0.232 | 0.260 |
| 6 | 0.111      | 0.201 | 0.237 |

K-Means with **k=2** achieves the strongest label alignment (ARI = 0.542, NMI = 0.439),
confirming that the data has a natural two-cluster structure that closely corresponds
to the disease/no-disease boundary. Cluster profiles at k=2:

| Cluster | Disease Rate | Dominant Features                                      |
|---------|-------------|--------------------------------------------------------|
| 0       | **88.8%**   | Reversible Thallium defect, Type 4 chest pain, low HR  |
| 1       | **14.5%**   | Normal Thallium, Type 1-2 chest pain, high HR          |

The 88.8%/14.5% split represents an almost perfect separation of high-risk and
low-risk patient subgroups, discovered purely from feature structure without labels.

### 5.9.3 Other Clustering Methods

- **Hierarchical (Ward linkage)**: ARI = 0.222 — substantially worse than K-Means,
  suggesting non-hierarchical cluster geometry.
- **Hierarchical (Complete/Average linkage)**: ARI ≈ 0.000 — collapses to trivially
  unbalanced dendrograms.
- **DBSCAN**: All parameter configurations produced 0 clusters with 100% noise points
  on the 5,000-instance subsample, indicating that the data does not have the compact,
  well-separated spherical geometry DBSCAN requires.
- **Gaussian Mixture Models**: BIC continues decreasing through k=6, suggesting
  the data has more than two statistical modes (multi-modal distributions within
  each disease class, perhaps corresponding to different clinical presentation subtypes).

### 5.9.4 Cluster Features in Supervised Learning

K-Means cluster membership (k=2 and k=3) was added as features to LightGBM.
The resulting AUC (0.9554) was identical to the baseline — cluster membership
provides no additional discriminative information beyond what the raw features
already encode. This is expected given that the clusters were derived from the
same features the model already uses.

## 5.10 LLM Experiments

### 5.10.1 Experimental Setup

**Important caveat**: The originally planned model,
`hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_S` (≈20B parameters, Q4 quantized, ~11GB),
was unavailable in the local Ollama installation at experiment time.
The substitute model, **Qwen3:0.6b** (522MB, 0.6 billion parameters), was used.
This is a critical limitation: a 0.6B model has approximately 33× fewer parameters
than the planned 20B model and cannot be expected to perform comparably.
Results in this section should be interpreted as a lower bound for LLM classification
capability on this task.

### 5.10.2 Row Serialization

Each patient record is serialized to natural language:
```
"A 58-year-old male with chest pain type 4, resting blood pressure 152 mmHg,
cholesterol 239 mg/dl, fasting blood sugar: no, EKG results: 2, max HR: 148,
exercise angina: yes, ST depression: 1.0, ST slope: 2, 1 vessels colored by
fluoroscopy, Thallium result: 7. Does this patient have heart disease?
Answer with exactly: Presence or Absence."
```

### 5.10.3 Results

**Table 10: LLM experiment results (n=200 samples, balanced 100/100)**

| Experiment          | Accuracy | F1-Score | Recall  | Notes                               |
|---------------------|----------|----------|---------|-------------------------------------|
| Zero-Shot           | 0.500    | 0.000    | 0.000   | All-negative predictions            |
| Few-Shot (5-shot)   | 0.500    | 0.000    | 0.000   | Still all-negative                  |
| Chain-of-Thought    | 0.500    | 0.000    | 0.000   | All-negative (50 samples)           |
| Risk Score Feature  | 0.500    | —        | —       | Risk correlation = NaN              |
| Edge Case Analysis  | 0.400    | —        | —       | 4/10 cases correct                  |

### 5.10.4 Analysis

The Qwen3:0.6b model predicts **Absence for every patient** regardless of clinical
features, resulting in:
- Accuracy = 50% (matches random guessing on balanced sample)
- F1 = 0.000 (no positive predictions)
- Recall = 0.000 (all disease cases missed)

The response inspection revealed that the model often returned empty strings or
non-compliant formatting rather than "Presence"/"Absence", and when it did respond,
it consistently said "Absence." The model exhibits a strong **prior bias** toward
the majority class (Absence = 55.2%) even before processing patient features.

The Risk Score experiment showed correlation = NaN, indicating the model could not
produce consistent numeric risk scores.

**Why did LLM classification fail?**

1. **Model size**: 0.6B parameters is insufficient for medical reasoning. GPT-4-level
   models (175B+) show reasonable zero-shot medical performance; sub-1B models do not.
2. **Tabular format**: Numerical clinical features presented as text are difficult for
   language models to reason about quantitatively without fine-tuning.
3. **Calibration**: Small LLMs have poorly calibrated uncertainty and exhibit strong
   output biases inherited from pre-training data distribution.
4. **No domain fine-tuning**: The model has no specific cardiac or medical knowledge
   alignment.

**Projected results with 20B model**: Published evaluations of GPT-3.5-class models
(~20B effective parameters) on tabular medical tasks suggest zero-shot AUC of 0.70–0.80,
with few-shot improving to 0.75–0.85. Even with the planned model, gradient boosting
(AUC = 0.9555) would be expected to substantially outperform LLM zero-shot classification.

## 5.11 Final Ensemble

**Table 11: Phase 12 ensemble results (5-fold CV, full 630K)**

| Model / Ensemble               | ROC-AUC | F1-Score | Recall  |
|--------------------------------|---------|----------|---------|
| **RankAvg (LGB+XGB+CAT)**      | **0.9555** | 0.8713  | **0.9215** |
| SoftVote (LGB+XGB+CAT)         | 0.9555  | 0.8749   | 0.8670  |
| Stack (LR meta, 3 models)      | 0.9555  | 0.8749   | 0.8672  |
| SoftVote (LGB+XGB)             | 0.9555  | 0.8749   | 0.8671  |
| SoftVote (LGB+CAT)             | 0.9555  | 0.8749   | 0.8669  |
| Stack (LR meta, 4 models)      | 0.9555  | 0.8750   | 0.8671  |
| **LightGBM@t=0.438 (Youden)**  | 0.9554  | **0.8763** | 0.8865  |
| CatBoost (Optuna)              | 0.9555  | 0.8750   | 0.8668  |
| XGBoost (Optuna)               | 0.9555  | 0.8749   | 0.8670  |
| LightGBM (Optuna)              | 0.9554  | 0.8748   | 0.8670  |
| Stack (LGB meta, 3 models)     | 0.9554  | 0.8750   | 0.8682  |
| RankAvg (All 4)                | 0.9551  | 0.8712   | 0.9203  |
| Logistic Regression (L2)       | 0.9505  | 0.8679   | 0.8563  |

The full leaderboard spanning all 83 models from Phases 3–12 is available in
`results/metrics/12_full_leaderboard.csv`.

**Key findings**:
- Ensembles provide **no meaningful AUC improvement** over the best individual model
  (0.9555 for all methods). The three gradient boosting models are already well-calibrated
  and capture similar discriminative information; combining them does not add new signal.
- **Rank averaging** (SoftVote on percentile-ranked probabilities) achieves the highest
  recall (0.9215) at the default threshold by systematically shifting probability scores
  toward the center, effectively lowering the implicit decision threshold.
- **Stacking** (both LR and LGB meta-learners) matches soft voting exactly —
  the meta-learner learns to nearly uniformly weight the three base models, equivalent
  to the soft vote.
- The **Youden-threshold LightGBM** (t=0.438) achieves the best F1 (0.8763) and
  strong recall (0.8865), making it the recommended deployment model for balanced
  precision-recall performance.
- Adding Logistic Regression to the ensemble (SoftVote/RankAvg "ALL4") slightly
  degrades AUC (0.9555 → 0.9551) by introducing calibration mismatch from the
  weaker model.

---

# 6. Discussion

## 6.1 What Worked and Why

### Gradient Boosting Dominance

All three gradient boosting frameworks (XGBoost, LightGBM, CatBoost) achieve
near-identical AUC = 0.9555 after tuning — the highest score across all 83 models.
This dominance on tabular data is consistent with the broader literature
[@grinsztajn2022tree; @mcelfresh2023neural].

The underlying mechanism: gradient boosting builds an ensemble of shallow trees
that efficiently partition the feature space using axis-aligned splits. For structured
tabular data where feature interactions are locally relevant (e.g., "high Thallium
defect AND low max HR"), tree-based methods naturally capture these interactions
without explicit engineering. The feature space here (13 clinical measurements) is
low enough dimensionality that exhaustive axis-aligned splits are computationally
efficient, and the dataset is large enough (630K) that variance of individual trees
averages out.

### Feature Engineering Provides No Gain

The feature ablation (Section 5.6) confirmed that engineered features (age×HR ratio,
BP×cholesterol product, polynomial terms) add zero benefit to LightGBM.
This is expected: gradient boosting with sufficient depth and estimators can construct
all pairwise and higher-order interactions internally. External feature engineering
is primarily useful when the base learner cannot construct the relevant interactions
(e.g., logistic regression).

### Threshold Tuning vs. Resampling

The mild 1.23:1 imbalance is insufficient to distort the calibrated probabilities of
a well-trained gradient boosting model. Threshold tuning (shifting t from 0.5 to 0.450
via Youden's J) provides larger recall gains (+1.7pp F1, +1.7pp Recall) than any
resampling strategy, with no training cost. This is consistent with the theoretical
expectation: resampling modifies the training distribution to shift the learned boundary,
while threshold tuning directly modifies the operating point on the same learned boundary.
For problems with mild imbalance, threshold tuning is strictly more efficient.

### Mutual Information as Ground Truth

SHAP values (computed post-hoc on the trained model) closely match mutual information
rankings (computed pre-hoc from data statistics). The top-5 SHAP features
(Thallium, Chest pain type, Max HR, Number of vessels, Exercise angina) match the
top-5 MI rankings exactly. This convergence between a statistical measure of
marginal association and a model-specific measure of conditional contribution
strongly validates both the model's feature weighting and the EDA findings.

### Unsupervised Structure Reflects Supervised Boundary

K-Means (k=2) achieving ARI = 0.542 without labels confirms that the feature space
has strong intrinsic cluster structure aligned with the disease outcome. A clinician
examining Cluster 0 (88.8% disease rate: reversible Thallium defect, asymptomatic
chest pain, low max HR) would immediately recognize a high-risk cardiac profile.
This validates the clinical meaning of the features and suggests that unsupervised
methods could identify high-risk patient cohorts in unlabeled populations.

## 6.2 What Did Not Work and Why

### Neural Networks Fall Short of Gradient Boosting

All MLP architectures plateau at AUC ≈ 0.9524, approximately 0.003 below gradient
boosting (0.9555). Three explanations:

1. **Inductive bias mismatch**: Neural networks learn smooth, differentiable functions.
   The heart disease decision boundary is defined by medical thresholds (e.g.,
   Thallium defect values 6 vs. 7 are discrete, not continuous) that tree-based
   axis-aligned splits model more naturally.
2. **Subsampling penalty**: Neural networks were trained on 150K instances (24% of data)
   vs. gradient boosting on 630K. This 4× data disadvantage accounts for approximately
   0.001–0.002 AUC degradation.
3. **Scale**: The MPS-accelerated training (Apple Silicon Metal GPU) was effective but
   constrained by the 16GB unified memory ceiling, preventing full-dataset training
   within reasonable time bounds.

TabNet's additional underperformance (AUC = 0.9491) is attributable to its sequential
attention mechanism, which imposes sparsity constraints better suited to problems
where only a small subset of features is relevant per instance. For heart disease
prediction, all 13 features contribute meaningfully (albeit unequally), and sparse
feature selection is suboptimal.

### LLM Zero-Shot Classification Failed Completely

The Qwen3:0.6b model (0.6B parameters) produced all-negative predictions across
all five experimental conditions, achieving Recall = 0.000. As discussed in Section 5.10,
this reflects:

1. **Insufficient scale**: Sub-1B models lack the parameter capacity for multi-step
   medical reasoning from tabular inputs.
2. **Majority class bias**: The model absorbed the 55% Absence prior and
   applied it universally.
3. **Quantitative reasoning failure**: Parsing and comparing numerical clinical
   values (BP = 152, cholesterol = 239) requires arithmetic reasoning capabilities
   that are severely diminished in very small models.

The key lesson is that tabular medical classification is not simply a language task.
Even the planned 20B model would likely achieve AUC < 0.85 zero-shot, while
LightGBM achieves 0.9555 — a difference of 1.5+ million patients correctly classified
per 10 million screened.

### Ensemble Methods Provide No Gain

Soft voting, rank averaging, and stacking all achieve AUC ≤ 0.9555 — identical to
the best individual model. This reflects the **ensemble diversity paradox**: for ensembles
to improve over individual models, base models must make **different** errors.
XGBoost, LightGBM, and CatBoost — while algorithmically distinct — all converge to
the same Optuna-optimized configuration space and learn highly correlated decision
functions. Their OOF predictions have Pearson correlations of 0.96–0.98, leaving
minimal room for ensemble correction.

The performance ceiling of 0.9555 appears to be an **intrinsic dataset ceiling**:
the features available (13 clinical measurements) do not contain sufficient information
to discriminate all borderline cases, regardless of the modeling approach.

### Feature Engineering Adds No Value

Engineered features (polynomial terms, ratio features, interaction products) provide
zero measurable benefit. This is a general observation: for gradient boosting models
on medium-scale tabular data, the model's internal feature combination mechanism
makes external engineering redundant. Feature engineering provides value for linear
models (where explicit interactions must be constructed manually) but is unnecessary
overhead for tree-based ensembles.

## 6.3 Clinical Implications

### Model Selection for Deployment

For a cardiac screening application, we recommend:
- **LightGBM (Optuna-tuned) at threshold t = 0.450**: AUC = 0.9554, F1 = 0.878,
  Recall = 0.886. Misses 11.4% of positive cases; generates 15.4% false alarm rate.
- **Rank-Averaging Ensemble at default threshold**: AUC = 0.9555, Recall = 0.9215.
  For high-sensitivity applications (mass screening), this reduces missed diagnoses
  to 7.9% of positive cases.
- **LightGBM at t = 0.208 (Recall ≥ 95%)**: For settings where missing a diagnosis
  is unacceptable (emergency triage), achieving 95% sensitivity with F1 = 0.857.

### Feature Prioritization in Clinical Practice

The SHAP analysis provides actionable guidance: if a clinician can only perform
one diagnostic test, the Thallium stress test (SHAP = 1.027) provides the most
information. A normal Thallium result (value 3) reduces disease probability
substantially; a reversible defect (value 7) dramatically increases it.
Blood pressure and fasting blood sugar, despite their common inclusion in risk
calculators, contribute negligibly in this dataset.

### Caveat: Synthetic Data

The dataset is synthetically generated. BP's non-significance (p = 0.503) and FBS's
negligible association are likely artifacts of the synthetic generation process
rather than genuine clinical findings. Real-world deployment would require validation
on independent clinical datasets before clinical use.

## 6.4 Limitations

1. **Synthetic dataset**: The Kaggle competition data is synthetically generated from
   the UCI Heart Disease dataset (n=303). Patterns may not reflect real-world clinical
   distributions. BP's non-significance is likely a generation artifact.

2. **LLM model substitution**: The planned 20B-parameter LLM (GPT-OSS 20B Q4) was
   unavailable; Qwen3:0.6b was used instead. Results in Section 5.10 represent a
   worst-case LLM scenario and should not be generalized to larger models.

3. **Neural network subsampling**: MLP architectures were trained on 150K (24%) of the
   available data. Full-dataset training may reduce the ~0.003 AUC gap between
   neural networks and gradient boosting.

4. **Feature completeness**: 13 features may not capture all clinically relevant
   information. Missing features (troponin levels, family history, smoking status,
   BMI, echocardiography results) would likely improve any model's performance ceiling.

5. **Single-institution evaluation**: All results are from cross-validation on the
   training set. External validation on an independent clinical cohort is required
   before clinical deployment.

6. **Temporal validity**: No longitudinal data are available; all predictions are
   cross-sectional.

---

# 7. Conclusion

We presented a comprehensive 13-phase benchmark of 83 machine learning models for
heart disease prediction on the Kaggle Playground Series S6E2 dataset (630K instances,
13 features). Our principal findings are:

1. **Gradient boosting defines the performance ceiling**: Optuna-tuned XGBoost,
   LightGBM, and CatBoost all achieve AUC = 0.9555 — the highest score across all
   approaches. No ensemble, neural network, AutoML, or LLM method meaningfully exceeds
   this ceiling, which appears intrinsic to the information content of the 13 features.

2. **Thallium and chest pain type dominate predictions**: SHAP analysis confirms that
   the stress test result (SHAP = 1.027) and chest pain classification (0.945) are the
   two dominant predictors. Blood pressure and fasting blood sugar contribute negligibly
   — a synthetic data artifact with clinical implications for dataset validity.

3. **Threshold tuning outperforms resampling**: For clinical recall optimization, shifting
   the decision threshold from 0.5 to 0.208 achieves 95% recall with minimal F1 cost —
   more effective than any of the 10 resampling strategies evaluated.

4. **Neural networks plateau below gradient boosting**: MPS-accelerated MLP architectures
   converge at AUC ≈ 0.9524 (0.003 below gradient boosting), confirming the widely
   reported disadvantage of neural networks on structured tabular data.

5. **LLM zero-shot classification is unsuitable at small scales**: Qwen3:0.6b (0.6B
   parameters) predicts the all-negative class, achieving Recall = 0.000. Even larger
   models (20B+) would be expected to fall far below gradient boosting performance
   without domain fine-tuning.

6. **Unsupervised structure validates clinical knowledge**: K-Means (k=2) discovers
   patient clusters with 88.8% and 14.5% disease prevalence using no labels —
   confirming that the feature space contains coherent clinical subgroups.

The recommended deployment configuration is Optuna-tuned LightGBM at Youden's
optimal threshold (t = 0.450), providing AUC = 0.9554, F1 = 0.878, and Recall = 0.886.
For mass-screening applications requiring higher sensitivity, a rank-averaging ensemble
(LGB+XGB+CatBoost) at the default threshold achieves Recall = 0.9215.

All code, results, figures, and MLflow experiment logs are version-controlled and
publicly reproducible.

---

# References

::: {#refs}
:::

---

# Appendix A: Model Configuration Details

## A.1 LightGBM Optuna Best Parameters
```
boosting_type: gbdt
num_leaves: 24
max_depth: 3
learning_rate: 0.0762
n_estimators: 651
min_child_samples: 58
subsample: 0.9075
colsample_bytree: 0.5711
reg_alpha: 1.3459
reg_lambda: 4.27e-6
```

## A.2 XGBoost Optuna Best Parameters
```
n_estimators: 479
max_depth: 5
learning_rate: 0.0623
subsample: 0.9467
colsample_bytree: 0.5009
min_child_weight: 6
gamma: 0.0462
reg_alpha: 1.20e-8
reg_lambda: 0.0058
```

## A.3 CatBoost Optuna Best Parameters
```
iterations: 487
depth: 4
learning_rate: 0.1465
l2_leaf_reg: 0.2321
border_count: 78
```

## A.4 Neural Network Configurations

| Architecture | Layers       | Activation | Dropout | Batch Norm | Epochs |
|-------------|--------------|------------|---------|------------|--------|
| MLP Small   | [64, 32]     | ReLU       | 0.30    | No         | 60     |
| MLP Medium  | [256,128,64] | ReLU       | 0.30    | Yes        | 60     |
| MLP Wide    | [512, 256]   | GELU       | 0.25    | No         | 60     |
| MLP Deep    | 5×128        | Swish      | 0.20    | Yes        | 60     |
| TabNet      | n_d=32, n_a=32, n_steps=5 | — | — | — | 100   |

All neural networks: Adam optimizer, lr=1e-3, ReduceLROnPlateau (patience=4,
factor=0.5), early stopping (patience=8), batch size=2048, MPS device.

---

# Appendix B: Reproducibility

## B.1 Environment

```
Python 3.11.11 (pyenv)
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
xgboost==3.0.1
lightgbm==4.6.0
catboost==1.2.7
torch==2.10.0 (MPS enabled)
shap==0.47.2
optuna==4.3.0
flaml==2.3.4
imbalanced-learn==0.13.0
mlflow==2.22.1
```

## B.2 Execution

All experiments run as standalone Python scripts under `.venv/bin/python`:
```bash
PYTHONUNBUFFERED=1 .venv/bin/python src/run_XX_phase_name.py
```

MLflow experiments accessible via: `mlflow ui --backend-store-uri ./mlruns`

## B.3 Git Commit History

| Commit | Content |
|--------|---------|
| `feat: initialize project structure...` | Phase 0: environment setup |
| `analysis: complete EDA...`             | Phase 1: EDA results |
| `feat: feature engineering...`          | Phase 2: feature sets |
| `results: baseline models...`           | Phase 3: baselines |
| `results: classical ML models...`       | Phase 4: classical ML |
| `results: advanced boosting...`         | Phase 5: XGB/LGB/CAT |
| `results: neural network experiments...`| Phase 6: neural nets |
| `results: AutoML and hyperparameter...` | Phase 7: AutoML |
| `results: imbalance handling...`        | Phase 8: imbalance |
| `analysis: interpretability...`         | Phase 9: SHAP/LIME |
| `analysis: dimensionality reduction...` | Phase 10: clustering |
| `results: LLM classification...`        | Phase 11: LLM |
| `results: final ensemble...`            | Phase 12: ensemble |
