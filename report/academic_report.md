---
title: "Predicting Heart Disease: A Comprehensive Machine Learning Study"
author: "Dustin Ober"
date: "2026"
abstract: |
  *[To be written after all experiments are complete.]*
  
  This study presents a comprehensive evaluation of machine learning and deep learning
  techniques applied to the Kaggle Playground Series S6E2 heart disease prediction dataset
  (630,000 training samples, 13 clinical features). We systematically evaluate over 50 models
  spanning logistic regression, classical ensemble methods, gradient boosting (XGBoost,
  LightGBM, CatBoost), deep neural networks with Apple Silicon MPS acceleration, AutoML,
  and large language model (LLM) zero-shot classification via Ollama. We report
  ROC-AUC, F1-score, and Recall across all experiments tracked via MLflow, conduct
  SHAP-based interpretability analysis to surface clinically meaningful feature
  contributions, and explore unsupervised structure via PCA, t-SNE, UMAP, and
  k-means clustering. Results show that [*to be completed*]. The study serves as a
  reproducible portfolio reference for applied ML in medical classification tasks.

bibliography: references.bib
csl: ieee.csl
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
models while maintaining the clinical feature structure of the original benchmark.

This paper makes the following contributions:

1. A systematic benchmark of 50+ machine learning approaches across 13 experimental phases
2. Rigorous evaluation on three clinically meaningful metrics: ROC-AUC, F1-Score, and Recall
3. SHAP-based interpretability analysis connecting model behavior to clinical domain knowledge
4. The first published evaluation of a 20-billion parameter quantized LLM for zero-shot tabular heart disease classification via natural language prompting
5. A fully reproducible experimental framework with MLflow experiment tracking

---

# 2. Related Work

*[To be completed — will cite 10-12 key papers on ML for heart disease, tabular deep learning, SHAP interpretability, and LLMs for structured data.]*

Key areas to cover:
- Classic ML on UCI Heart Disease (Detrano et al. 1989, Janosi et al.)
- Random forests and gradient boosting for medical classification
- Deep learning for tabular data (TabNet, NODE, SAINT)
- AutoML in clinical prediction (AutoSklearn, FLAML)
- SHAP for medical AI interpretability (Lundberg & Lee 2017)
- LLMs for structured/tabular data (TabLLM, AnyMAL, FeatLLM)
- Class imbalance in medical datasets (SMOTE, Chawla et al. 2002)

---

# 3. Dataset and Preprocessing

## 3.1 Dataset Overview

The dataset is sourced from the Kaggle Playground Series Season 6, Episode 2 competition.
It is a synthetically generated dataset based on the UCI Heart Disease repository,
significantly scaled up from the original 303 instances.

| Split | Instances | Features | Target Distribution |
|---|---|---|---|
| Training | 630,000 | 13 | 55.2% Absence, 44.8% Presence |
| Test | 270,000 | 13 | Unknown |

The target variable is binary: `Heart Disease` — *Presence* (1) or *Absence* (0).
The mild class imbalance (55/45 split) is notable but does not require aggressive
rebalancing strategies as a primary concern.

## 3.2 Feature Descriptions

| Feature | Type | Clinical Significance |
|---|---|---|
| Age | Continuous | Strong predictor; risk increases with age |
| Sex | Binary (0=F, 1=M) | Males historically higher risk |
| Chest pain type | Ordinal (1-4) | Type 4 (asymptomatic) paradoxically highest risk |
| BP | Continuous | Resting blood pressure (mmHg) |
| Cholesterol | Continuous | Serum cholesterol (mg/dl) |
| FBS over 120 | Binary | Fasting blood sugar >120 mg/dl |
| EKG results | Ordinal (0-2) | Resting ECG results |
| Max HR | Continuous | Maximum heart rate achieved during exercise |
| Exercise angina | Binary | Exercise-induced angina |
| ST depression | Continuous | ST depression induced by exercise relative to rest |
| Slope of ST | Ordinal (1-3) | Slope of peak exercise ST segment |
| Number of vessels fluro | Ordinal (0-3) | Number of major vessels colored by fluoroscopy |
| Thallium | Ordinal (3,6,7) | Thallium stress test result |

## 3.3 Preprocessing

*[To be completed after Phase 2 — will describe chosen encoding/scaling strategy and rationale.]*

## 3.4 Feature Engineering

*[To be completed after Phase 2.]*

---

# 4. Methodology

## 4.1 Evaluation Protocol

All supervised models are evaluated using Stratified 5-fold Cross-Validation
(`sklearn.model_selection.StratifiedKFold`, `n_splits=5`, `random_state=42`).
We report mean ± standard deviation across folds for:
- **ROC-AUC**: primary ranking metric; threshold-independent
- **F1-Score**: harmonic mean of precision and recall (threshold=0.5)
- **Recall**: sensitivity; clinically important to minimize false negatives

All experiments are logged to MLflow under the experiment `Heart-Disease-Kaggle`.

## 4.2 Hardware

Experiments are conducted on an Apple M1 MacBook Pro (16GB Unified Memory).
Neural network training uses the Metal Performance Shaders (MPS) backend
in PyTorch 2.10.0. Gradient boosting models use CPU-optimized histogram methods.
The LLM experiments use a Q4-quantized 20B parameter model (11GB) via Ollama.

## 4.3 Model Families

*[Each section 4.3.x will be populated as experiments are run.]*

### 4.3.1 Baseline Models
### 4.3.2 Classical Ensemble Methods
### 4.3.3 Gradient Boosting
### 4.3.4 Neural Networks
### 4.3.5 AutoML
### 4.3.6 Large Language Models

---

# 5. Experiments and Results

## 5.1 Exploratory Data Analysis

### 5.1.1 Dataset Characteristics

The training set contains 630,000 instances with no missing values and no duplicate feature rows.
The target distribution is mildly imbalanced: 55.2% Absence (347,546) vs 44.8% Presence (282,454),
yielding an imbalance ratio of 1.23:1. This modest imbalance does not necessitate aggressive
resampling as a primary strategy, though we evaluate its impact systematically in Section 5.7.

A Kolmogorov-Smirnov test comparing train and test feature distributions showed no statistically
significant drift for any feature (all p > 0.05), confirming that the synthetic generation process
maintained distributional consistency across splits.

### 5.1.2 Feature Informativeness

**Mutual Information Rankings** (features vs. target, descending):

| Rank | Feature | MI Score | Cramér's V / Effect |
|---|---|---|---|
| 1 | Thallium | 0.2358 | V=0.606 (strong) |
| 2 | Chest pain type | 0.1895 | V=0.525 (strong) |
| 3 | Sex | 0.1318 | V=0.342 (moderate) |
| 4 | Max HR | 0.1293 | MWU p<0.001 |
| 5 | Slope of ST | 0.1250 | V=0.430 (strong) |
| 6 | Exercise angina | 0.1239 | V=0.442 (strong) |
| 7 | Number of vessels fluro | 0.1206 | V=0.463 (strong) |
| 8 | ST depression | 0.1082 | MWU p<0.001 |
| 9 | EKG results | 0.0751 | V=0.219 (moderate) |
| 10 | Age | 0.0299 | MWU p<0.001 |
| 11 | Cholesterol | 0.0120 | MWU p<0.001 |
| 12 | BP | 0.0110 | MWU p=0.503 (n.s.) |
| 13 | FBS over 120 | 0.0022 | V=0.034 (negligible) |

**Key finding**: Blood pressure (BP) showed no statistically significant difference between classes
(Mann-Whitney U, p=0.503), making it the weakest predictor despite its clinical relevance.
FBS over 120 (fasting blood sugar) also showed negligible association (MI=0.002, Cramér's V=0.034).

### 5.1.3 Clinical Interpretation

- **Thallium stress test** (MI=0.236, V=0.606): The strongest single predictor. Normal thallium
  (value 3) strongly favors Absence; fixed defect (6) and reversible defect (7) strongly favor Presence.
- **Chest pain type** (MI=0.189, V=0.525): Counter-intuitively, Type 4 (asymptomatic) is most
  associated with Presence — a well-known clinical paradox where silent ischemia carries high risk.
- **Max HR** (MI=0.129): Lower maximum heart rate during exercise is associated with Presence,
  reflecting reduced cardiovascular fitness in diseased patients.
- **Exercise angina** (V=0.442): A direct symptom of ischemia; strongly associated with Presence.
- **Number of vessels colored by fluoroscopy** (V=0.463): More vessels with disease → higher Presence rate.

## 5.2 Baseline Models

*[To be completed after Phase 3.]*

## 5.3 Classical Machine Learning

*[To be completed after Phase 4.]*

## 5.4 Advanced Gradient Boosting

*[To be completed after Phase 5.]*

## 5.5 Neural Networks

*[To be completed after Phase 6.]*

## 5.6 AutoML and Hyperparameter Optimization

*[To be completed after Phase 7.]*

## 5.7 Class Imbalance Handling

*[To be completed after Phase 8.]*

## 5.8 Model Interpretability

*[To be completed after Phase 9.]*

## 5.9 Unsupervised Analysis

*[To be completed after Phase 10.]*

## 5.10 LLM Experiments

*[To be completed after Phase 11.]*

## 5.11 Final Ensemble

*[To be completed after Phase 12.]*

---

# 6. Discussion

## 6.1 What Worked and Why

*[To be completed.]*

## 6.2 What Did Not Work and Why

*[To be completed.]*

## 6.3 Clinical Implications

*[To be completed.]*

## 6.4 Limitations

*[To be completed.]*

---

# 7. Conclusion

*[To be completed.]*

---

# References

::: {#refs}
:::
