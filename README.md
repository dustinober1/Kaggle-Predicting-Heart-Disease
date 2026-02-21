# Heart Disease Prediction — Complete ML Exploration

A comprehensive machine learning study on the [Kaggle Playground Series S6E2](https://www.kaggle.com/competitions/playground-series-s6e2) dataset.
This is a portfolio project with a formal academic report covering the full spectrum of ML techniques applied to binary heart disease classification.

---

## Dataset

| File | Rows | Description |
|---|---|---|
| `playground-series-s6e2/train.csv` | 630,000 | Training data with labels |
| `playground-series-s6e2/test.csv` | 270,000 | Test data (no labels) |
| `playground-series-s6e2/sample_submission.csv` | 270,000 | Submission format |

**Target**: `Heart Disease` — `Presence` (1) / `Absence` (0)

**Features** (13): Age, Sex, Chest pain type, BP, Cholesterol, FBS over 120,
EKG results, Max HR, Exercise angina, ST depression, Slope of ST,
Number of vessels fluro, Thallium

---

## Project Structure

```
├── notebooks/          # 13 numbered Jupyter notebooks (one per experimental phase)
├── src/                # Shared Python modules (data_utils, evaluation, visualization)
├── results/
│   ├── figures/        # All saved plots (committed)
│   ├── metrics/        # Experiment result CSVs/JSONs (committed)
│   └── models/         # Saved model files (gitignored — too large)
├── report/
│   ├── academic_report.md   # Live-updated formal academic paper
│   └── academic_report.pdf  # Final rendered PDF
├── playground-series-s6e2/  # Raw data CSVs
├── requirements.txt
└── .python-version          # pyenv version pin (3.11.11)
```

---

## Setup

```bash
# 1. Install pyenv and Python 3.11.11
brew install pyenv
pyenv install 3.11.11

# 2. Clone and enter the project
git clone <repo-url>
cd Kaggle-Predicting-Heart-Disease

# 3. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Register Jupyter kernel
python -m ipykernel install --user --name heart-disease --display-name "Heart Disease (3.11)"

# 6. Launch MLflow UI (optional, in a separate terminal)
mlflow ui --port 5000
# Then visit http://localhost:5000

# 7. Start JupyterLab
jupyter lab
```

---

## Notebooks

| # | Notebook | Phase |
|---|---|---|
| 01 | `01_eda.ipynb` | Exploratory Data Analysis |
| 02 | `02_feature_engineering.ipynb` | Feature Engineering |
| 03 | `03_baseline_models.ipynb` | Baseline Models |
| 04 | `04_classical_ml.ipynb` | Classical ML |
| 05 | `05_advanced_boosting.ipynb` | XGBoost / LightGBM / CatBoost + Optuna |
| 06 | `06_neural_networks.ipynb` | PyTorch MPS Neural Networks + TabNet |
| 07 | `07_automl_hyperparameter_tuning.ipynb` | FLAML AutoML + Optuna |
| 08 | `08_imbalance_handling.ipynb` | SMOTE / Threshold Tuning |
| 09 | `09_interpretability.ipynb` | SHAP / LIME / PDP |
| 10 | `10_dimensionality_clustering.ipynb` | PCA / t-SNE / UMAP / Clustering |
| 11 | `11_llm_experiments.ipynb` | Ollama LLM Experiments |
| 12 | `12_final_ensemble.ipynb` | Final Ensemble + Leaderboard |
| 13 | `13_report_generation.ipynb` | Report Compilation → PDF |

---

## Experiment Tracking

All model runs are tracked in MLflow under the experiment **`Heart-Disease-Kaggle`**.

```bash
# Start MLflow UI
source .venv/bin/activate
mlflow ui --port 5000
```

---

## Academic Report

The formal academic report (IEEE/NeurIPS style) is written incrementally in
`report/academic_report.md` and rendered to `report/academic_report.pdf` at the end:

```bash
cd report
pandoc academic_report.md -o academic_report.pdf \
  --bibliography=references.bib \
  --csl=ieee.csl \
  --pdf-engine=xelatex \
  -V geometry:margin=1in
```

---

## Hardware

- **Machine**: Apple M1 MacBook Pro, 16GB Unified Memory
- **Acceleration**: PyTorch MPS for neural network training
- **LLM**: Ollama `hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_S` (11GB, Q4 quantized)
