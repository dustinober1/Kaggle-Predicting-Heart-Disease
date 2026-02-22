"""
Phase 7: AutoML & Hyperparameter Optimization
- FLAML AutoML (300s, 600s budgets)
- Optuna 200-trial extended search on LightGBM
- CMA-ES sampler comparison
- Feature set ablation study
Results → results/metrics/07_automl_results.csv
"""
import sys, warnings, pathlib, time, json
sys.path.insert(0, '/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from flaml import AutoML

from src.data_utils import load_data, get_X_y, add_engineered_features, get_feature_sets
from src.evaluation import log_mlflow_run
from src.visualization import save_fig, PALETTE

sns.set_theme(style='whitegrid', palette=PALETTE)
mlflow.set_tracking_uri('file:/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease/mlruns')
mlflow.set_experiment('Heart-Disease-Kaggle')

RESULTS_DIR = pathlib.Path('results/metrics')
FIGS_DIR = pathlib.Path('results/figures')
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = []

# ── Load data ──────────────────────────────────────────────────────────────────
train = load_data('train')
X_raw, y = get_X_y(train, extra_features=False)
y_np = y.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.values).astype(np.float32)

# Engineered features
train_eng = add_engineered_features(train.copy())
X_eng, _ = get_X_y(train_eng, extra_features=True)
scaler_eng = StandardScaler()
X_eng_scaled = scaler_eng.fit_transform(X_eng.values).astype(np.float32)

print(f"Data loaded: {X_raw.shape[0]:,} rows, {X_raw.shape[1]} raw features, {X_eng.shape[1]} engineered features")

# ── FLAML AutoML ───────────────────────────────────────────────────────────────
for budget, label in [(300, 'FLAML_300s'), (600, 'FLAML_600s')]:
    print(f'\n── {label} (budget={budget}s) ──')
    automl = AutoML()
    t0 = time.time()
    automl.fit(
        X_scaled, y_np,
        task='classification',
        metric='roc_auc',
        time_budget=budget,
        estimator_list=['lgbm', 'xgboost', 'rf', 'extra_tree', 'catboost'],
        n_splits=5,
        eval_method='cv',
        seed=42,
        verbose=0
    )
    elapsed = time.time() - t0
    best_model_name = automl.best_estimator
    best_loss = automl.best_loss
    best_auc = 1 - best_loss  # FLAML returns 1-AUC as loss for roc_auc metric

    # Get CV AUC with best model on full data
    best_config = automl.best_config
    print(f"  Best estimator: {best_model_name}")
    print(f"  Best ROC-AUC (CV): {best_auc:.4f}")
    print(f"  Best config: {best_config}")

    metrics = {'roc_auc_mean': best_auc, 'roc_auc_std': 0.0, 'f1_mean': 0.0, 'f1_std': 0.0,
               'recall_mean': 0.0, 'recall_std': 0.0}
    params = {'budget_s': budget, 'best_estimator': best_model_name, **{k: str(v) for k, v in best_config.items()}}
    log_mlflow_run(label, metrics, params=params, tags={'phase': 'automl'})
    all_results.append({'model': label, 'phase': 'automl', 'roc_auc_mean': best_auc,
                        'roc_auc_std': 0.0, 'f1_mean': 0.0, 'f1_std': 0.0,
                        'recall_mean': 0.0, 'recall_std': 0.0,
                        'notes': f'{best_model_name}, {elapsed:.0f}s'})

# Subsample for Optuna trials (fast) — final CV validated on full data
rng = np.random.RandomState(42)
idx_opt = rng.choice(len(X_raw), size=50_000, replace=False)
X_opt = X_scaled[idx_opt].astype(np.float32)
y_opt = y_np[idx_opt]
print(f"Optuna subsample: {len(X_opt):,} rows for trial search")

# ── Optuna 100-trial LightGBM (TPE) ───────────────────────────────────────────
print('\n── Optuna LightGBM 100 trials (TPE, 50K subsample) ──')

def lgb_objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_jobs': -1, 'random_state': 42
    }
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(**params)
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_opt, y_opt, cv=cv3, scoring='roc_auc', n_jobs=1)
    return scores.mean()

study_tpe = optuna.create_study(
    direction='maximize',
    storage=f'sqlite:///results/optuna.db',
    study_name='lgbm_tpe_200_v2',
    load_if_exists=True,
    sampler=optuna.samplers.TPESampler(seed=42)
)
t0 = time.time()
study_tpe.optimize(lgb_objective, n_trials=100, show_progress_bar=False)
elapsed = time.time() - t0
best_tpe = study_tpe.best_value
print(f"  TPE best (3-fold): {best_tpe:.4f}  [{elapsed:.0f}s]")

# Validate with 5-fold
from lightgbm import LGBMClassifier
best_params_tpe = {**study_tpe.best_params, 'objective': 'binary', 'verbosity': -1, 'n_jobs': -1, 'random_state': 42}
lgb_tpe = LGBMClassifier(**best_params_tpe)
scores_tpe = cross_val_score(lgb_tpe, X_scaled, y_np, cv=CV, scoring='roc_auc', n_jobs=-1)
print(f"  TPE 5-fold AUC: {scores_tpe.mean():.4f}±{scores_tpe.std():.4f}")
log_mlflow_run('LightGBM Optuna TPE 200', 
               {'roc_auc_mean': float(scores_tpe.mean()), 'roc_auc_std': float(scores_tpe.std()),
                'f1_mean': 0.0, 'f1_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0},
               params=study_tpe.best_params, tags={'phase': 'automl', 'sampler': 'TPE'})
all_results.append({'model': 'LightGBM_Optuna_TPE_200', 'phase': 'automl',
                    'roc_auc_mean': float(scores_tpe.mean()), 'roc_auc_std': float(scores_tpe.std()),
                    'f1_mean': 0.0, 'f1_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                    'notes': f'TPE, {elapsed:.0f}s'})

# ── Optuna CMA-ES sampler ──────────────────────────────────────────────────────
print('\n── Optuna LightGBM 50 trials (CMA-ES, 50K subsample) ──')
study_cmaes = optuna.create_study(
    direction='maximize',
    storage='sqlite:///results/optuna.db',
    study_name='lgbm_cmaes_100_v2',
    load_if_exists=True,
    sampler=optuna.samplers.CmaEsSampler(seed=42)
)
t0 = time.time()
study_cmaes.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
elapsed = time.time() - t0
best_cmaes = study_cmaes.best_value
print(f"  CMA-ES best (3-fold): {best_cmaes:.4f}  [{elapsed:.0f}s]")

best_params_cmaes = {**study_cmaes.best_params, 'objective': 'binary', 'verbosity': -1, 'n_jobs': -1, 'random_state': 42}
lgb_cmaes = LGBMClassifier(**best_params_cmaes)
scores_cmaes = cross_val_score(lgb_cmaes, X_scaled, y_np, cv=CV, scoring='roc_auc', n_jobs=-1)
print(f"  CMA-ES 5-fold AUC: {scores_cmaes.mean():.4f}±{scores_cmaes.std():.4f}")
log_mlflow_run('LightGBM Optuna CMA-ES 100',
               {'roc_auc_mean': float(scores_cmaes.mean()), 'roc_auc_std': float(scores_cmaes.std()),
                'f1_mean': 0.0, 'f1_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0},
               params=study_cmaes.best_params, tags={'phase': 'automl', 'sampler': 'CMA-ES'})
all_results.append({'model': 'LightGBM_Optuna_CMAES_100', 'phase': 'automl',
                    'roc_auc_mean': float(scores_cmaes.mean()), 'roc_auc_std': float(scores_cmaes.std()),
                    'f1_mean': 0.0, 'f1_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                    'notes': f'CMA-ES, {elapsed:.0f}s'})

# ── Feature Set Ablation ───────────────────────────────────────────────────────
print('\n── Feature Set Ablation (LightGBM default) ──')
from src.data_utils import CONTINUOUS_COLS, BINARY_COLS, ORDINAL_COLS

# Use best_params_tpe for consistent comparison
lgb_ablation = LGBMClassifier(**best_params_tpe)

feature_sets = {
    'raw_13': X_raw.values,
    'scaled_13': X_scaled,
    'engineered_18': X_eng_scaled,
}
ablation_results = {}
for fs_name, X_fs in feature_sets.items():
    scores = cross_val_score(lgb_ablation, X_fs.astype(np.float32), y_np, cv=CV, scoring='roc_auc', n_jobs=-1)
    ablation_results[fs_name] = {'mean': float(scores.mean()), 'std': float(scores.std())}
    print(f"  {fs_name}: {scores.mean():.4f}±{scores.std():.4f}")
    all_results.append({'model': f'LGB_ablation_{fs_name}', 'phase': 'ablation',
                        'roc_auc_mean': float(scores.mean()), 'roc_auc_std': float(scores.std()),
                        'f1_mean': 0.0, 'f1_std': 0.0, 'recall_mean': 0.0, 'recall_std': 0.0,
                        'notes': f'ablation, {X_fs.shape[1]} features'})

# Save ablation results
with open(RESULTS_DIR / '07_ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=2)

# Save best params
best_automl_params = {
    'lgb_tpe_200_best': study_tpe.best_params,
    'lgb_cmaes_100_best': study_cmaes.best_params,
    'tpe_5fold_auc': float(scores_tpe.mean()),
    'cmaes_5fold_auc': float(scores_cmaes.mean()),
    'ablation': ablation_results
}
with open(RESULTS_DIR / '07_automl_best_params.json', 'w') as f:
    json.dump(best_automl_params, f, indent=2)

# ── Save Results ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_DIR / '07_automl_results.csv', index=False)

print('\n=== AUTOML LEADERBOARD ===')
print(results_df[['model', 'roc_auc_mean', 'roc_auc_std', 'notes']].to_string(index=False))

# ── Optuna convergence plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tpe_vals = [t.value for t in study_tpe.trials if t.value is not None]
cmaes_vals = [t.value for t in study_cmaes.trials if t.value is not None]
tpe_best = np.maximum.accumulate(tpe_vals)
cmaes_best = np.maximum.accumulate(cmaes_vals)

axes[0].plot(tpe_best, label='TPE best-so-far', color='steelblue', lw=2)
axes[0].scatter(range(len(tpe_vals)), tpe_vals, alpha=0.2, s=8, color='steelblue')
axes[0].set_xlabel('Trial'); axes[0].set_ylabel('ROC-AUC (3-fold)')
axes[0].set_title('LightGBM Optuna TPE (200 trials)'); axes[0].legend()

axes[1].plot(cmaes_best, label='CMA-ES best-so-far', color='coral', lw=2)
axes[1].scatter(range(len(cmaes_vals)), cmaes_vals, alpha=0.2, s=8, color='coral')
axes[1].set_xlabel('Trial'); axes[1].set_ylabel('ROC-AUC (3-fold)')
axes[1].set_title('LightGBM CMA-ES (100 trials)'); axes[1].legend()

fig.tight_layout()
fig.savefig(FIGS_DIR / '07_optuna_convergence.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Ablation bar chart
fig, ax = plt.subplots(figsize=(9, 4))
names = list(ablation_results.keys())
means = [ablation_results[k]['mean'] for k in names]
stds = [ablation_results[k]['std'] for k in names]
colors = sns.color_palette(PALETTE, n_colors=len(names))
bars = ax.barh(names, means, xerr=stds, color=colors, capsize=4)
ax.set_xlabel('ROC-AUC (5-fold CV)')
ax.set_title('Feature Set Ablation — LightGBM (Optuna best params)')
ax.set_xlim(max(0, min(means) - 0.01), 1.0)
for bar, val in zip(bars, means):
    ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
fig.tight_layout()
fig.savefig(FIGS_DIR / '07_feature_ablation.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print('\nAll results saved.')
