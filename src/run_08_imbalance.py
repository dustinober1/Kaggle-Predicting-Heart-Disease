"""
Phase 8: Class Imbalance Handling
- SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE (all on 50K subsample)
- RandomUnderSampler, TomekLinks
- SMOTEENN, SMOTETomek
- class_weight='balanced' variants
- Threshold tuning via Youden's J and F1-optimal
Results → results/metrics/08_imbalance_results.csv
"""
import sys, warnings, pathlib, json
sys.path.insert(0, '/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                              roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

from src.data_utils import load_data, get_X_y
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

# Subsample 150K for resampling experiments (SMOTE is slow on full data)
rng = np.random.RandomState(42)
idx = rng.choice(len(X_scaled), size=150_000, replace=False)
X_s = X_scaled[idx]
y_s = y_np[idx]
print(f"Working sample: {len(X_s):,} rows  "
      f"class ratio: {(y_s==1).mean():.3f} pos / {(y_s==0).mean():.3f} neg")

# Best LightGBM params from Phase 7
with open(RESULTS_DIR / '07_automl_best_params.json') as f:
    automl_params = json.load(f)
best_params = {**automl_params['lgb_tpe_200_best'],
               'objective': 'binary', 'verbosity': -1, 'n_jobs': -1, 'random_state': 42}

def eval_with_resampler(name, resampler, X, y, params, cv):
    """Run 5-fold CV with resampling inside each fold."""
    aucs, f1s, recalls, precisions = [], [], [], []
    for tr_idx, va_idx in cv.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if resampler is not None:
            try:
                X_tr, y_tr = resampler.fit_resample(X_tr, y_tr)
            except Exception as e:
                print(f"  [WARN] {name} resampling failed: {e}")
                continue
        clf = LGBMClassifier(**params)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_va)[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_va, proba))
        f1s.append(f1_score(y_va, pred))
        recalls.append(recall_score(y_va, pred))
        precisions.append(precision_score(y_va, pred))
    return {
        'roc_auc_mean': float(np.mean(aucs)), 'roc_auc_std': float(np.std(aucs)),
        'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
        'recall_mean': float(np.mean(recalls)), 'recall_std': float(np.std(recalls)),
        'precision_mean': float(np.mean(precisions)), 'precision_std': float(np.std(precisions)),
    }

# ── Baseline (no resampling) ───────────────────────────────────────────────────
print('\nBaseline (no resampling)...')
m = eval_with_resampler('Baseline', None, X_s, y_s, best_params, CV)
log_mlflow_run('LGB Baseline (no resample)', m, params={}, tags={'phase': 'imbalance'})
all_results.append({'strategy': 'Baseline (no resampling)', **m})
print(f"  AUC={m['roc_auc_mean']:.4f}  F1={m['f1_mean']:.4f}  Recall={m['recall_mean']:.4f}")

# ── Class weight balanced ──────────────────────────────────────────────────────
print('class_weight=balanced...')
cw_params = {**best_params, 'class_weight': 'balanced'}
m = eval_with_resampler('ClassWeight', None, X_s, y_s, cw_params, CV)
log_mlflow_run('LGB class_weight=balanced', m, params={'class_weight': 'balanced'}, tags={'phase': 'imbalance'})
all_results.append({'strategy': 'class_weight=balanced', **m})
print(f"  AUC={m['roc_auc_mean']:.4f}  F1={m['f1_mean']:.4f}  Recall={m['recall_mean']:.4f}")

# ── Oversampling strategies ────────────────────────────────────────────────────
oversamplers_main = [
    ('SMOTE', SMOTE(random_state=42, k_neighbors=5)),
    ('ADASYN', ADASYN(random_state=42)),
    ('BorderlineSMOTE', BorderlineSMOTE(random_state=42, kind='borderline-1')),
    # SVMSMOTE on 150K is intractable — use 30K subsample
]
# SVMSMOTE on small subsample only
print('SVMSMOTE (30K subsample due to SVM cost)...')
idx_svm = rng.choice(len(X_s), size=30_000, replace=False)
X_svm, y_svm = X_s[idx_svm], y_s[idx_svm]
m_svm = eval_with_resampler('SVMSMOTE', SVMSMOTE(random_state=42, m_neighbors=10, k_neighbors=5),
                             X_svm, y_svm, best_params, CV)
log_mlflow_run('LGB SVMSMOTE', m_svm, params={'strategy': 'SVMSMOTE', 'subsample': 30000},
               tags={'phase': 'imbalance', 'type': 'oversample'})
all_results.append({'strategy': 'SVMSMOTE (30K)', **m_svm})
print(f"  AUC={m_svm['roc_auc_mean']:.4f}  F1={m_svm['f1_mean']:.4f}  Recall={m_svm['recall_mean']:.4f}")

for name, sampler in oversamplers_main:
    print(f'{name}...')
    m = eval_with_resampler(name, sampler, X_s, y_s, best_params, CV)
    log_mlflow_run(f'LGB {name}', m, params={'strategy': name}, tags={'phase': 'imbalance', 'type': 'oversample'})
    all_results.append({'strategy': name, **m})
    print(f"  AUC={m['roc_auc_mean']:.4f}  F1={m['f1_mean']:.4f}  Recall={m['recall_mean']:.4f}")
undersamplers = [
    ('RandomUnderSampler', RandomUnderSampler(random_state=42)),
    ('TomekLinks', TomekLinks(n_jobs=-1)),
]
for name, sampler in undersamplers:
    print(f'{name}...')
    m = eval_with_resampler(name, sampler, X_s, y_s, best_params, CV)
    log_mlflow_run(f'LGB {name}', m, params={'strategy': name}, tags={'phase': 'imbalance', 'type': 'undersample'})
    all_results.append({'strategy': name, **m})
    print(f"  AUC={m['roc_auc_mean']:.4f}  F1={m['f1_mean']:.4f}  Recall={m['recall_mean']:.4f}")

# ── Combined strategies ────────────────────────────────────────────────────────
combined = [
    ('SMOTEENN', SMOTEENN(random_state=42)),
    ('SMOTETomek', SMOTETomek(random_state=42)),
]
for name, sampler in combined:
    print(f'{name}...')
    m = eval_with_resampler(name, sampler, X_s, y_s, best_params, CV)
    log_mlflow_run(f'LGB {name}', m, params={'strategy': name}, tags={'phase': 'imbalance', 'type': 'combined'})
    all_results.append({'strategy': name, **m})
    print(f"  AUC={m['roc_auc_mean']:.4f}  F1={m['f1_mean']:.4f}  Recall={m['recall_mean']:.4f}")

# ── Threshold Tuning ───────────────────────────────────────────────────────────
print('\nThreshold tuning on held-out fold...')
from sklearn.model_selection import train_test_split
X_tr_main, X_val_main, y_tr_main, y_val_main = train_test_split(
    X_s, y_s, test_size=0.2, stratify=y_s, random_state=42)
clf_base = LGBMClassifier(**best_params)
clf_base.fit(X_tr_main, y_tr_main)
proba_val = clf_base.predict_proba(X_val_main)[:, 1]

# Youden's J — maximize (TPR - FPR)
fpr, tpr, thresholds = roc_curve(y_val_main, proba_val)
youdens_j = tpr - fpr
best_thresh_youden = float(thresholds[np.argmax(youdens_j)])
pred_youden = (proba_val >= best_thresh_youden).astype(int)

# F1-optimal threshold
prec, rec, thresh_pr = precision_recall_curve(y_val_main, proba_val)
f1_curve = 2 * prec * rec / (prec + rec + 1e-8)
best_thresh_f1 = float(thresh_pr[np.argmax(f1_curve[:-1])])
pred_f1 = (proba_val >= best_thresh_f1).astype(int)

# Recall-optimal @ 95% recall floor
recall_thresh_idx = np.where(rec >= 0.95)[0]
if len(recall_thresh_idx) > 0:
    best_thresh_recall = float(thresh_pr[recall_thresh_idx[-1]])
else:
    best_thresh_recall = 0.3
pred_recall = (proba_val >= best_thresh_recall).astype(int)

print(f"  Default (0.5): AUC={roc_auc_score(y_val_main, proba_val):.4f}  "
      f"F1={f1_score(y_val_main, (proba_val>=0.5).astype(int)):.4f}  "
      f"Recall={recall_score(y_val_main, (proba_val>=0.5).astype(int)):.4f}")
print(f"  Youden (t={best_thresh_youden:.3f}): F1={f1_score(y_val_main, pred_youden):.4f}  "
      f"Recall={recall_score(y_val_main, pred_youden):.4f}")
print(f"  F1-optimal (t={best_thresh_f1:.3f}): F1={f1_score(y_val_main, pred_f1):.4f}  "
      f"Recall={recall_score(y_val_main, pred_f1):.4f}")
print(f"  Recall@95% (t={best_thresh_recall:.3f}): F1={f1_score(y_val_main, pred_recall):.4f}  "
      f"Recall={recall_score(y_val_main, pred_recall):.4f}")

threshold_results = {
    'default_0.5': {'threshold': 0.5, 'f1': float(f1_score(y_val_main, (proba_val>=0.5).astype(int))),
                    'recall': float(recall_score(y_val_main, (proba_val>=0.5).astype(int)))},
    'youden': {'threshold': best_thresh_youden, 'f1': float(f1_score(y_val_main, pred_youden)),
               'recall': float(recall_score(y_val_main, pred_youden))},
    'f1_optimal': {'threshold': best_thresh_f1, 'f1': float(f1_score(y_val_main, pred_f1)),
                   'recall': float(recall_score(y_val_main, pred_f1))},
    'recall_95': {'threshold': best_thresh_recall, 'f1': float(f1_score(y_val_main, pred_recall)),
                  'recall': float(recall_score(y_val_main, pred_recall))},
}
with open(RESULTS_DIR / '08_threshold_results.json', 'w') as f:
    json.dump(threshold_results, f, indent=2)

# ── Save Results ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_DIR / '08_imbalance_results.csv', index=False)

print('\n=== IMBALANCE STRATEGY LEADERBOARD ===')
print(results_df[['strategy','roc_auc_mean','f1_mean','recall_mean','precision_mean']].to_string(index=False))

# ── Comparison bar chart ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_df = results_df.sort_values('roc_auc_mean', ascending=True)
colors = sns.color_palette(PALETTE, n_colors=len(plot_df))
for ax, (mc, sc, title) in zip(axes, [('roc_auc_mean','roc_auc_std','ROC-AUC'),
                                        ('f1_mean','f1_std','F1 Score'),
                                        ('recall_mean','recall_std','Recall')]):
    bars = ax.barh(plot_df['strategy'], plot_df[mc], xerr=plot_df[sc], color=colors, capsize=3)
    ax.set_xlabel(title); ax.set_title(f'{title} — Imbalance Strategies')
    ax.set_xlim(max(0, plot_df[mc].min()-0.02), 1.0)
    for bar, val in zip(bars, plot_df[mc]):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=7)
fig.suptitle('Imbalance Handling — 5-fold CV (150K subsample)', fontsize=13)
fig.tight_layout()
fig.savefig(FIGS_DIR / '08_imbalance_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Precision-Recall & threshold curve ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(rec, prec, lw=2, color='steelblue', label='PR curve')
axes[0].axvline(x=0.95, color='coral', linestyle='--', label='Recall=0.95')
axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curve'); axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(thresholds, tpr[1:], label='TPR', color='steelblue', lw=2)
axes[1].plot(thresholds, fpr[1:], label='FPR', color='coral', lw=2)
axes[1].plot(thresholds, tpr[1:]-fpr[1:], label="Youden's J", color='green', lw=2, linestyle='--')
axes[1].axvline(x=best_thresh_youden, color='gray', linestyle=':', label=f'Best t={best_thresh_youden:.3f}')
axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('Rate')
axes[1].set_title("Threshold vs TPR/FPR/Youden's J"); axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / '08_threshold_curves.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print('\nAll results saved.')
