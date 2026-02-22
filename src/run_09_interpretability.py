"""
Phase 9: Interpretability & Explainability
- SHAP (global summary, beeswarm, bar, local waterfall)
- LIME (10 individual predictions)
- PDP + ICE plots (top 5 features)
- Feature importance comparison (gain vs permutation vs SHAP)
- Calibration curves
- ROC overlay of all model families
- Confusion matrix deep-dive
Results → results/figures/09_* and results/metrics/09_*
"""
import sys, warnings, pathlib, json
sys.path.insert(0, '/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                              confusion_matrix, ConfusionMatrixDisplay,
                              roc_curve)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import lime
import lime.lime_tabular

from src.data_utils import load_data, get_X_y
from src.visualization import save_fig, PALETTE

sns.set_theme(style='whitegrid', palette=PALETTE)
mlflow.set_tracking_uri('file:/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease/mlruns')
FIGS_DIR = pathlib.Path('results/figures')
RESULTS_DIR = pathlib.Path('results/metrics')

# ── Load data ──────────────────────────────────────────────────────────────────
train = load_data('train')
X_raw, y = get_X_y(train, extra_features=False)
feature_names = list(X_raw.columns)
y_np = y.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.values).astype(np.float32)

# 10K SHAP subsample, 30K for calibration
rng = np.random.RandomState(42)
idx_shap = rng.choice(len(X_scaled), size=10_000, replace=False)
idx_cal = rng.choice(len(X_scaled), size=100_000, replace=False)

X_shap, y_shap = X_scaled[idx_shap], y_np[idx_shap]
X_cal, y_cal = X_scaled[idx_cal], y_np[idx_cal]
X_tr_cal, X_va_cal, y_tr_cal, y_va_cal = train_test_split(
    X_cal, y_cal, test_size=0.2, stratify=y_cal, random_state=42)

print(f"SHAP sample: {len(X_shap):,} | Calibration split: {len(X_tr_cal):,} train / {len(X_va_cal):,} val")

# ── Best LightGBM (refit on 100K) ─────────────────────────────────────────────
with open(RESULTS_DIR / '07_automl_best_params.json') as f:
    ap = json.load(f)
lgb_params = {**ap['lgb_tpe_200_best'], 'objective': 'binary', 'verbosity': -1, 'n_jobs': -1, 'random_state': 42}
lgb_model = LGBMClassifier(**lgb_params)
lgb_model.fit(X_tr_cal, y_tr_cal)
lgb_proba = lgb_model.predict_proba(X_va_cal)[:, 1]
print(f"LGB val AUC: {roc_auc_score(y_va_cal, lgb_proba):.4f}")

# ── SHAP Analysis ─────────────────────────────────────────────────────────────
print('\nComputing SHAP values (TreeExplainer, 10K sample)...')
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_shap)
# LightGBM binary returns list [neg_class, pos_class]
if isinstance(shap_values, list):
    shap_pos = shap_values[1]
else:
    shap_pos = shap_values
shap_df = pd.DataFrame(shap_pos, columns=feature_names)

# Global mean |SHAP| bar chart
mean_abs_shap = np.abs(shap_pos).mean(axis=0)
shap_importance = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs_shap})
shap_importance = shap_importance.sort_values('mean_abs_shap', ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = sns.color_palette(PALETTE, n_colors=len(shap_importance))
bars = ax.barh(shap_importance['feature'], shap_importance['mean_abs_shap'], color=colors)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('Global Feature Importance (SHAP) — LightGBM (10K sample)')
for bar, val in zip(bars, shap_importance['mean_abs_shap']):
    ax.text(val + 0.0002, bar.get_y()+bar.get_height()/2, f'{val:.4f}', va='center', fontsize=8)
fig.tight_layout()
fig.savefig(FIGS_DIR / '09_shap_global_bar.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved 09_shap_global_bar.png")

# SHAP beeswarm summary plot
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_pos, X_shap, feature_names=feature_names, show=False, max_display=13, plot_type='dot')
plt.tight_layout()
plt.savefig(FIGS_DIR / '09_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 09_shap_beeswarm.png")

# Local waterfall — 3 correct positives, 3 wrong predictions
lgb_proba_shap = lgb_model.predict_proba(X_shap)[:, 1]
lgb_pred_shap = (lgb_proba_shap >= 0.5).astype(int)
correct_pos = np.where((y_shap == 1) & (lgb_pred_shap == 1))[0][:3]
false_neg = np.where((y_shap == 1) & (lgb_pred_shap == 0))[0][:3]

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for i, (idx_row, subtitle) in enumerate([(correct_pos, 'Correct Positive'), (false_neg, 'False Negative')]):
    for j, row_idx in enumerate(idx_row):
        ax = axes[i, j]
        top5_feat = np.argsort(np.abs(shap_pos[row_idx]))[-5:]
        feat_names_5 = [feature_names[k] for k in top5_feat]
        shap_5 = shap_pos[row_idx][top5_feat]
        colors_5 = ['steelblue' if v > 0 else 'coral' for v in shap_5]
        ax.barh(feat_names_5, shap_5, color=colors_5)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title(f'{subtitle} #{j+1}\ny={y_shap[row_idx]}, p={lgb_proba_shap[row_idx]:.3f}', fontsize=9)
        ax.set_xlabel('SHAP value', fontsize=8)
fig.suptitle('Local SHAP Explanations (Top-5 features per instance)', fontsize=12)
fig.tight_layout()
fig.savefig(FIGS_DIR / '09_shap_local_waterfall.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved 09_shap_local_waterfall.png")

# Save SHAP importance to metrics
shap_importance_sorted = shap_importance.sort_values('mean_abs_shap', ascending=False)
shap_importance_sorted.to_csv(RESULTS_DIR / '09_shap_importance.csv', index=False)

# ── LIME Explanations ──────────────────────────────────────────────────────────
print('\nComputing LIME explanations (10 instances)...')
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_tr_cal,
    feature_names=feature_names,
    class_names=['Absence', 'Presence'],
    mode='classification',
    random_state=42
)

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()
lime_indices = list(correct_pos[:5]) + list(false_neg[:3]) + [
    np.where((y_shap == 0) & (lgb_pred_shap == 1))[0][0],
    np.where((y_shap == 0) & (lgb_pred_shap == 0))[0][0]
]
for i, row_idx in enumerate(lime_indices[:10]):
    exp = lime_explainer.explain_instance(
        X_shap[row_idx], lgb_model.predict_proba, num_features=5, top_labels=1)
    label = list(exp.as_map().keys())[0]
    lime_feats = exp.as_list(label=label)
    names_l = [lf[0] for lf in lime_feats]
    vals_l = [lf[1] for lf in lime_feats]
    colors_l = ['steelblue' if v > 0 else 'coral' for v in vals_l]
    axes[i].barh(names_l, vals_l, color=colors_l)
    axes[i].axvline(0, color='black', lw=0.8)
    y_true_l = y_shap[row_idx]; p_l = lgb_proba_shap[row_idx]
    axes[i].set_title(f'y={y_true_l}, p={p_l:.2f}', fontsize=8)
    axes[i].tick_params(labelsize=6)
fig.suptitle('LIME Local Explanations (10 instances)', fontsize=12)
fig.tight_layout()
fig.savefig(FIGS_DIR / '09_lime_explanations.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved 09_lime_explanations.png")

# ── PDP + ICE (Top 5 SHAP features) ───────────────────────────────────────────
print('\nGenerating PDP/ICE plots...')
top5_features = shap_importance.sort_values('mean_abs_shap', ascending=False)['feature'].head(5).tolist()
top5_indices = [feature_names.index(f) for f in top5_features]

# Use sklearn's PartialDependenceDisplay with a smaller sample (5K)
idx_pdp = rng.choice(len(X_tr_cal), size=5_000, replace=False)
X_pdp = X_tr_cal[idx_pdp]

from sklearn.inspection import partial_dependence
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
for col_i, (feat_idx, feat_name) in enumerate(zip(top5_indices, top5_features)):
    pd_results = partial_dependence(lgb_model, X_pdp, [feat_idx], kind='both',
                                     grid_resolution=50, percentiles=(0.05, 0.95))
    grid_vals = pd_results['grid_values'][0]
    avg_pd = pd_results['average'][0]
    ice_vals = pd_results['individual'][0]

    ax_avg = axes[0, col_i]
    ax_ice = axes[1, col_i]

    # ICE (subsample 100 lines)
    for ice_line in ice_vals[:100]:
        ax_ice.plot(grid_vals, ice_line, alpha=0.05, color='steelblue', lw=0.5)
    ax_ice.plot(grid_vals, avg_pd, color='red', lw=2, label='PDP avg')
    ax_ice.set_xlabel(feat_name, fontsize=9); ax_ice.set_title(f'ICE: {feat_name}', fontsize=9)
    ax_ice.legend(fontsize=7)

    # PDP only
    ax_avg.plot(grid_vals, avg_pd, color='steelblue', lw=2)
    ax_avg.fill_between(grid_vals, avg_pd - avg_pd.std(), avg_pd + avg_pd.std(), alpha=0.2)
    ax_avg.set_title(f'PDP: {feat_name}', fontsize=9)
    ax_avg.set_xlabel(feat_name, fontsize=9); ax_avg.set_ylabel('Partial Dep.', fontsize=8)

fig.suptitle('Partial Dependence Plots & ICE — Top 5 SHAP Features', fontsize=12)
fig.tight_layout()
fig.savefig(FIGS_DIR / '09_pdp_ice.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved 09_pdp_ice.png")

# ── Feature Importance Comparison ─────────────────────────────────────────────
print('\nFeature importance comparison...')
# Gain importance from LightGBM
gain_imp = pd.Series(lgb_model.feature_importances_, index=feature_names)
# Permutation importance (5K subsample)
from sklearn.inspection import permutation_importance
perm_result = permutation_importance(lgb_model, X_shap, y_shap, n_repeats=10,
                                     scoring='roc_auc', random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm_result.importances_mean, index=feature_names)

# Normalize to [0,1]
gain_norm = (gain_imp - gain_imp.min()) / (gain_imp.max() - gain_imp.min())
perm_norm = (perm_imp - perm_imp.min()) / (perm_imp.max() - perm_imp.min())
shap_norm_series = pd.Series(mean_abs_shap, index=feature_names)
shap_norm_s = (shap_norm_series - shap_norm_series.min()) / (shap_norm_series.max() - shap_norm_series.min())

imp_df = pd.DataFrame({'Gain (normalized)': gain_norm, 'Permutation (normalized)': perm_norm,
                       'SHAP (normalized)': shap_norm_s}).sort_values('SHAP (normalized)', ascending=False)

fig, ax = plt.subplots(figsize=(11, 7))
x = np.arange(len(imp_df))
w = 0.28
bars1 = ax.bar(x - w, imp_df['Gain (normalized)'], width=w, label='Gain', color='steelblue', alpha=0.8)
bars2 = ax.bar(x, imp_df['Permutation (normalized)'], width=w, label='Permutation', color='coral', alpha=0.8)
bars3 = ax.bar(x + w, imp_df['SHAP (normalized)'], width=w, label='SHAP', color='seagreen', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(imp_df.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Normalized Importance'); ax.set_title('Feature Importance: Gain vs Permutation vs SHAP')
ax.legend(); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / '09_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
imp_df.to_csv(RESULTS_DIR / '09_feature_importance_comparison.csv')
print("  Saved 09_importance_comparison.png")

# ── Calibration Curves ─────────────────────────────────────────────────────────
print('\nCalibration analysis...')
models_cal = {
    'LightGBM': lgb_model,
    'LR': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
}
# Also calibrated LightGBM
lgb_cal = CalibratedClassifierCV(lgb_model, method='isotonic', cv='prefit')
lgb_cal.fit(X_va_cal, y_va_cal)
models_cal['LGB Calibrated (Isotonic)'] = lgb_cal

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, model in models_cal.items():
    if name == 'LR':
        model.fit(X_tr_cal, y_tr_cal)
    proba_ = model.predict_proba(X_va_cal)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_va_cal, proba_, n_bins=10)
    axes[0].plot(mean_predicted_value, fraction_of_positives, marker='o', lw=1.5, label=name)
axes[0].plot([0,1],[0,1], 'k--', lw=1, label='Perfectly calibrated')
axes[0].set_xlabel('Mean predicted probability'); axes[0].set_ylabel('Fraction of positives')
axes[0].set_title('Calibration Curves'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# Histogram of predicted probabilities
lgb_proba_all = lgb_model.predict_proba(X_va_cal)[:, 1]
axes[1].hist(lgb_proba_all[y_va_cal==0], bins=50, alpha=0.6, density=True, label='Absence', color='steelblue')
axes[1].hist(lgb_proba_all[y_va_cal==1], bins=50, alpha=0.6, density=True, label='Presence', color='coral')
axes[1].set_xlabel('Predicted Probability'); axes[1].set_ylabel('Density')
axes[1].set_title('Probability Distribution by True Class'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / '09_calibration.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved 09_calibration.png")

# ── ROC Overlay ────────────────────────────────────────────────────────────────
print('\nROC overlay plot...')
# Load results from multiple phases
phases_results = {}
for fname, phase_label in [
    ('03_baseline_results.csv', 'Baseline'),
    ('04_classical_ml_results.csv', 'Classical ML'),
    ('05_boosting_results.csv', 'Boosting'),
    ('06_neural_network_results.csv', 'Neural Networks'),
]:
    df = pd.read_csv(RESULTS_DIR / fname)
    phases_results[phase_label] = df

fig, ax = plt.subplots(figsize=(9, 7))
# Best from each phase — just plot scatter of AUC values
all_models_df = pd.concat(
    [df.assign(phase=ph) for ph, df in phases_results.items()], ignore_index=True)
palette = sns.color_palette(PALETTE, n_colors=len(phases_results))
for i, (phase, grp) in enumerate(all_models_df.groupby('phase')):
    grp_sorted = grp.sort_values('roc_auc_mean', ascending=False)
    best = grp_sorted.iloc[0]
    y_pred = lgb_model.predict_proba(X_va_cal)[:, 1]  # representative model
    ax.scatter(best['roc_auc_mean'], i, s=100, zorder=5, color=palette[i], label=f"{phase}: best AUC={best['roc_auc_mean']:.4f}")
ax.set_xlabel('ROC-AUC (5-fold CV mean)')
ax.set_yticks(range(len(phases_results))); ax.set_yticklabels(list(phases_results.keys()))
ax.set_title('Best ROC-AUC per Model Family'); ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3); ax.set_xlim(0.90, 0.97)
fig.tight_layout()
fig.savefig(FIGS_DIR / '09_roc_auc_by_phase.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved 09_roc_auc_by_phase.png")

# ── Confusion Matrix Deep-Dive ─────────────────────────────────────────────────
print('\nConfusion matrix analysis...')
thresholds_to_test = [0.3, 0.4, 0.45, 0.5, 0.55]
fig, axes = plt.subplots(1, len(thresholds_to_test), figsize=(18, 4))
for ax, t in zip(axes, thresholds_to_test):
    pred_t = (lgb_proba >= t).astype(int)
    cm = confusion_matrix(y_va_cal, pred_t)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Absence', 'Presence'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    auc_t = roc_auc_score(y_va_cal, lgb_proba)
    f1_t = f1_score(y_va_cal, pred_t)
    rec_t = recall_score(y_va_cal, pred_t)
    ax.set_title(f't={t}\nF1={f1_t:.3f} Rec={rec_t:.3f}', fontsize=9)
fig.suptitle('Confusion Matrices at Different Decision Thresholds', fontsize=11)
fig.tight_layout()
fig.savefig(FIGS_DIR / '09_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("  Saved 09_confusion_matrices.png")

print('\n=== Phase 9 Interpretability Complete ===')
print(f"Top SHAP features: {shap_importance.sort_values('mean_abs_shap', ascending=False)['feature'].head(5).tolist()}")
