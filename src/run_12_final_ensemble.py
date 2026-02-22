"""
Phase 12: Final Ensemble — Complete Model Leaderboard
Combines top performers from all phases using voting, stacking, and blending.
"""
import os, json, time
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import load_data, get_X_y

print("=" * 60)
print("Phase 12: Final Ensemble")
print("=" * 60)

# ─── Load data ───────────────────────────────────────────────
train_df = load_data(split="train")
X, y = get_X_y(train_df)
print(f"Data: {X.shape}, positive rate: {y.mean():.3f}")

# ─── Load best params ─────────────────────────────────────────
best_params_path = Path("results/metrics/07_automl_best_params.json")
with open(best_params_path) as f:
    best_params_all = json.load(f)
lgb_params = best_params_all["lgb_tpe_200_best"]

xgb_params_path = Path("results/metrics/05_optuna_best_params.json")
with open(xgb_params_path) as f:
    optuna_params = json.load(f)
xgb_raw = optuna_params.get("xgb_best", optuna_params.get("xgboost", {}))
cat_raw = optuna_params.get("cat_best", optuna_params.get("catboost", {}))
# Handle nested "best_params" structure from phase 5
xgb_params = xgb_raw.get("best_params", xgb_raw) if isinstance(xgb_raw, dict) else {}
cat_params = cat_raw.get("best_params", cat_raw) if isinstance(cat_raw, dict) else {}

# ─── Define base models ──────────────────────────────────────
lgb_best = lgb.LGBMClassifier(**lgb_params, random_state=42, verbose=-1, n_jobs=-1)

xgb_best = xgb.XGBClassifier(
    **{k: v for k, v in xgb_params.items() if k not in ("objective",)},
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
) if xgb_params else xgb.XGBClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
)

cat_best = CatBoostClassifier(
    **{k: v for k, v in cat_params.items()},
    random_seed=42, verbose=0,
) if cat_params else CatBoostClassifier(
    iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=0,
)

lr_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
])

models_individual = [
    ("LightGBM_Optuna", lgb_best),
    ("XGBoost_Optuna", xgb_best),
    ("CatBoost_Optuna", cat_best),
    ("LogisticRegression_L2", lr_scaled),
]

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─── Evaluate individual top models ──────────────────────────
print("\n── Individual top models (5-fold CV) ──")
oof_preds = {}
individual_results = []

for name, model in models_individual:
    t0 = time.time()
    oof = cross_val_predict(model, X, y, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
    auc = roc_auc_score(y, oof)
    preds_bin = (oof >= 0.5).astype(int)
    f1 = f1_score(y, preds_bin)
    rec = recall_score(y, preds_bin)
    elapsed = time.time() - t0
    oof_preds[name] = oof
    individual_results.append({"model": name, "roc_auc": auc, "f1": f1, "recall": rec})
    print(f"  {name}: AUC={auc:.4f}  F1={f1:.4f}  Recall={rec:.4f}  [{elapsed:.0f}s]")

# ─── Ensemble: Soft Voting ────────────────────────────────────
print("\n── Ensemble methods (OOF probability averaging) ──")
ensemble_results = []

def eval_oof(name, probs):
    auc = roc_auc_score(y, probs)
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(y, preds)
    rec = recall_score(y, preds)
    ensemble_results.append({"model": name, "roc_auc": auc, "f1": f1, "recall": rec})
    print(f"  {name}: AUC={auc:.4f}  F1={f1:.4f}  Recall={rec:.4f}")
    return probs

# Soft voting: average probabilities
lgb_oof = oof_preds["LightGBM_Optuna"]
xgb_oof = oof_preds["XGBoost_Optuna"]
cat_oof = oof_preds["CatBoost_Optuna"]
lr_oof  = oof_preds["LogisticRegression_L2"]

eval_oof("SoftVote_LGB+XGB+CAT", (lgb_oof + xgb_oof + cat_oof) / 3)
eval_oof("SoftVote_ALL4",        (lgb_oof + xgb_oof + cat_oof + lr_oof) / 4)
eval_oof("SoftVote_LGB+XGB",     (lgb_oof + xgb_oof) / 2)
eval_oof("SoftVote_LGB+CAT",     (lgb_oof + cat_oof) / 2)

# Rank averaging
def rank_avg(*arrs):
    from scipy.stats import rankdata
    stacked = np.vstack([rankdata(a) / len(a) for a in arrs])
    return stacked.mean(axis=0)

eval_oof("RankAvg_LGB+XGB+CAT", rank_avg(lgb_oof, xgb_oof, cat_oof))
eval_oof("RankAvg_ALL4",        rank_avg(lgb_oof, xgb_oof, cat_oof, lr_oof))

# ─── Stacking with OOF predictions (no data leakage) ─────────
print("\n── Stacking (meta-learner on OOF preds) ──")
stacking_results = []

def stack_oof(name, meta_X_oof, meta_y, meta_model):
    """Train meta-model on OOF predictions using CV."""
    meta_oof = cross_val_predict(meta_model, meta_X_oof, meta_y,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=99),
                                  method="predict_proba")[:, 1]
    auc = roc_auc_score(meta_y, meta_oof)
    preds = (meta_oof >= 0.5).astype(int)
    f1 = f1_score(meta_y, preds)
    rec = recall_score(meta_y, preds)
    stacking_results.append({"model": name, "roc_auc": auc, "f1": f1, "recall": rec})
    print(f"  {name}: AUC={auc:.4f}  F1={f1:.4f}  Recall={rec:.4f}")
    return meta_oof

# Meta-feature matrix from OOF
meta_3 = np.column_stack([lgb_oof, xgb_oof, cat_oof])
meta_4 = np.column_stack([lgb_oof, xgb_oof, cat_oof, lr_oof])

meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15,
                                random_state=42, verbose=-1)

stack_oof("Stack_LR_meta(3)",   meta_3, y, meta_lr)
stack_oof("Stack_LR_meta(4)",   meta_4, y, meta_lr)
stack_oof("Stack_LGB_meta(3)",  meta_3, y, meta_lgb)
stack_oof("Stack_LGB_meta(4)",  meta_4, y, meta_lgb)

# ─── Threshold-tuned best model ──────────────────────────────
print("\n── Threshold optimization on best model ──")
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, lgb_oof)
youden_j = tpr - fpr
best_thresh = thresholds[np.argmax(youden_j)]
preds_tuned = (lgb_oof >= best_thresh).astype(int)
auc_t = roc_auc_score(y, lgb_oof)
f1_t = f1_score(y, preds_tuned)
rec_t = recall_score(y, preds_tuned)
print(f"  LightGBM@threshold={best_thresh:.3f}: AUC={auc_t:.4f}  F1={f1_t:.4f}  Recall={rec_t:.4f}")
ensemble_results.append({
    "model": f"LightGBM_Optuna@t={best_thresh:.3f}",
    "roc_auc": auc_t, "f1": f1_t, "recall": rec_t,
})

# ─── Build full leaderboard ───────────────────────────────────
print("\n── Loading historical results for full leaderboard ──")

def load_csv_results(path, phase, model_col="model"):
    try:
        df = pd.read_csv(path)
        if model_col not in df.columns:
            # try to find model column
            candidates = [c for c in df.columns if "model" in c.lower() or "name" in c.lower()]
            if candidates:
                df = df.rename(columns={candidates[0]: "model"})
        df["phase"] = phase
        return df
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return pd.DataFrame()

phase_files = [
    ("results/metrics/03_baseline_results.csv",      "P3_Baseline"),
    ("results/metrics/04_classical_ml_results.csv",  "P4_ClassicalML"),
    ("results/metrics/05_boosting_results.csv",      "P5_Boosting"),
    ("results/metrics/06_neural_network_results.csv","P6_NeuralNet"),
    ("results/metrics/07_automl_results.csv",        "P7_AutoML"),
    ("results/metrics/08_imbalance_results.csv",     "P8_Imbalance"),
]

hist_dfs = []
for fpath, phase in phase_files:
    df = load_csv_results(fpath, phase)
    if not df.empty:
        hist_dfs.append(df)

hist_df = pd.concat(hist_dfs, ignore_index=True) if hist_dfs else pd.DataFrame()

# Normalize column names
rename_map = {}
for col in hist_df.columns:
    if col == "roc_auc_mean":
        rename_map[col] = "roc_auc"
    elif col == "f1_mean":
        rename_map[col] = "f1"
    elif col == "recall_mean":
        rename_map[col] = "recall"
    elif "auc" in col.lower() and "roc" not in col.lower() and "_std" not in col.lower():
        rename_map[col] = "roc_auc"
hist_df = hist_df.rename(columns=rename_map)

# Ensure required columns exist
for c in ["roc_auc", "f1", "recall"]:
    if c not in hist_df.columns:
        hist_df[c] = np.nan

# Current phase results
cur_df = pd.DataFrame(individual_results + ensemble_results + stacking_results)
cur_df["phase"] = "P12_Ensemble"

# Combine
leaderboard = pd.concat([hist_df, cur_df], ignore_index=True)

# Keep only useful columns
keep_cols = ["phase", "model", "roc_auc", "f1", "recall"]
leaderboard = leaderboard[[c for c in keep_cols if c in leaderboard.columns]]
leaderboard = leaderboard.dropna(subset=["roc_auc"])
leaderboard = leaderboard.sort_values("roc_auc", ascending=False).reset_index(drop=True)

print(f"\nFull leaderboard: {len(leaderboard)} models")
print(leaderboard.head(20).to_string(index=False))

# ─── Save results ─────────────────────────────────────────────
Path("results/metrics").mkdir(parents=True, exist_ok=True)

# Save Phase 12 ensemble results
ensemble_df = pd.DataFrame(individual_results + ensemble_results + stacking_results)
ensemble_df.to_csv("results/metrics/12_ensemble_results.csv", index=False)

# Save full leaderboard
leaderboard.to_csv("results/metrics/12_full_leaderboard.csv", index=False)
print("\nSaved: results/metrics/12_ensemble_results.csv")
print("Saved: results/metrics/12_full_leaderboard.csv")

# ─── Plots ────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

Path("results/figures").mkdir(parents=True, exist_ok=True)

# Plot 1: Top-30 leaderboard
fig, ax = plt.subplots(figsize=(12, 10))
top30 = leaderboard.head(30).copy()
colors = ["#e74c3c" if "P12" in str(p) else "#3498db" for p in top30["phase"]]
bars = ax.barh(range(len(top30)), top30["roc_auc"], color=colors, edgecolor="white", height=0.7)
ax.set_yticks(range(len(top30)))
ax.set_yticklabels([f"{str(r.model)[:40]} [{r.phase}]" for r in top30.itertuples()], fontsize=7)
ax.set_xlabel("ROC-AUC")
ax.set_title("Top-30 Models: Full Leaderboard (red = Phase 12 Ensemble)")
ax.set_xlim(0.48, max(top30["roc_auc"]) + 0.01)
ax.invert_yaxis()
ax.axvline(0.9554, color="gray", linestyle="--", alpha=0.5, label="Best single model (0.9554)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/12_leaderboard_top30.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Ensemble comparison
fig, ax = plt.subplots(figsize=(10, 7))
ens_df = pd.DataFrame(ensemble_results + stacking_results + individual_results)
ens_df = ens_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
c = ["#e74c3c" if any(x in m for x in ["Vote","Stack","Rank","@t="])
     else "#2ecc71" for m in ens_df["model"]]
ax.barh(range(len(ens_df)), ens_df["roc_auc"], color=c, edgecolor="white", height=0.7)
ax.set_yticks(range(len(ens_df)))
ax.set_yticklabels(ens_df["model"], fontsize=9)
ax.set_xlabel("ROC-AUC")
ax.set_title("Ensemble Methods vs Individual Models (Phase 12)")
ax.axvline(0.9554, color="navy", linestyle="--", alpha=0.7, label="Best prior (0.9554)")
ax.legend()
ax.invert_yaxis()
xleft = min(ens_df["roc_auc"]) - 0.001
ax.set_xlim(xleft, max(ens_df["roc_auc"]) + 0.002)
plt.tight_layout()
plt.savefig("results/figures/12_ensemble_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 3: F1 vs AUC scatter for all models
fig, ax = plt.subplots(figsize=(10, 7))
phase_colors = {
    "P3_Baseline": "#95a5a6",
    "P4_ClassicalML": "#3498db",
    "P5_Boosting": "#e67e22",
    "P6_NeuralNet": "#9b59b6",
    "P7_AutoML": "#1abc9c",
    "P8_Imbalance": "#e74c3c",
    "P12_Ensemble": "#2ecc71",
}
for phase, grp in leaderboard.groupby("phase"):
    color = phase_colors.get(phase, "gray")
    ax.scatter(grp["roc_auc"], grp["f1"], label=phase, color=color, alpha=0.7, s=50)
ax.set_xlabel("ROC-AUC")
ax.set_ylabel("F1 Score")
ax.set_title("ROC-AUC vs F1 Score — All Models (colored by phase)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/12_auc_vs_f1_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# ─── Summary stats ────────────────────────────────────────────
best_auc_row = leaderboard.iloc[0]
best_f1_row  = leaderboard.loc[leaderboard["f1"].idxmax()] if "f1" in leaderboard.columns else best_auc_row
best_rec_row = leaderboard.loc[leaderboard["recall"].idxmax()] if "recall" in leaderboard.columns else best_auc_row

summary = {
    "n_models_total": len(leaderboard),
    "best_roc_auc": {"model": best_auc_row["model"], "phase": best_auc_row["phase"],
                     "roc_auc": float(best_auc_row["roc_auc"])},
    "best_f1":      {"model": best_f1_row["model"],  "phase": best_f1_row["phase"],
                     "roc_auc": float(best_f1_row.get("roc_auc", 0)),
                     "f1": float(best_f1_row.get("f1", 0))},
    "best_recall":  {"model": best_rec_row["model"], "phase": best_rec_row["phase"],
                     "roc_auc": float(best_rec_row.get("roc_auc", 0)),
                     "recall": float(best_rec_row.get("recall", 0))},
    "ensemble_results": ensemble_df.to_dict(orient="records"),
}
with open("results/metrics/12_ensemble_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

# ─── MLflow logging ──────────────────────────────────────────
mlflow.set_experiment("Heart-Disease-Kaggle")
with mlflow.start_run(run_name="phase_12_final_ensemble"):
    mlflow.log_metric("best_roc_auc", float(best_auc_row["roc_auc"]))
    mlflow.log_metric("n_models_evaluated", len(leaderboard))
    mlflow.log_artifact("results/metrics/12_full_leaderboard.csv")
    mlflow.log_artifact("results/figures/12_leaderboard_top30.png")

print("\n=== Phase 12 Final Ensemble Complete ===")
print(f"Total models in leaderboard: {len(leaderboard)}")
print(f"Best ROC-AUC: {best_auc_row['model']} [{best_auc_row['phase']}] = {best_auc_row['roc_auc']:.4f}")
print(f"Best F1:      {best_f1_row['model']} = {float(best_f1_row.get('f1',0)):.4f}")
print(f"Best Recall:  {best_rec_row['model']} = {float(best_rec_row.get('recall',0)):.4f}")
