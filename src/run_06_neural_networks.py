"""
Neural network experiments — runs 4 MLP architectures + TabNet via 5-fold CV.
Results saved to results/metrics/06_neural_network_results.csv
"""
import sys
sys.path.insert(0, '/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow, json, pathlib, time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score

from pytorch_tabnet.tab_model import TabNetClassifier

from src.data_utils import load_data, get_X_y
from src.evaluation import log_mlflow_run
from src.visualization import save_fig, PALETTE

sns.set_theme(style='whitegrid', palette=PALETTE)
mlflow.set_tracking_uri('file:/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease/mlruns')
mlflow.set_experiment('Heart-Disease-Kaggle')

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

train = load_data('train')
X_raw, y = get_X_y(train, extra_features=False)

# Subsample to 150K for NN training — representative & memory-friendly on M1 16GB
# Full 630K is overkill for NN convergence testing; boosting handles the full set
rng = np.random.RandomState(42)
idx_sample = rng.choice(len(X_raw), size=150_000, replace=False)
idx_sample.sort()
X_raw = X_raw.iloc[idx_sample].reset_index(drop=True)
y = y.iloc[idx_sample].reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.values).astype(np.float32)
y_np = y.values.astype(np.int64)
N_FEATURES = X_scaled.shape[1]
print(f'Subsampled to {len(X_scaled):,} rows for NN experiments')

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
RESULTS_DIR = pathlib.Path('/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease/results/metrics')
FIGS_DIR = pathlib.Path('/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease/results/figures')
all_results = []
print(f'X shape: {X_scaled.shape}, device: {DEVICE}')


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation='relu', dropout=0.3, batch_norm=True):
        super().__init__()
        act_map = {'relu': nn.ReLU, 'gelu': nn.GELU, 'swish': nn.SiLU, 'tanh': nn.Tanh}
        layers = []
        in_dim = N_FEATURES
        for out_dim in layer_sizes:
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(act_map.get(activation, nn.ReLU)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_eval_mlp(model_fn, arch_name, n_epochs=80, batch_size=2048, lr=1e-3, patience=10):
    fold_aucs, fold_f1s, fold_recalls = [], [], []
    train_losses_all, val_losses_all = [], []

    for fold, (tr_idx, va_idx) in enumerate(CV.split(X_scaled, y_np)):
        X_tr, X_va = X_scaled[tr_idx], X_scaled[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)

        model = model_fn().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_val_loss = float('inf')
        patience_count = 0
        train_losses, val_losses = [], []
        best_weights = None

        X_va_t = torch.tensor(X_va).to(DEVICE)
        y_va_t = torch.tensor(y_va).to(DEVICE)

        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                out = model(Xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_va_t), y_va_t).item()

            train_losses.append(epoch_loss / len(tr_ds))
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

        if best_weights:
            model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            proba = torch.softmax(model(X_va_t), dim=1)[:, 1].cpu().numpy()

        pred = (proba >= 0.5).astype(int)
        fold_aucs.append(roc_auc_score(y_va, proba))
        fold_f1s.append(f1_score(y_va, pred))
        fold_recalls.append(recall_score(y_va, pred))
        train_losses_all.append(train_losses)
        val_losses_all.append(val_losses)
        print(f'    Fold {fold+1}: AUC={fold_aucs[-1]:.4f}  epochs={len(train_losses)}')

    metrics = {
        'roc_auc_mean': float(np.mean(fold_aucs)), 'roc_auc_std': float(np.std(fold_aucs)),
        'f1_mean': float(np.mean(fold_f1s)), 'f1_std': float(np.std(fold_f1s)),
        'recall_mean': float(np.mean(fold_recalls)), 'recall_std': float(np.std(fold_recalls)),
    }
    return metrics, train_losses_all, val_losses_all


# ── MLP Experiments ────────────────────────────────────────────────────────────
architectures = [
    ('MLP Small (64-32, ReLU)',        lambda: MLP([64, 32], 'relu', 0.3),        {'layers': '64-32', 'act': 'relu'}),
    ('MLP Medium (256-128-64, BN)',    lambda: MLP([256, 128, 64], 'relu', 0.3),  {'layers': '256-128-64', 'act': 'relu'}),
    ('MLP Wide (512-256, GELU)',       lambda: MLP([512, 256], 'gelu', 0.4),      {'layers': '512-256', 'act': 'gelu'}),
    ('MLP Deep (5-layer, Swish)',      lambda: MLP([128, 128, 64, 64, 32], 'swish', 0.3), {'layers': '128-128-64-64-32', 'act': 'swish'}),
]

deep_tl, deep_vl = None, None
for name, model_fn, arch in architectures:
    print(f'\nTraining {name}...')
    t0 = time.time()
    metrics, tl, vl = train_eval_mlp(model_fn, name, n_epochs=60, batch_size=2048, lr=1e-3, patience=8)
    elapsed = time.time() - t0
    log_mlflow_run(name, metrics, params=arch, tags={'phase': 'neural_network', 'device': str(DEVICE)})
    all_results.append({'model': name, 'phase': 'neural_network', **metrics, 'elapsed_s': round(elapsed, 1)})
    print(f'  AUC={metrics["roc_auc_mean"]:.4f}±{metrics["roc_auc_std"]:.4f}  '
          f'F1={metrics["f1_mean"]:.4f}  Recall={metrics["recall_mean"]:.4f}  [{elapsed:.0f}s]')
    if 'Deep' in name:
        deep_tl, deep_vl = tl, vl

# ── Training Curves (MLP Deep) ─────────────────────────────────────────────────
if deep_tl:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = sns.color_palette(PALETTE, n_colors=5)
    for fi, (tli, vli) in enumerate(zip(deep_tl, deep_vl)):
        axes[0].plot(tli, color=colors[fi], alpha=0.7, lw=1.5, label=f'Fold {fi+1}')
        axes[1].plot(vli, color=colors[fi], alpha=0.7, lw=1.5, label=f'Fold {fi+1}')
    axes[0].set_title('Train Loss — MLP Deep (5-fold)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy')
    axes[0].legend(fontsize=8)
    axes[1].set_title('Validation Loss — MLP Deep (5-fold)')
    axes[1].set_xlabel('Epoch'); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / '06_mlp_deep_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('\nTraining curves saved.')

# ── TabNet — subsample to 50K for speed ───────────────────────────────────────
print('\nTraining TabNet (50K subsample)...')
idx_tab = np.random.RandomState(42).choice(len(X_scaled), size=50_000, replace=False)
X_tab = X_scaled[idx_tab]
y_tab = y_np[idx_tab]
tabnet_aucs, tabnet_f1s, tabnet_recalls = [], [], []
for fold, (tr_idx, va_idx) in enumerate(CV.split(X_tab, y_tab)):
    X_tr, X_va = X_tab[tr_idx], X_tab[va_idx]
    y_tr, y_va = y_tab[tr_idx], y_tab[va_idx]
    clf = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5, gamma=1.3,
        n_independent=2, n_shared=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={'lr': 2e-3},
        mask_type='entmax', verbose=0, seed=42,
        device_name='mps' if torch.backends.mps.is_available() else 'cpu'
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric=['auc'],
            max_epochs=60, patience=8, batch_size=1024, virtual_batch_size=256)
    proba = clf.predict_proba(X_va)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tabnet_aucs.append(roc_auc_score(y_va, proba))
    tabnet_f1s.append(f1_score(y_va, pred))
    tabnet_recalls.append(recall_score(y_va, pred))
    print(f'  Fold {fold+1}: AUC={tabnet_aucs[-1]:.4f}')

tabnet_metrics = {
    'roc_auc_mean': float(np.mean(tabnet_aucs)), 'roc_auc_std': float(np.std(tabnet_aucs)),
    'f1_mean': float(np.mean(tabnet_f1s)), 'f1_std': float(np.std(tabnet_f1s)),
    'recall_mean': float(np.mean(tabnet_recalls)), 'recall_std': float(np.std(tabnet_recalls)),
}
name = 'TabNet (n_d=32, n_steps=5)'
log_mlflow_run(name, tabnet_metrics, params={'n_d': 32, 'n_steps': 5}, tags={'phase': 'neural_network'})
all_results.append({'model': name, 'phase': 'neural_network', **tabnet_metrics})
print(f'  TabNet AUC={tabnet_metrics["roc_auc_mean"]:.4f}±{tabnet_metrics["roc_auc_std"]:.4f}  '
      f'F1={tabnet_metrics["f1_mean"]:.4f}  Recall={tabnet_metrics["recall_mean"]:.4f}')

# ── Save Results ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
print('\n=== NEURAL NETWORK LEADERBOARD ===')
print(results_df[['model','roc_auc_mean','f1_mean','recall_mean']].sort_values('roc_auc_mean', ascending=False).to_string(index=False))
results_df.to_csv(RESULTS_DIR / '06_neural_network_results.csv', index=False)
print('\nSaved to results/metrics/06_neural_network_results.csv')

# Bar chart
plot_df = results_df.sort_values('roc_auc_mean', ascending=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (mc, sc, title) in zip(axes, [('roc_auc_mean','roc_auc_std','ROC-AUC'),('f1_mean','f1_std','F1'),('recall_mean','recall_std','Recall')]):
    colors = sns.color_palette(PALETTE, n_colors=len(plot_df))
    bars = ax.barh(plot_df['model'], plot_df[mc], xerr=plot_df[sc], color=colors, capsize=3)
    ax.set_xlabel(title); ax.set_title(f'{title} (5-fold CV)')
    ax.set_xlim(max(0, plot_df[mc].min()-0.02), 1.0)
    for bar, val in zip(bars, plot_df[mc]):
        ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)
fig.suptitle('Neural Networks — Cross-Validation Performance', fontsize=13)
fig.tight_layout()
fig.savefig(FIGS_DIR / '06_neural_network_results.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Figures saved.')
