"""
Phase 10: Dimensionality Reduction & Clustering
- PCA 2D/3D, t-SNE (20K subsample), UMAP
- K-Means (k=2-8), Hierarchical, DBSCAN, GMM
- Cluster evaluation: ARI, NMI, Silhouette
- Cluster profiling + use as features
Results → results/figures/10_* and results/metrics/10_*
"""
import sys, warnings, pathlib, json
sys.path.insert(0, '/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                              silhouette_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier
import umap

from src.data_utils import load_data, get_X_y
from src.visualization import PALETTE

sns.set_theme(style='whitegrid', palette=PALETTE)
FIGS_DIR = pathlib.Path('results/figures')
RESULTS_DIR = pathlib.Path('results/metrics')

# ── Load data ──────────────────────────────────────────────────────────────────
train = load_data('train')
X_raw, y = get_X_y(train, extra_features=False)
feature_names = list(X_raw.columns)
y_np = y.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.values).astype(np.float32)

# Subsample 20K for t-SNE/UMAP (slow), 30K for clustering
rng = np.random.RandomState(42)
idx_20k = rng.choice(len(X_scaled), size=20_000, replace=False)
idx_30k = rng.choice(len(X_scaled), size=30_000, replace=False)
X_20k, y_20k = X_scaled[idx_20k], y_np[idx_20k]
X_30k, y_30k = X_scaled[idx_30k], y_np[idx_30k]
print(f"Subsamples: 20K for dim reduction, 30K for clustering")

# ── PCA ────────────────────────────────────────────────────────────────────────
print('\n── PCA ──')
pca = PCA(n_components=min(13, X_scaled.shape[1]), random_state=42)
pca.fit(X_scaled)
explained_var = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Scree plot
axes[0].bar(range(1, len(explained_var)+1), explained_var, color='steelblue', alpha=0.8)
axes[0].plot(range(1, len(explained_var)+1), np.cumsum(explained_var), 'ro-', lw=2, label='Cumulative')
axes[0].axhline(y=0.95, color='gray', linestyle='--', label='95%')
axes[0].set_xlabel('Principal Component'); axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('PCA Scree Plot'); axes[0].legend()

# PCA 2D scatter
X_pca2 = pca.transform(X_20k)[:, :2]
for cls, label, color in [(0, 'Absence', 'steelblue'), (1, 'Presence', 'coral')]:
    mask = y_20k == cls
    axes[1].scatter(X_pca2[mask, 0], X_pca2[mask, 1], alpha=0.3, s=3, label=label, color=color)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA 2D (20K sample, colored by label)'); axes[1].legend()

# PCA loadings heatmap
loadings = pd.DataFrame(pca.components_[:5].T, index=feature_names,
                        columns=[f'PC{i+1}' for i in range(5)])
sns.heatmap(loadings, cmap='coolwarm', center=0, ax=axes[2], annot=True, fmt='.2f',
            annot_kws={'fontsize': 7})
axes[2].set_title('PCA Loadings (PC1-PC5)')
fig.tight_layout()
fig.savefig(FIGS_DIR / '10_pca_analysis.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Variance by 2 PCs: {sum(explained_var[:2]):.3f}  by 5: {sum(explained_var[:5]):.3f}")

# ── t-SNE ─────────────────────────────────────────────────────────────────────
print('\n── t-SNE (perplexity 30 and 50, 20K sample) ──')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, perp in zip(axes, [30, 50]):
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=500, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_20k)
    for cls, label, color in [(0, 'Absence', 'steelblue'), (1, 'Presence', 'coral')]:
        mask = y_20k == cls
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.2, s=2, label=label, color=color)
    ax.set_title(f't-SNE (perplexity={perp}, 20K)'); ax.legend()
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    print(f"  perplexity={perp} done")
fig.suptitle('t-SNE Visualization — Heart Disease', fontsize=12)
fig.tight_layout()
fig.savefig(FIGS_DIR / '10_tsne.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── UMAP ──────────────────────────────────────────────────────────────────────
print('\n── UMAP (neighbors 15 and 50, 20K sample) ──')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, n_neighbors in zip(axes, [15, 50]):
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
                        random_state=42, n_jobs=-1)
    X_umap = reducer.fit_transform(X_20k)
    for cls, label, color in [(0, 'Absence', 'steelblue'), (1, 'Presence', 'coral')]:
        mask = y_20k == cls
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], alpha=0.2, s=2, label=label, color=color)
    ax.set_title(f'UMAP (n_neighbors={n_neighbors}, 20K)')
    ax.legend(); ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    print(f"  n_neighbors={n_neighbors} done")
fig.suptitle('UMAP Visualization — Heart Disease', fontsize=12)
fig.tight_layout()
fig.savefig(FIGS_DIR / '10_umap.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── K-Means ────────────────────────────────────────────────────────────────────
print('\n── K-Means (k=2 to 8, 30K sample) ──')
inertias, silhouettes, aris, nmis = [], [], [], []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_30k)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_30k, labels, sample_size=5000, random_state=42)
    ari = adjusted_rand_score(y_30k, labels)
    nmi = normalized_mutual_info_score(y_30k, labels)
    silhouettes.append(sil); aris.append(ari); nmis.append(nmi)
    print(f"  k={k}: inertia={km.inertia_:.0f}  silhouette={sil:.3f}  ARI={ari:.3f}  NMI={nmi:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(k_range, inertias, 'bo-'); axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[0].set_title('K-Means Elbow Curve'); axes[0].grid(True, alpha=0.3)
axes[1].plot(k_range, silhouettes, 'go-'); axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('K-Means Silhouette'); axes[1].grid(True, alpha=0.3)
axes[2].plot(k_range, aris, 'ro-', label='ARI')
axes[2].plot(k_range, nmis, 'bs-', label='NMI')
axes[2].set_xlabel('k'); axes[2].set_title('K-Means Alignment with Target')
axes[2].legend(); axes[2].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / '10_kmeans_analysis.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Hierarchical Clustering (k=2) ──────────────────────────────────────────────
print('\n── Hierarchical clustering (k=2, 5K sample) ──')
idx_5k = rng.choice(len(X_30k), size=5_000, replace=False)
X_5k, y_5k = X_30k[idx_5k], y_30k[idx_5k]
hier_results = {}
for linkage in ['ward', 'complete', 'average']:
    ag = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    labels_ag = ag.fit_predict(X_5k)
    ari_ag = adjusted_rand_score(y_5k, labels_ag)
    nmi_ag = normalized_mutual_info_score(y_5k, labels_ag)
    hier_results[linkage] = {'ARI': ari_ag, 'NMI': nmi_ag}
    print(f"  {linkage}: ARI={ari_ag:.3f}  NMI={nmi_ag:.3f}")

# ── DBSCAN ─────────────────────────────────────────────────────────────────────
print('\n── DBSCAN (eps grid, 5K sample) ──')
dbscan_results = []
for eps in [0.3, 0.5, 0.8, 1.0, 1.5]:
    db = DBSCAN(eps=eps, min_samples=10, n_jobs=-1)
    labels_db = db.fit_predict(X_5k)
    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise = (labels_db == -1).sum()
    if n_clusters > 1:
        sil_db = silhouette_score(X_5k[labels_db != -1], labels_db[labels_db != -1],
                                  sample_size=min(3000, (labels_db != -1).sum())) if n_clusters > 1 else np.nan
        ari_db = adjusted_rand_score(y_5k, labels_db)
    else:
        sil_db = np.nan; ari_db = np.nan
    dbscan_results.append({'eps': eps, 'n_clusters': n_clusters, 'n_noise': n_noise,
                           'silhouette': sil_db, 'ARI': ari_db})
    sil_str = f'{sil_db:.3f}' if not np.isnan(sil_db) else 'nan'
    print(f"  eps={eps}: clusters={n_clusters}, noise={n_noise}, sil={sil_str}")

# ── GMM ────────────────────────────────────────────────────────────────────────
print('\n── GMM (n_components 2-6, 30K sample) ──')
gmm_results = []
for n_comp in range(2, 7):
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full',
                          random_state=42, max_iter=100)
    gmm.fit(X_30k)
    labels_gmm = gmm.predict(X_30k)
    ari_gmm = adjusted_rand_score(y_30k, labels_gmm)
    nmi_gmm = normalized_mutual_info_score(y_30k, labels_gmm)
    bic = gmm.bic(X_30k); aic = gmm.aic(X_30k)
    gmm_results.append({'n_components': n_comp, 'BIC': bic, 'AIC': aic,
                        'ARI': ari_gmm, 'NMI': nmi_gmm})
    print(f"  n_comp={n_comp}: BIC={bic:.0f}  ARI={ari_gmm:.3f}  NMI={nmi_gmm:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
gmm_df = pd.DataFrame(gmm_results)
axes[0].plot(gmm_df['n_components'], gmm_df['BIC'], 'bo-', label='BIC')
axes[0].plot(gmm_df['n_components'], gmm_df['AIC'], 'rs-', label='AIC')
axes[0].set_xlabel('n_components'); axes[0].set_title('GMM BIC/AIC'); axes[0].legend()
axes[1].plot(gmm_df['n_components'], gmm_df['ARI'], 'go-', label='ARI')
axes[1].plot(gmm_df['n_components'], gmm_df['NMI'], 'bs-', label='NMI')
axes[1].set_xlabel('n_components'); axes[1].set_title('GMM Alignment with Target')
axes[1].legend(); axes[1].set_ylim(0, 0.5)
fig.tight_layout()
fig.savefig(FIGS_DIR / '10_gmm_analysis.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── Cluster Profiles (K-Means k=2) ────────────────────────────────────────────
print('\n── Cluster profiling (k=2) ──')
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
km2_labels = km2.fit_predict(X_30k)
cluster_df = pd.DataFrame(X_30k, columns=feature_names)
cluster_df['cluster'] = km2_labels
cluster_df['target'] = y_30k

profile = cluster_df.groupby('cluster').agg(['mean']).round(3)
print(f"\nCluster profiles (mean values):")
print(profile.to_string())
print(f"\nCluster target alignment:")
for c in [0, 1]:
    pos_rate = cluster_df[cluster_df['cluster']==c]['target'].mean()
    print(f"  Cluster {c}: {(km2_labels==c).sum():,} samples, {pos_rate:.3f} positive rate")
    
km2_ari = adjusted_rand_score(y_30k, km2_labels)
print(f"K-Means k=2 ARI with true labels: {km2_ari:.3f}")

# ── Clusters as Features ───────────────────────────────────────────────────────
print('\n── Clusters as features (LightGBM) ──')
with open(RESULTS_DIR / '07_automl_best_params.json') as f:
    ap = json.load(f)
lgb_params = {**ap['lgb_tpe_200_best'], 'objective': 'binary', 'verbosity': -1, 'n_jobs': -1, 'random_state': 42}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Baseline (no cluster features)
scores_base = cross_val_score(LGBMClassifier(**lgb_params), X_30k, y_30k, cv=cv5, scoring='roc_auc', n_jobs=-1)
# With cluster assignments as extra feature (k=2,3,4)
best_cluster_auc = float(scores_base.mean())
print(f"  Baseline: {scores_base.mean():.4f}±{scores_base.std():.4f}")
cluster_feature_results = [{'feature_set': 'baseline', 'auc_mean': float(scores_base.mean()), 'auc_std': float(scores_base.std())}]
for k_c in [2, 3, 4]:
    km_c = KMeans(n_clusters=k_c, random_state=42, n_init=10)
    cl_feat = km_c.fit_predict(X_30k).reshape(-1, 1).astype(np.float32)
    X_with_clusters = np.hstack([X_30k, cl_feat])
    scores_c = cross_val_score(LGBMClassifier(**lgb_params), X_with_clusters, y_30k, cv=cv5, scoring='roc_auc', n_jobs=-1)
    print(f"  +KMeans(k={k_c}): {scores_c.mean():.4f}±{scores_c.std():.4f}")
    cluster_feature_results.append({'feature_set': f'+KMeans_k{k_c}', 'auc_mean': float(scores_c.mean()), 'auc_std': float(scores_c.std())})

# ── Save All Results ───────────────────────────────────────────────────────────
clustering_summary = {
    'kmeans': [{'k': k, 'silhouette': s, 'ARI': a, 'NMI': n}
               for k, s, a, n in zip(k_range, silhouettes, aris, nmis)],
    'hierarchical': hier_results,
    'dbscan': dbscan_results,
    'gmm': gmm_results,
    'cluster_features': cluster_feature_results,
    'pca_explained_var_2pc': float(sum(explained_var[:2])),
    'pca_explained_var_5pc': float(sum(explained_var[:5])),
}
with open(RESULTS_DIR / '10_clustering_summary.json', 'w') as f:
    json.dump(clustering_summary, f, indent=2, default=float)

print('\n=== Phase 10 Clustering Complete ===')
print(f"Best K-Means ARI (k=2): {aris[0]:.3f}")
print(f"PCA 2D explains {sum(explained_var[:2])*100:.1f}% variance")
