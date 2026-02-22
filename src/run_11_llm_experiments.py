"""
Phase 11: LLM Experiments via Ollama
Model: qwen3:0.6b (originally planned: hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_S)
Note: 20B model was not available in this session; qwen3:0.6b used as substitute
Experiments:
1. Zero-shot classification (200 samples)
2. Few-shot (5-shot) classification
3. Chain-of-Thought (CoT) prompting
4. LLM as feature extractor (risk score 0-10)
5. Edge case analysis (hardest samples)
"""
import sys, warnings, pathlib, json, time, re
sys.path.insert(0, '/Users/dustinober/Projects/Kaggle-Predicting-Heart-Disease')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                              confusion_matrix, ConfusionMatrixDisplay, accuracy_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from src.data_utils import load_data, get_X_y
from src.visualization import PALETTE

sns.set_theme(style='whitegrid', palette=PALETTE)
FIGS_DIR = pathlib.Path('results/figures')
RESULTS_DIR = pathlib.Path('results/metrics')

OLLAMA_URL = 'http://localhost:11434/api/generate'
LLM_MODEL = 'qwen3:0.6b'
N_SAMPLES = 200  # limit for speed; 0.6b generates ~3-5 tokens/s

# ── Feature descriptions for natural language prompts ─────────────────────────
FEATURE_DESCRIPTIONS = {
    'Age': ('Age', 'years'),
    'Sex': ('Sex', '1=male, 0=female'),
    'Chest pain type': ('Chest pain type', '1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic'),
    'BP': ('Resting blood pressure', 'mmHg'),
    'Cholesterol': ('Serum cholesterol', 'mg/dl'),
    'FBS over 120': ('Fasting blood sugar > 120 mg/dl', '1=yes, 0=no'),
    'EKG results': ('Resting EKG results', '0=normal, 1=ST-T abnormality, 2=LV hypertrophy'),
    'Max HR': ('Maximum heart rate achieved', 'bpm'),
    'Exercise angina': ('Exercise-induced angina', '1=yes, 0=no'),
    'ST depression': ('ST depression induced by exercise', 'mm'),
    'Slope of ST': ('Slope of peak exercise ST segment', '1=upsloping, 2=flat, 3=downsloping'),
    'Number of vessels fluro': ('Number of major vessels colored by fluoroscopy', '0-3'),
    'Thallium': ('Thallium stress test result', '3=normal, 6=fixed defect, 7=reversible defect'),
}

def row_to_natural_language(row):
    """Convert a data row to a natural language description."""
    parts = []
    for col, (desc, unit) in FEATURE_DESCRIPTIONS.items():
        val = row[col]
        parts.append(f"{desc}: {val} ({unit})")
    return ". ".join(parts) + "."

def query_llm(prompt, max_tokens=20, timeout=30):
    """Query Ollama and return the response text."""
    payload = {
        'model': LLM_MODEL,
        'prompt': prompt,
        'stream': False,
        'options': {'num_predict': max_tokens, 'temperature': 0.0, 'seed': 42}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get('response', '').strip()
    except Exception as e:
        return f"ERROR: {e}"

def parse_prediction(response, default=0):
    """Parse LLM response to binary prediction."""
    r = response.lower()
    if any(w in r for w in ['presence', 'yes', 'positive', 'disease', 'has heart', 'likely has', '1']):
        return 1
    if any(w in r for w in ['absence', 'no', 'negative', 'healthy', "doesn't", 'unlikely', '0']):
        return 0
    return default

def parse_risk_score(response):
    """Extract numeric risk score 0-10 from LLM response."""
    matches = re.findall(r'\b([0-9]|10)\b', response)
    if matches:
        return float(matches[0])
    return 5.0  # default

# ── Load data ──────────────────────────────────────────────────────────────────
train = load_data('train')
X_raw, y = get_X_y(train, extra_features=False)
feature_names = list(X_raw.columns)

# Sample N_SAMPLES rows (stratified)
rng = np.random.RandomState(42)
idx_0 = np.where(y.values == 0)[0]; idx_1 = np.where(y.values == 1)[0]
n_half = N_SAMPLES // 2
idx_sample = np.concatenate([
    rng.choice(idx_0, size=n_half, replace=False),
    rng.choice(idx_1, size=n_half, replace=False)
])
rng.shuffle(idx_sample)
X_llm = X_raw.iloc[idx_sample].reset_index(drop=True)
y_llm = y.iloc[idx_sample].reset_index(drop=True).values
print(f"LLM experiment sample: {len(X_llm)} rows ({y_llm.sum()} positive)")

# Build few-shot examples (first 10 of training data, not in sample)
few_shot_pool = X_raw.drop(index=idx_sample).head(20)
few_shot_y = y.drop(index=idx_sample).head(20).values
few_shot_examples = []
for i in range(len(few_shot_pool)):
    nl = row_to_natural_language(few_shot_pool.iloc[i])
    label = "Presence" if few_shot_y[i] == 1 else "Absence"
    few_shot_examples.append(f"Patient: {nl}\nAnswer: {label}")

# ── Experiment 1: Zero-Shot ────────────────────────────────────────────────────
print(f'\n── Experiment 1: Zero-Shot Classification ({N_SAMPLES} samples) ──')
ZS_PROMPT_TEMPLATE = """You are a medical AI assistant. Based on the following patient data, predict whether the patient has heart disease.
Answer with ONLY "Presence" or "Absence".

Patient data: {patient_data}

Answer:"""

zs_preds, zs_responses = [], []
t0 = time.time()
for i in range(len(X_llm)):
    nl = row_to_natural_language(X_llm.iloc[i])
    prompt = ZS_PROMPT_TEMPLATE.format(patient_data=nl)
    response = query_llm(prompt, max_tokens=5)
    pred = parse_prediction(response)
    zs_preds.append(pred); zs_responses.append(response)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(X_llm)} done  [{time.time()-t0:.0f}s]")

zs_preds = np.array(zs_preds)
zs_acc = accuracy_score(y_llm, zs_preds)
zs_f1 = f1_score(y_llm, zs_preds)
zs_recall = recall_score(y_llm, zs_preds)
print(f"  Zero-Shot: Accuracy={zs_acc:.3f}  F1={zs_f1:.3f}  Recall={zs_recall:.3f}")

# ── Experiment 2: Few-Shot (5-shot) ───────────────────────────────────────────
print(f'\n── Experiment 2: Few-Shot (5-shot, {N_SAMPLES} samples) ──')
few_shot_header = "Examples:\n" + "\n\n".join(few_shot_examples[:5])
FS_PROMPT_TEMPLATE = """{examples}

Now predict for this patient. Answer with ONLY "Presence" or "Absence".

Patient: {patient_data}

Answer:"""

fs_preds = []
t0 = time.time()
for i in range(len(X_llm)):
    nl = row_to_natural_language(X_llm.iloc[i])
    prompt = FS_PROMPT_TEMPLATE.format(examples=few_shot_header, patient_data=nl)
    response = query_llm(prompt, max_tokens=5)
    pred = parse_prediction(response)
    fs_preds.append(pred)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(X_llm)} done  [{time.time()-t0:.0f}s]")

fs_preds = np.array(fs_preds)
fs_acc = accuracy_score(y_llm, fs_preds)
fs_f1 = f1_score(y_llm, fs_preds)
fs_recall = recall_score(y_llm, fs_preds)
print(f"  Few-Shot (5): Accuracy={fs_acc:.3f}  F1={fs_f1:.3f}  Recall={fs_recall:.3f}")

# ── Experiment 3: Chain-of-Thought ────────────────────────────────────────────
print(f'\n── Experiment 3: Chain-of-Thought (50 samples) ──')
COT_PROMPT_TEMPLATE = """You are a medical AI. Analyze this patient's heart disease risk step by step.

Patient data: {patient_data}

Think step by step about the clinical risk factors, then conclude with ONLY "Presence" or "Absence" on the last line."""

cot_preds = []
cot_sample_size = 50  # CoT is slower due to longer responses
for i in range(cot_sample_size):
    nl = row_to_natural_language(X_llm.iloc[i])
    prompt = COT_PROMPT_TEMPLATE.format(patient_data=nl)
    response = query_llm(prompt, max_tokens=80)
    # Extract last line for final answer
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    final_answer = lines[-1] if lines else response
    pred = parse_prediction(final_answer)
    cot_preds.append(pred)

cot_preds = np.array(cot_preds)
cot_y = y_llm[:cot_sample_size]
cot_acc = accuracy_score(cot_y, cot_preds)
cot_f1 = f1_score(cot_y, cot_preds, zero_division=0)
cot_recall = recall_score(cot_y, cot_preds, zero_division=0)
print(f"  CoT (50 samples): Accuracy={cot_acc:.3f}  F1={cot_f1:.3f}  Recall={cot_recall:.3f}")

# ── Experiment 4: LLM Risk Score as Feature ───────────────────────────────────
print(f'\n── Experiment 4: LLM Risk Score as Feature (100 samples) ──')
RISK_PROMPT = """Rate the heart disease risk of this patient on a scale of 0-10 (0=no risk, 10=highest risk).
Respond with ONLY a single integer between 0 and 10.

Patient: {patient_data}

Risk score:"""

risk_scores = []
risk_sample_size = 100
for i in range(risk_sample_size):
    nl = row_to_natural_language(X_llm.iloc[i])
    prompt = RISK_PROMPT.format(patient_data=nl)
    response = query_llm(prompt, max_tokens=3)
    score = parse_risk_score(response)
    risk_scores.append(score)

risk_scores = np.array(risk_scores)
risk_y = y_llm[:risk_sample_size]
risk_correlation = np.corrcoef(risk_scores, risk_y)[0, 1]
print(f"  Risk score correlation with target: r={risk_correlation:.3f}")

# Test if LLM risk score improves ML performance
scaler = StandardScaler()
X_risk_base = scaler.fit_transform(X_llm.values[:risk_sample_size]).astype('float32')
X_risk_plus = np.hstack([X_risk_base, risk_scores.reshape(-1, 1)])
with open(RESULTS_DIR / '07_automl_best_params.json') as f:
    lgb_params = {**json.load(f)['lgb_tpe_200_best'], 'objective': 'binary', 'verbosity': -1, 'n_jobs': -1, 'random_state': 42}
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores_base_risk = cross_val_score(LGBMClassifier(**lgb_params), X_risk_base, risk_y, cv=cv3, scoring='roc_auc')
scores_plus_risk = cross_val_score(LGBMClassifier(**lgb_params), X_risk_plus, risk_y, cv=cv3, scoring='roc_auc')
print(f"  ML without LLM score: {scores_base_risk.mean():.4f}")
print(f"  ML + LLM risk score: {scores_plus_risk.mean():.4f}")

# ── Experiment 5: Edge Case Analysis ─────────────────────────────────────────
print(f'\n── Experiment 5: Edge Cases (top 10 uncertain samples) ──')
# Use a pre-trained LightGBM to find uncertain predictions
scaler_full = StandardScaler()
X_llm_scaled = scaler_full.fit_transform(X_llm.values).astype('float32')
lgb_edge = LGBMClassifier(**lgb_params)
lgb_edge.fit(X_llm_scaled, y_llm)
proba_edge = lgb_edge.predict_proba(X_llm_scaled)[:, 1]
# Most uncertain = closest to 0.5
uncertainty = np.abs(proba_edge - 0.5)
edge_indices = np.argsort(uncertainty)[:10]

EDGE_PROMPT = """This is a challenging medical case. Analyze all clinical factors and predict heart disease.
Reply with "Presence" or "Absence" followed by a one-sentence explanation.

Patient: {patient_data}

Prediction:"""

edge_preds, edge_responses = [], []
for idx in edge_indices:
    nl = row_to_natural_language(X_llm.iloc[idx])
    prompt = EDGE_PROMPT.format(patient_data=nl)
    response = query_llm(prompt, max_tokens=40)
    pred = parse_prediction(response)
    edge_preds.append(pred)
    edge_responses.append({'true_label': int(y_llm[idx]), 'ml_prob': float(proba_edge[idx]),
                           'llm_pred': pred, 'llm_response': response[:100]})

edge_preds = np.array(edge_preds)
edge_y = y_llm[edge_indices]
edge_acc = accuracy_score(edge_y, edge_preds)
print(f"  Edge cases accuracy: {edge_acc:.3f} (ML model is ~50% certain on these)")

# ── Save All Results ───────────────────────────────────────────────────────────
llm_results = {
    'model': LLM_MODEL,
    'n_samples': N_SAMPLES,
    'zero_shot': {'accuracy': float(zs_acc), 'f1': float(zs_f1), 'recall': float(zs_recall)},
    'few_shot_5': {'accuracy': float(fs_acc), 'f1': float(fs_f1), 'recall': float(fs_recall)},
    'chain_of_thought': {'n_samples': cot_sample_size, 'accuracy': float(cot_acc),
                         'f1': float(cot_f1), 'recall': float(cot_recall)},
    'risk_score_feature': {
        'correlation_with_target': float(risk_correlation),
        'ml_auc_without_llm': float(scores_base_risk.mean()),
        'ml_auc_with_llm_score': float(scores_plus_risk.mean()),
    },
    'edge_cases': {'n_cases': 10, 'accuracy': float(edge_acc), 'examples': edge_responses[:3]}
}
with open(RESULTS_DIR / '11_llm_results.json', 'w') as f:
    json.dump(llm_results, f, indent=2)

# ── Comparison bar chart ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
methods = ['Zero-Shot', 'Few-Shot (5)', 'CoT (50 samples)']
accs = [zs_acc, fs_acc, cot_acc]
f1s = [zs_f1, fs_f1, cot_f1]
colors = sns.color_palette(PALETTE, n_colors=3)
axes[0].bar(methods, accs, color=colors); axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1); axes[0].set_title('LLM Accuracy by Prompting Strategy')
axes[0].axhline(y=0.5, color='gray', linestyle='--', label='Random baseline')
axes[0].axhline(y=max(y_llm.mean(), 1-y_llm.mean()), color='orange', linestyle='--', label='Majority baseline')
axes[0].legend(fontsize=8)
for bar, val in zip(axes[0].patches, accs):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}', ha='center', fontsize=9)

axes[1].bar(methods, f1s, color=colors); axes[1].set_ylabel('F1 Score')
axes[1].set_ylim(0, 1); axes[1].set_title('LLM F1 Score by Prompting Strategy')
for bar, val in zip(axes[1].patches, f1s):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}', ha='center', fontsize=9)

fig.suptitle(f'LLM Classification Results ({LLM_MODEL})', fontsize=12)
fig.tight_layout()
fig.savefig(FIGS_DIR / '11_llm_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print('\n=== Phase 11 LLM Experiments Complete ===')
print(json.dumps(llm_results, indent=2))
