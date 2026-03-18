import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, log_loss,
    average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = 'data.csv'
OUTPUT_DIR  = 'model_outputs'
TARGET      = 'Bankrupt?'
THRESHOLD   = 0.35
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load & Clean ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
X = df.drop(columns=[TARGET])
y = df[TARGET]
X.columns = X.columns.str.strip()
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y)
X_test, X_oos, y_test, y_oos = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# ── Train ─────────────────────────────────────────────────────────────────────
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
sample_weights = np.where(y_train == 1, scale_pos, 1.0)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
X_oos_sc   = scaler.transform(X_oos)

model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05,
    max_depth=4, subsample=0.8,
    random_state=42, min_samples_leaf=10
)
model.fit(X_train_sc, y_train, sample_weight=sample_weights)

# ── Predict ───────────────────────────────────────────────────────────────────
probs = model.predict_proba(X_test_sc)[:, 1]
preds = (probs >= THRESHOLD).astype(int)

# ── 1. metrics.json ───────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, preds)
metrics = {
    "roc_auc":          round(roc_auc_score(y_test, probs), 4),
    "f1":               round(f1_score(y_test, preds), 4),
    "log_loss":         round(log_loss(y_test, probs), 4),
    "avg_precision":    round(average_precision_score(y_test, probs), 4),
    "brier":            round(brier_score_loss(y_test, probs), 4),
    "threshold":        THRESHOLD,
    "split":            "test"
}
with open(f'{OUTPUT_DIR}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ── 2. roc_curve.json ─────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, probs)
# Downsample to 200 points so the JSON stays small
idx = np.linspace(0, len(fpr)-1, 200, dtype=int)
roc_data = {
    "fpr": fpr[idx].tolist(),
    "tpr": tpr[idx].tolist()
}
with open(f'{OUTPUT_DIR}/roc_curve.json', 'w') as f:
    json.dump(roc_data, f)

# ── 3. pr_curve.json ──────────────────────────────────────────────────────────
precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
idx2 = np.linspace(0, len(precision_vals)-1, 200, dtype=int)
pr_data = {
    "precision": precision_vals[idx2].tolist(),
    "recall":    recall_vals[idx2].tolist()
}
with open(f'{OUTPUT_DIR}/pr_curve.json', 'w') as f:
    json.dump(pr_data, f)

# ── 4. score_dist.json ────────────────────────────────────────────────────────
BINS = 25
bin_edges = np.linspace(0, 1, BINS + 1)
legit_hist,  _ = np.histogram(probs[y_test == 0], bins=bin_edges)
bankr_hist,  _ = np.histogram(probs[y_test == 1], bins=bin_edges)
bin_labels     = [f"{bin_edges[i]:.2f}" for i in range(BINS)]
dist_data = {
    "bins":      bin_labels,
    "legitimate": legit_hist.tolist(),
    "bankrupt":   bankr_hist.tolist()
}
with open(f'{OUTPUT_DIR}/score_dist.json', 'w') as f:
    json.dump(dist_data, f)

# ── 5. confusion.json ─────────────────────────────────────────────────────────
tn, fp, fn, tp = cm.ravel()
confusion = { "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn) }
with open(f'{OUTPUT_DIR}/confusion.json', 'w') as f:
    json.dump(confusion, f, indent=2)

# ── 6. feature_importance.json ────────────────────────────────────────────────
fi = pd.Series(model.feature_importances_, index=X.columns)
top15 = fi.nlargest(15)
feat_data = {
    "features":    top15.index.tolist(),
    "importances": [round(v, 5) for v in top15.values.tolist()]
}
with open(f'{OUTPUT_DIR}/feature_importance.json', 'w') as f:
    json.dump(feat_data, f, indent=2)

# ── 7. summary.json ───────────────────────────────────────────────────────────
n_bankrupt = int(y.sum())
n_legit    = int((y == 0).sum())
summary = {
    "total_records":    len(df),
    "n_features":       X.shape[1],
    "n_bankrupt":       n_bankrupt,
    "n_legit":          n_legit,
    "bankrupt_pct":     round(n_bankrupt / len(df) * 100, 2),
    "imbalance_ratio":  round(n_legit / n_bankrupt, 1),
    "train_size":       X_train.shape[0],
    "test_size":        X_test.shape[0],
    "oos_size":         X_oos.shape[0],
    "model":            "GradientBoostingClassifier",
    "model_version":    "v2.1.0",
    "threshold":        THRESHOLD
}
with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("All JSON outputs saved to model_outputs/")
print(f"  ROC-AUC: {metrics['roc_auc']} | F1: {metrics['f1']} | Log Loss: {metrics['log_loss']}")