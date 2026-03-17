import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              roc_curve, precision_recall_curve, f1_score,
                              average_precision_score, log_loss, brier_score_loss)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ── Load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip()

TARGET = 'Bankrupt?'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Strip column name spaces
X.columns = X.columns.str.strip()

# Replace inf with NaN then median impute
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

print("Dataset shape:", X.shape)
print("Class dist:", y.value_counts().to_dict())

# ── Splits: Train / Test / OOS ────────────────────────────────────────────────
# 60% train | 20% test | 20% OOS
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y)
X_test, X_oos, y_test, y_oos = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | OOS: {X_oos.shape[0]}")

# ── Model: Gradient Boosting with class_weight via sample_weight ───────────────
# GradientBoostingClassifier chosen: handles imbalance well, good for tabular
# financial data, produces calibrated probabilities, no external libs needed
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
sample_weights = np.where(y_train == 1, scale_pos, 1.0)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
X_oos_sc   = scaler.transform(X_oos)

model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    subsample=0.8, random_state=42, min_samples_leaf=10
)
model.fit(X_train_sc, y_train, sample_weight=sample_weights)

# ── Predictions ───────────────────────────────────────────────────────────────
for split_name, Xs, ys in [('TEST', X_test_sc, y_test), ('OOS', X_oos_sc, y_oos)]:
    prob = model.predict_proba(Xs)[:,1]
    pred = (prob >= 0.35).astype(int)   # lower threshold due to imbalance
    print(f"\n── {split_name} Metrics ──")
    print(f"  ROC-AUC:  {roc_auc_score(ys, prob):.4f}")
    print(f"  F1:       {f1_score(ys, pred):.4f}")
    print(f"  Avg Prec: {average_precision_score(ys, prob):.4f}")
    print(f"  Log Loss: {log_loss(ys, prob):.4f}")
    print(f"  Brier:    {brier_score_loss(ys, prob):.4f}")
    print(classification_report(ys, pred, target_names=['Legit','Bankrupt']))

# Save artefacts for plots
np.save('data/train_probs.npy', model.predict_proba(X_train_sc)[:,1])
np.save('data/test_probs.npy',  model.predict_proba(X_test_sc)[:,1])
np.save('data/oos_probs.npy',   model.predict_proba(X_oos_sc)[:,1])
np.save('data/y_train.npy', y_train.values)
np.save('data/y_test.npy',  y_test.values)
np.save('data/y_oos.npy',   y_oos.values)

# Feature importance
fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fi.to_csv('data/feature_importance.csv')

# Save train/test/oos arrays for PSI
np.save('data/X_train_sc.npy', X_train_sc)
np.save('data/X_test_sc.npy',  X_test_sc)
np.save('data/X_oos_sc.npy',   X_oos_sc)
np.save('data/feature_names.npy', np.array(X.columns.tolist()))

# ROC curve data
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_sc)[:,1])
np.save('data/roc_fpr.npy', fpr)
np.save('data/roc_tpr.npy', tpr)

print("\nAll arrays saved.")
