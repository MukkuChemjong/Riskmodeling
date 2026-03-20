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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

import joblib
import os

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/bankruptcy_model_v2.1.0.joblib')
joblib.dump(scaler, 'models/scaler_v2.1.0.joblib')  # save the scaler too
print("Model saved to models/bankruptcy_model_v2.1.0.joblib")

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
    
def compute_psi(expected, actual, buckets=10):
    """
    Compute PSI between two arrays.
    
    expected : array from the reference dataset (usually train)
    actual   : array from the comparison dataset (test or OOS)
    buckets  : number of bins to split the distribution into
    
    Returns  : psi_value (float), breakdown (dict with bin details)
    """
    
    # Create bin edges based on the expected (train) distribution
    # Using quantile-based bins so each bin has roughly equal train samples
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, buckets + 1))
    
    # Remove duplicate breakpoints (can happen with sparse data)
    breakpoints = np.unique(breakpoints)
    
    def get_bin_percents(arr, breaks):
        counts, _ = np.histogram(arr, bins=breaks)
        percents   = counts / len(arr)
        # Replace zeros with small value to avoid log(0) = -inf
        percents   = np.where(percents == 0, 1e-6, percents)
        return percents
    
    expected_pct = get_bin_percents(expected, breakpoints)
    actual_pct   = get_bin_percents(actual,   breakpoints)
    
    # PSI formula
    psi_bins = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_value = np.sum(psi_bins)
    
    breakdown = {
        'expected_pct': expected_pct.tolist(),
        'actual_pct':   actual_pct.tolist(),
        'psi_per_bin':  psi_bins.tolist(),
        'total_psi':    round(float(psi_value), 6)
    }
    
    return float(psi_value), breakdown


# ── COMPUTE PSI FOR ALL FEATURES ──────────────────────────────────────────────
def run_psi_analysis(X_train, X_test, X_oos, feature_names, output_dir):
    """
    Run PSI for every feature across train vs test and train vs OOS.
    Saves results to JSON and generates a plot.
    """
    
    psi_results = {
        'train_vs_test': {},
        'train_vs_oos':  {}
    }
    
    for i, feature in enumerate(feature_names):
        train_col = X_train[:, i]
        test_col  = X_test[:, i]
        oos_col   = X_oos[:, i]
        
        psi_test, _ = compute_psi(train_col, test_col)
        psi_oos,  _ = compute_psi(train_col, oos_col)
        
        psi_results['train_vs_test'][feature] = round(psi_test, 6)
        psi_results['train_vs_oos'][feature]  = round(psi_oos,  6)
    
    # Save full results to JSON
    with open(f'{output_dir}/psi_results.json', 'w') as f:
        json.dump(psi_results, f, indent=2)
    
    # ── Flag features by severity ──────────────────────────────────────────
    def categorize(psi_val):
        if psi_val < 0.10:  return 'stable',  '#39ff82'
        if psi_val < 0.20:  return 'monitor', '#ffcc00'
        return 'unstable', '#ff4757'
    
    # ── Build summary DataFrame ────────────────────────────────────────────
    summary = []
    for feat in feature_names:
        psi_t = psi_results['train_vs_test'][feat]
        psi_o = psi_results['train_vs_oos'][feat]
        status_t, _ = categorize(psi_t)
        status_o, _ = categorize(psi_o)
        summary.append({
            'feature':       feat,
            'psi_test':      psi_t,
            'psi_oos':       psi_o,
            'status_test':   status_t,
            'status_oos':    status_o
        })
    
    summary_df = pd.DataFrame(summary).sort_values('psi_test', ascending=False)
    
    # Print top 20 features with highest drift
    print("\n── PSI Results (Top 20 by Test Drift) ──")
    print(f"{'Feature':<45} {'PSI Test':>10} {'Status':>10} {'PSI OOS':>10} {'Status':>10}")
    print("─" * 90)
    for _, row in summary_df.head(20).iterrows():
        print(
            f"{row['feature']:<45} "
            f"{row['psi_test']:>10.4f} "
            f"{row['status_test']:>10} "
            f"{row['psi_oos']:>10.4f} "
            f"{row['status_oos']:>10}"
        )
    
    unstable_test = summary_df[summary_df['status_test'] == 'unstable']
    monitor_test  = summary_df[summary_df['status_test'] == 'monitor']
    print(f"\nUnstable features (PSI > 0.20): {len(unstable_test)}")
    print(f"Monitor features  (PSI 0.10-0.20): {len(monitor_test)}")
    print(f"Stable features   (PSI < 0.10): {len(summary_df) - len(unstable_test) - len(monitor_test)}")
    
    # ── Plot top 20 features ───────────────────────────────────────────────
    plot_psi(summary_df.head(20), output_dir)
    
    return summary_df


def plot_psi(summary_df, output_dir):
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0F1923')
    
    def style_ax(ax, title):
        ax.set_facecolor('#1A2535')
        ax.tick_params(colors='#CBD5E0', labelsize=8)
        ax.xaxis.label.set_color('#CBD5E0')
        ax.yaxis.label.set_color('#CBD5E0')
        for sp in ax.spines.values():
            sp.set_edgecolor('#2D3748')
        ax.set_title(title, color='white', fontsize=11,
                     fontweight='bold', pad=10)
        ax.grid(color='#2D3748', linestyle='--',
                linewidth=0.5, alpha=0.6, axis='x')
    
    def color_bars(values):
        return ['#ff4757' if v >= 0.20
                else '#ffcc00' if v >= 0.10
                else '#39ff82'
                for v in values]
    
    features_short = [
        f[:35] + '…' if len(f) > 35 else f
        for f in summary_df['feature']
    ]
    
    # Left — Train vs Test
    ax = axes[0]
    style_ax(ax, 'PSI: Train vs Test (Top 20 Features)')
    colors = color_bars(summary_df['psi_test'])
    bars = ax.barh(features_short[::-1],
                   summary_df['psi_test'].values[::-1],
                   color=colors[::-1], alpha=0.88, height=0.65)
    ax.axvline(0.10, color='#ffcc00', lw=1.5,
               linestyle='--', label='Monitor (0.10)')
    ax.axvline(0.20, color='#ff4757', lw=1.5,
               linestyle='--', label='Unstable (0.20)')
    ax.set_xlabel('PSI Value')
    ax.legend(fontsize=8, facecolor='#1A2535',
              labelcolor='white', edgecolor='#2D3748')
    for bar, val in zip(bars, summary_df['psi_test'].values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center',
                color='#CBD5E0', fontsize=7)
    
    # Right — Train vs OOS
    ax = axes[1]
    style_ax(ax, 'PSI: Train vs OOS (Top 20 Features)')
    colors = color_bars(summary_df['psi_oos'])
    bars = ax.barh(features_short[::-1],
                   summary_df['psi_oos'].values[::-1],
                   color=colors[::-1], alpha=0.88, height=0.65)
    ax.axvline(0.10, color='#ffcc00', lw=1.5,
               linestyle='--', label='Monitor (0.10)')
    ax.axvline(0.20, color='#ff4757', lw=1.5,
               linestyle='--', label='Unstable (0.20)')
    ax.set_xlabel('PSI Value')
    ax.legend(fontsize=8, facecolor='#1A2535',
              labelcolor='white', edgecolor='#2D3748')
    for bar, val in zip(bars, summary_df['psi_oos'].values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center',
                color='#CBD5E0', fontsize=7)
    
    fig.suptitle('Population Stability Index — Feature Drift Analysis',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/psi_plot.png',
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"PSI plot saved to {output_dir}/psi_plot.png")

import shap

# ── SHAP ANALYSIS ─────────────────────────────────────────────────────────────
def run_shap_analysis(model, X_train_sc, X_test_sc,
                      feature_names, output_dir):
    """
    Compute SHAP values and generate beeswarm + summary plots.
    
    model        : your fitted GradientBoostingClassifier
    X_train_sc   : scaled training data (used to build the explainer)
    X_test_sc    : scaled test data (what we explain)
    feature_names: list of column names
    output_dir   : where to save plots and JSON
    """
    
    print("\nComputing SHAP values — this may take 1-2 minutes...")
    
    # TreeExplainer is optimised for tree-based models like GradientBoosting
    # It computes exact SHAP values, not approximations
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values for the test set
    # Use a sample of 500 rows if the dataset is large — speeds things up
    sample_size = min(500, X_test_sc.shape[0])
    np.random.seed(42)
    sample_idx  = np.random.choice(X_test_sc.shape[0],
                                   size=sample_size, replace=False)
    X_sample    = X_test_sc[sample_idx]
    
    # shap_values shape: (n_samples, n_features, n_classes)
    # For binary classification we take index [1] = probability of bankruptcy
    shap_values = explainer.shap_values(X_sample)
    
    # For GradientBoostingClassifier shap_values is a single 2D array
    # For RandomForest it returns a list — handle both:
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]   # class 1 = Bankrupt
    else:
        shap_vals = shap_values
    
    # ── Save mean absolute SHAP values to JSON ─────────────────────────────
    mean_shap = np.abs(shap_vals).mean(axis=0)
    shap_importance = dict(zip(
        feature_names,
        [round(float(v), 6) for v in mean_shap]
    ))
    shap_importance_sorted = dict(
        sorted(shap_importance.items(),
               key=lambda x: x[1], reverse=True)
    )
    
    with open(f'{output_dir}/shap_importance.json', 'w') as f:
        json.dump(shap_importance_sorted, f, indent=2)
    
    print("Top 10 features by mean |SHAP|:")
    for feat, val in list(shap_importance_sorted.items())[:10]:
        print(f"  {feat:<45} {val:.6f}")
    
    # ── Plot 1: Beeswarm ───────────────────────────────────────────────────
    plot_shap_beeswarm(shap_vals, X_sample, feature_names, output_dir)
    
    # ── Plot 2: Bar summary ────────────────────────────────────────────────
    plot_shap_bar(shap_importance_sorted, output_dir)
    
    # ── Plot 3: Dependence plot for top feature ────────────────────────────
    top_feature     = list(shap_importance_sorted.keys())[0]
    top_feature_idx = list(feature_names).index(top_feature)
    plot_shap_dependence(shap_vals, X_sample, feature_names,
                         top_feature_idx, top_feature, output_dir)
    
    return shap_vals, shap_importance_sorted


def plot_shap_beeswarm(shap_vals, X_sample, feature_names, output_dir):
    """
    Standard SHAP beeswarm using the shap library's built-in plot.
    Shows top 20 features.
    """
    
    # Create an Explanation object — required for newer shap plots
    explanation = shap.Explanation(
        values          = shap_vals,
        data            = X_sample,
        feature_names   = list(feature_names)
    )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#0F1923')
    ax.set_facecolor('#1A2535')
    
    plt.sca(ax)
    shap.plots.beeswarm(
        explanation,
        max_display=20,        # show top 20 features
        show=False,            # don't open a window — we save manually
        color_bar=True,
        plot_size=None         # use our figure size
    )
    
    ax.set_facecolor('#1A2535')
    fig.patch.set_facecolor('#0F1923')
    ax.tick_params(colors='#CBD5E0')
    ax.xaxis.label.set_color('#CBD5E0')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2D3748')
    ax.set_title('SHAP Beeswarm — Feature Impact on Bankruptcy Risk\n'
                 '(Red = High Feature Value, Blue = Low Feature Value)',
                 color='white', fontsize=12, fontweight='bold', pad=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_beeswarm.png',
                dpi=150, bbox_inches='tight',
                facecolor='#0F1923')
    plt.close()
    print(f"Beeswarm plot saved to {output_dir}/shap_beeswarm.png")


def plot_shap_bar(shap_importance_sorted, output_dir):
    """
    Horizontal bar chart of mean absolute SHAP values — top 15 features.
    """
    
    top15_items = list(shap_importance_sorted.items())[:15]
    features    = [f[:35] + '…' if len(f) > 35 else f
                   for f, _ in top15_items]
    values      = [v for _, v in top15_items]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0F1923')
    ax.set_facecolor('#1A2535')
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(features)))
    bars   = ax.barh(features[::-1], values[::-1],
                     color=colors, alpha=0.9, height=0.65)
    
    ax.set_xlabel('Mean |SHAP Value| — Average Impact on Prediction',
                  color='#CBD5E0')
    ax.tick_params(colors='#CBD5E0', labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2D3748')
    ax.grid(color='#2D3748', linestyle='--',
            linewidth=0.5, alpha=0.6, axis='x')
    
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.5f}', va='center',
                color='#CBD5E0', fontsize=8)
    
    ax.set_title('SHAP Feature Importance — Top 15 Features\n'
                 'Mean Absolute SHAP Value across Test Set',
                 color='white', fontsize=12, fontweight='bold', pad=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar.png',
                dpi=150, bbox_inches='tight',
                facecolor='#0F1923')
    plt.close()
    print(f"SHAP bar chart saved to {output_dir}/shap_bar.png")


def plot_shap_dependence(shap_vals, X_sample, feature_names,
                         feature_idx, feature_name, output_dir):
    """
    Dependence plot for the single most important feature.
    Shows how its SHAP value changes across its value range.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0F1923')
    ax.set_facecolor('#1A2535')
    
    x_vals    = X_sample[:, feature_idx]
    shap_col  = shap_vals[:, feature_idx]
    
    sc = ax.scatter(x_vals, shap_col,
                    c=x_vals, cmap='RdBu_r',
                    s=15, alpha=0.7, vmin=np.percentile(x_vals, 5),
                    vmax=np.percentile(x_vals, 95))
    
    ax.axhline(0, color='#718096', lw=1.2, linestyle='--', alpha=0.7)
    ax.set_xlabel(f'{feature_name} (scaled)',  color='#CBD5E0')
    ax.set_ylabel('SHAP Value',                color='#CBD5E0')
    ax.tick_params(colors='#CBD5E0')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2D3748')
    ax.grid(color='#2D3748', linestyle='--', linewidth=0.5, alpha=0.5)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Feature Value', color='#CBD5E0')
    cbar.ax.yaxis.set_tick_params(color='#CBD5E0')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#CBD5E0')
    
    short_name = (feature_name[:40] + '…'
                  if len(feature_name) > 40 else feature_name)
    ax.set_title(f'SHAP Dependence Plot — {short_name}\n'
                 'How this feature\'s value drives the bankruptcy prediction',
                 color='white', fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_dependence_top_feature.png',
                dpi=150, bbox_inches='tight',
                facecolor='#0F1923')
    plt.close()
    print(f"Dependence plot saved to {output_dir}/shap_dependence_top_feature.png")
    
psi_summary = run_psi_analysis(
    X_train_sc,
    X_test_sc,
    X_oos_sc,
    feature_names = X.columns.tolist(),
    output_dir    = OUTPUT_DIR
)

# SHAP Analysis
shap_vals, shap_importance = run_shap_analysis(
    model,
    X_train_sc,
    X_test_sc,
    feature_names = X.columns.tolist(),
    output_dir    = OUTPUT_DIR
)

print("\nAll outputs saved to model_outputs/")
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