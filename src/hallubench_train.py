import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, roc_auc_score, roc_curve, auc
)
from xgboost import XGBClassifier

os.makedirs('data/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ── 1. Load ──────────────────────────────────────────────────
df = pd.read_csv('data/kyc_cases_external.csv')
print(f"HalluBench dataset: {df.shape}")

# ── 2. Feature engineering + BINARY target ───────────────────
df_model = df.copy()
df_model['is_pep']      = (df_model['PEPStatus'] == 'Yes').astype(int)
df_model['is_sanction'] = (df_model['SanctionStatus'] == 'Yes').astype(int)
df_model['aml_flag']    = (df_model['AMLFlag'] == 'Yes').astype(int)

le_country_h    = LabelEncoder()
le_occupation_h = LabelEncoder()
df_model['country_enc']    = le_country_h.fit_transform(df_model['Country'])
df_model['occupation_enc'] = le_occupation_h.fit_transform(df_model['Occupation'])

# BINARY: High Risk (1) vs Non-High Risk (0)
df_model['is_high_risk'] = (df_model['RiskCategory'] == 'High').astype(int)
print(f"\nBinary target distribution:")
print(df_model['is_high_risk'].value_counts())
print(f"High Risk proportion: {df_model['is_high_risk'].mean():.3f}")

features = [
    'country_enc', 'occupation_enc', 'Income',
    'is_pep', 'is_sanction', 'aml_flag', 'VerifiedDocuments'
]
X = df_model[features]
y = df_model['is_high_risk']

# ── 3. Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── 4. Tuning ────────────────────────────────────────────────
print("\n" + "="*60)
print("Hyperparameter Tuning (binary High vs Non-High)")
print("="*60)

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2, 0.5]
}

xgb_tuning = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    xgb_tuning, param_distributions=param_dist, n_iter=30,
    scoring='f1',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42, n_jobs=-1, verbose=1
)
random_search.fit(X_train, y_train)

print(f"\nBest parameters:")
for k, v in random_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV F1: {random_search.best_score_:.4f}")

xgb_best = random_search.best_estimator_
y_pred  = xgb_best.predict(X_test)
y_proba = xgb_best.predict_proba(X_test)[:, 1]

# ── 5. Results ───────────────────────────────────────────────
print(f"\n{'='*60}")
print("RESULTS — Binary High Risk classification on HalluBench")
print(f"{'='*60}")
print(f"Accuracy:            {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score:            {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:             {roc_auc_score(y_test, y_proba):.4f}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Non-High Risk', 'High Risk']))

# ── 6. Confusion matrix ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-High', 'High'],
            yticklabels=['Non-High', 'High'],
            ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title(
    'Figure 4.X: Confusion matrix — binary High Risk detection\n'
    '(BankKYC-HalluBench, test set n = 20,000)',
    fontweight='bold', fontsize=11)
ax.set_xlabel('Predicted label', fontsize=11)
ax.set_ylabel('True label', fontsize=11)
plt.tight_layout()
plt.savefig('data/figures/fig_hallubench_binary_cm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix saved → data/figures/fig_hallubench_binary_cm.png")

# ── 7. ROC curve ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'High Risk (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='#7f8c8d', lw=1.5, linestyle='--', label='Random classifier')
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title(
    'Figure 4.X: ROC curve — binary High Risk detection\n'
    '(BankKYC-HalluBench, tuned XGBoost)',
    fontweight='bold', fontsize=11)
ax2.legend(loc='lower right', fontsize=11)
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('data/figures/fig_hallubench_binary_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("ROC curve saved → data/figures/fig_hallubench_binary_roc.png")

# ── 8. SHAP ──────────────────────────────────────────────────
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(xgb_best)
X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(X_sample)

shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=features).sort_values()

fig3, ax3 = plt.subplots(figsize=(9, 5))
ax3.barh(range(len(shap_imp)), shap_imp.values, color='coral',
         edgecolor='white', linewidth=0.5)
ax3.set_yticks(range(len(shap_imp)))
ax3.set_yticklabels([f.replace('_', ' ') for f in shap_imp.index])
ax3.set_xlabel('Mean absolute SHAP value')
ax3.set_title(
    'Figure 4.X: SHAP feature importance — High Risk detection\n'
    '(BankKYC-HalluBench, tuned XGBoost)',
    fontweight='bold', fontsize=11)
ax3.grid(axis='x', alpha=0.3, linestyle='--')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('data/figures/fig_hallubench_binary_shap.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP figure saved → data/figures/fig_hallubench_binary_shap.png")

joblib.dump(xgb_best, 'models/xgboost_hallubench_binary.pkl')
print("\nModel saved. Done.")