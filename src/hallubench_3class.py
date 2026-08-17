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
    accuracy_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

os.makedirs('data/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ── 1. Load external dataset ──────────────────────────────────
df = pd.read_csv('data/kyc_cases_external.csv')
print(f"HalluBench dataset: {df.shape}")
print(f"RiskCategory distribution:\n{df['RiskCategory'].value_counts()}")

# ── 2. Feature engineering ────────────────────────────────────
df_model = df.copy()

df_model['is_pep']      = (df_model['PEPStatus'] == 'Yes').astype(int)
df_model['is_sanction'] = (df_model['SanctionStatus'] == 'Yes').astype(int)
df_model['aml_flag']    = (df_model['AMLFlag'] == 'Yes').astype(int)

le_country_h = LabelEncoder()
le_occupation_h = LabelEncoder()
df_model['country_enc']    = le_country_h.fit_transform(df_model['Country'])
df_model['occupation_enc'] = le_occupation_h.fit_transform(df_model['Occupation'])

label_map = {'Low': 0, 'Medium': 1, 'High': 2}
df_model['risk_label'] = df_model['RiskCategory'].map(label_map)

features = [
    'country_enc', 'occupation_enc', 'Income',
    'is_pep', 'is_sanction', 'aml_flag', 'VerifiedDocuments'
]
X = df_model[features]
y = df_model['risk_label']

print(f"\nFeatures: {features}")
print(f"Target classes: {sorted(y.unique())} (0=Low, 1=Medium, 2=High)")

# ── 3. Train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ── 4. Hyperparameter tuning (same procedure as thesis) ───────
print("\n" + "="*60)
print("Hyperparameter Tuning (RandomizedSearchCV)")
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

xgb_tuning = XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    xgb_tuning, param_distributions=param_dist, n_iter=30,
    scoring='f1_weighted',
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
y_proba = xgb_best.predict_proba(X_test)

# ── 5. Results ────────────────────────────────────────────────
label_names = ['Low Risk', 'Medium Risk', 'High Risk']
print(f"\n{'='*60}")
print("RESULTS — Tuned XGBoost trained on HalluBench (3-class)")
print(f"{'='*60}")
print(f"Accuracy:            {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"AUC-ROC (OvR):       {roc_auc_score(y_test, y_proba, multi_class='ovr'):.4f}")
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_names))

# ── 6. Confusion matrix ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title(
    'Figure 4.X: Confusion matrix — tuned XGBoost retrained on external dataset\n'
    '(BankKYC-HalluBench, three-class attempt, test set n = 20,000)',
    fontweight='bold', fontsize=11)
ax.set_xlabel('Predicted label', fontsize=11)
ax.set_ylabel('True label', fontsize=11)
plt.tight_layout()
plt.savefig('data/figures/fig_hallubench_3class_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix saved → data/figures/fig_hallubench_3class_confusion_matrix.png")

# ── 7. SHAP on retrained model ────────────────────────────────
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(xgb_best)
X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_high = shap_values[2]
else:
    shap_high = shap_values[:, :, 2]

shap_imp = pd.Series(np.abs(shap_high).mean(axis=0), index=features).sort_values()

fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.barh(range(len(shap_imp)), shap_imp.values, color='coral',
         edgecolor='white', linewidth=0.5)
ax2.set_yticks(range(len(shap_imp)))
ax2.set_yticklabels([f.replace('_', ' ') for f in shap_imp.index])
ax2.set_xlabel('Mean absolute SHAP value')
ax2.set_title(
    'Figure 4.X: SHAP feature importance — retrained model on external dataset\n'
    '(BankKYC-HalluBench, three-class attempt, High Risk class)',
    fontweight='bold', fontsize=11)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('data/figures/fig_hallubench_3class_shap.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP figure saved → data/figures/fig_hallubench_3class_shap.png")

joblib.dump(xgb_best, 'models/xgboost_hallubench_3class.pkl')
print("\nModel saved → models/xgboost_hallubench_3class.pkl")
print("Done.")
