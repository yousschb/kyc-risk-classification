import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from xgboost import XGBClassifier

# =============================================
# 1. LOAD DATA
# =============================================
print("="*60)
print("STEP 1 — Loading data")
print("="*60)

df = pd.read_csv('data/kyc_synthetic.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['risk_label'].value_counts().sort_index()}")

# =============================================
# 2. PREPROCESSING
# =============================================
print("\n" + "="*60)
print("STEP 2 — Preprocessing")
print("="*60)

# Drop non-feature columns
df_model = df.drop(columns=['client_id', 'country_risk'])  # country_risk is redundant with country

# Encode categorical variables
le_country = LabelEncoder()
le_sector = LabelEncoder()

df_model['country'] = le_country.fit_transform(df_model['country'])
df_model['sector'] = le_sector.fit_transform(df_model['sector'])

# Save encoders for later use in Streamlit
os.makedirs('models', exist_ok=True)
joblib.dump(le_country, 'models/le_country.pkl')
joblib.dump(le_sector, 'models/le_sector.pkl')

# Features and target
X = df_model.drop(columns=['risk_label'])
y = df_model['risk_label']

print(f"Features: {list(X.columns)}")
print(f"Target classes: {sorted(y.unique())} (0=Low, 1=Medium, 2=High)")

# Train/test split — stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# Scale features (important for some metrics, good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')

# =============================================
# 3. RANDOM FOREST
# =============================================
print("\n" + "="*60)
print("STEP 3 — Training Random Forest")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # handles class imbalance
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_rf = cross_val_score(rf, X, y, cv=cv, scoring='f1_weighted')

print(f"\nRandom Forest Results:")
print(f"  Accuracy:              {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  F1-Score (weighted):   {f1_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"  AUC-ROC (OvR):         {roc_auc_score(y_test, y_proba_rf, multi_class='ovr'):.4f}")
print(f"  CV F1 (5-fold):        {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_rf, target_names=['Low','Medium','High'])}")

# =============================================
# 4. XGBOOST
# =============================================
print("="*60)
print("STEP 4 — Training XGBoost")
print("="*60)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)

cv_scores_xgb = cross_val_score(xgb, X, y, cv=cv, scoring='f1_weighted')

print(f"\nXGBoost Results:")
print(f"  Accuracy:              {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"  F1-Score (weighted):   {f1_score(y_test, y_pred_xgb, average='weighted'):.4f}")
print(f"  AUC-ROC (OvR):         {roc_auc_score(y_test, y_proba_xgb, multi_class='ovr'):.4f}")
print(f"  CV F1 (5-fold):        {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_xgb, target_names=['Low','Medium','High'])}")

# =============================================
# 4b. XGBOOST — COST-SENSITIVE (asymmetric)
# =============================================
print("="*60)
print("STEP 4b — XGBoost Cost-Sensitive (Asymmetric Misclassification)")
print("="*60)
print("Rationale: Misclassifying a High Risk client as Low Risk carries")
print("far greater regulatory consequences than the reverse (FINMA/AMLA).")
print("We apply a sample weight matrix penalizing High->Low errors 5x more.")

# Build sample weights — penalize High Risk misclassification
sample_weights = np.ones(len(y_train))
# Clients qui sont High Risk (label=2) reçoivent un poids 3x plus élevé
sample_weights[y_train == 2] = 3.0
# Clients Medium Risk reçoivent un poids 1.5x
sample_weights[y_train == 1] = 1.5

xgb_cs = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

xgb_cs.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_cs = xgb_cs.predict(X_test)
y_proba_cs = xgb_cs.predict_proba(X_test)

cv_scores_cs = cross_val_score(xgb_cs, X, y, cv=cv, scoring='f1_weighted')

print(f"\nXGBoost Cost-Sensitive Results:")
print(f"  Accuracy:              {accuracy_score(y_test, y_pred_cs):.4f}")
print(f"  F1-Score (weighted):   {f1_score(y_test, y_pred_cs, average='weighted'):.4f}")
print(f"  AUC-ROC (OvR):         {roc_auc_score(y_test, y_proba_cs, multi_class='ovr'):.4f}")
print(f"  CV F1 (5-fold):        {cv_scores_cs.mean():.4f} ± {cv_scores_cs.std():.4f}")
print(f"\nClassification Report (Cost-Sensitive):\n{classification_report(y_test, y_pred_cs, target_names=['Low','Medium','High'])}")

# Analyse spécifique High Risk recall
cm_cs = confusion_matrix(y_test, y_pred_cs)
cm_std = confusion_matrix(y_test, y_pred_xgb)
print(f"\nHigh Risk Recall comparison:")
print(f"  Standard XGBoost:      {cm_std[2,2]/cm_std[2].sum():.4f}")
print(f"  Cost-Sensitive XGBoost:{cm_cs[2,2]/cm_cs[2].sum():.4f}")

# Save cost-sensitive model
joblib.dump(xgb_cs, 'models/xgboost_cost_sensitive.pkl')

# =============================================
# 5. COMPARISON TABLE
# =============================================
print("="*60)
print("STEP 5 — Model Comparison")
print("="*60)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Score (weighted)', 'AUC-ROC', 'CV F1 Mean', 'CV F1 Std'],
    'Random Forest': [
        f"{accuracy_score(y_test, y_pred_rf):.4f}",
        f"{f1_score(y_test, y_pred_rf, average='weighted'):.4f}",
        f"{roc_auc_score(y_test, y_proba_rf, multi_class='ovr'):.4f}",
        f"{cv_scores_rf.mean():.4f}",
        f"{cv_scores_rf.std():.4f}"
    ],
    'XGBoost': [
        f"{accuracy_score(y_test, y_pred_xgb):.4f}",
        f"{f1_score(y_test, y_pred_xgb, average='weighted'):.4f}",
        f"{roc_auc_score(y_test, y_proba_xgb, multi_class='ovr'):.4f}",
        f"{cv_scores_xgb.mean():.4f}",
        f"{cv_scores_xgb.std():.4f}"
    ]
})
print(comparison.to_string(index=False))

# =============================================
# 6. CONFUSION MATRICES
# =============================================
os.makedirs('data/figures', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_pred, title in zip(
    axes,
    [y_pred_rf, y_pred_xgb],
    ['Random Forest', 'XGBoost']
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'], ax=ax)
    ax.set_title(f'{title} — Confusion Matrix', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('data/figures/confusion_matrices.png', dpi=150, bbox_inches='tight')
print("\nConfusion matrices saved → data/figures/confusion_matrices.png")

# =============================================
# 7. FEATURE IMPORTANCE
# =============================================
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

for ax, model, title in zip(
    axes2,
    [rf, xgb],
    ['Random Forest', 'XGBoost']
):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'{title} — Feature Importances', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('data/figures/feature_importances.png', dpi=150, bbox_inches='tight')
print("Feature importances saved → data/figures/feature_importances.png")

# =============================================
# 8. SAVE MODELS
# =============================================
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(xgb, 'models/xgboost.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')

print("\n" + "="*60)
print("MODELS SAVED:")
print("  models/random_forest.pkl")
print("  models/xgboost.pkl")
print("  models/scaler.pkl")
print("  models/le_country.pkl")
print("  models/le_sector.pkl")
print("  models/feature_names.pkl")
print("="*60)
print("\nDone. Ready for SHAP analysis.")

# =============================================
# STEP 6 — HYPERPARAMETER TUNING (RandomizedSearchCV)
# =============================================
print("\n" + "="*60)
print("STEP 6 — Hyperparameter Tuning (RandomizedSearchCV)")
print("="*60)
print("Searching best parameters for XGBoost...")

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
    xgb_tuning,
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1_weighted',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for k, v in random_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV F1: {random_search.best_score_:.4f}")

xgb_best = random_search.best_estimator_
y_pred_best = xgb_best.predict(X_test)
y_proba_best = xgb_best.predict_proba(X_test)

print(f"\nTuned XGBoost Results:")
print(f"  Accuracy:              {accuracy_score(y_test, y_pred_best):.4f}")
print(f"  F1-Score (weighted):   {f1_score(y_test, y_pred_best, average='weighted'):.4f}")
print(f"  AUC-ROC (OvR):         {roc_auc_score(y_test, y_proba_best, multi_class='ovr'):.4f}")
print(f"\nClassification Report (Tuned):\n{classification_report(y_test, y_pred_best, target_names=['Low','Medium','High'])}")

joblib.dump(xgb_best, 'models/xgboost_tuned.pkl')
print("Tuned model saved → models/xgboost_tuned.pkl")