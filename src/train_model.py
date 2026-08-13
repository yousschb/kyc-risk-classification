import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, learning_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from xgboost import XGBClassifier


def cv_f1_weighted_cost_sensitive(estimator, X, y, cv):
    """Five-fold weighted-F1 cross-validation for the cost-sensitive model.

    The class weights are applied inside each fold, so the model is evaluated
    with the same weighting it is trained with. A plain cross_val_score would
    silently drop the sample weights and return the baseline score instead.
    """
    Xv, yv = X.values, y.values
    scores = []
    for tr, te in cv.split(Xv, yv):
        w = np.ones(len(tr))
        w[yv[tr] == 2] = 3.0
        w[yv[tr] == 1] = 1.5
        m = clone(estimator)
        m.fit(Xv[tr], yv[tr], sample_weight=w)
        scores.append(f1_score(yv[te], m.predict(Xv[te]), average='weighted'))
    return np.array(scores)

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

df_model = df.drop(columns=['client_id', 'country_risk'])

le_country = LabelEncoder()
le_sector = LabelEncoder()

df_model = df_model.copy()
df_model['country'] = le_country.fit_transform(df_model['country'])
df_model['sector'] = le_sector.fit_transform(df_model['sector'])

os.makedirs('models', exist_ok=True)
joblib.dump(le_country, 'models/le_country.pkl')
joblib.dump(le_sector, 'models/le_sector.pkl')

X = df_model.drop(columns=['risk_label'])
y = df_model['risk_label']

print(f"Features: {list(X.columns)}")
print(f"Target classes: {sorted(y.unique())} (0=Low, 1=Medium, 2=High)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

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
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_rf = cross_val_score(rf, X, y, cv=cv, scoring='f1_weighted')

print(f"\nRandom Forest Results:")
print(f"  Accuracy:              {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  F1-Score (weighted):   {f1_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"  AUC-ROC (OvR):         {roc_auc_score(y_test, y_proba_rf, multi_class='ovr'):.4f}")
print(f"  CV F1 (5-fold):        {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_rf, target_names=['Low','Medium','High'])}")

# =============================================
# 4. XGBOOST BASELINE
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
# 4b. XGBOOST COST-SENSITIVE
# =============================================
print("="*60)
print("STEP 4b — XGBoost Cost-Sensitive (Asymmetric Misclassification)")
print("="*60)

sample_weights = np.ones(len(y_train))
sample_weights[y_train == 2] = 3.0
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

cv_scores_cs = cv_f1_weighted_cost_sensitive(xgb_cs, X, y, cv)

print(f"\nXGBoost Cost-Sensitive Results:")
print(f"  Accuracy:              {accuracy_score(y_test, y_pred_cs):.4f}")
print(f"  F1-Score (weighted):   {f1_score(y_test, y_pred_cs, average='weighted'):.4f}")
print(f"  AUC-ROC (OvR):         {roc_auc_score(y_test, y_proba_cs, multi_class='ovr'):.4f}")
print(f"  CV F1 (5-fold):        {cv_scores_cs.mean():.4f} ± {cv_scores_cs.std():.4f}")
print(f"\nClassification Report (Cost-Sensitive):\n{classification_report(y_test, y_pred_cs, target_names=['Low','Medium','High'])}")

cm_cs = confusion_matrix(y_test, y_pred_cs)
cm_std_xgb = confusion_matrix(y_test, y_pred_xgb)
print(f"\nHigh Risk Recall comparison:")
print(f"  Standard XGBoost:      {cm_std_xgb[2,2]/cm_std_xgb[2].sum():.4f}")
print(f"  Cost-Sensitive XGBoost:{cm_cs[2,2]/cm_cs[2].sum():.4f}")

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
# 6. CONFUSION MATRICES (RF vs XGBoost baseline)
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
plt.close()
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
plt.close()
print("Feature importances saved → data/figures/feature_importances.png")

# =============================================
# 8. SAVE BASELINE MODELS
# =============================================
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(xgb, 'models/xgboost.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')

print("\n" + "="*60)
print("MODELS SAVED:")
print("  models/random_forest.pkl")
print("  models/xgboost.pkl")
print("  models/le_country.pkl")
print("  models/le_sector.pkl")
print("  models/feature_names.pkl")
print("="*60)

# =============================================
# STEP 6 — HYPERPARAMETER TUNING
# =============================================
print("\n" + "="*60)
print("STEP 6 — Hyperparameter Tuning (RandomizedSearchCV)")
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

# Confusion matrix — Tuned XGBoost (Figure 4.4 du mémoire)
fig3, ax3 = plt.subplots(figsize=(6, 5))
cm_tuned = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'], ax=ax3)
ax3.set_title('XGBoost (Tuned) — Confusion Matrix', fontsize=13, fontweight='bold')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('data/figures/confusion_matrix_xgb_tuned.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix (tuned XGBoost) saved → data/figures/confusion_matrix_xgb_tuned.png")

joblib.dump(xgb_best, 'models/xgboost_tuned.pkl')
print("Tuned model saved → models/xgboost_tuned.pkl")

# =============================================
# STEP 7 — SHAP ANALYSIS
# =============================================
print("\n" + "="*60)
print("STEP 7 — SHAP Analysis")
print("="*60)

explainer = shap.TreeExplainer(xgb_best)
X_test_shap = X_test.copy()
shap_values = explainer.shap_values(X_test_shap)

# Compatibilité format SHAP : liste (ancienne API) ou array 3D (nouvelle API)
if isinstance(shap_values, list):
    shap_high_risk = shap_values[2]
else:
    shap_high_risk = shap_values[:, :, 2]

# --- Figure 4.1 : Risk label distribution ---
fig41, ax41 = plt.subplots(figsize=(8, 5))
risk_counts = df['risk_label'].value_counts().sort_index()
risk_props = risk_counts / len(df) * 100
bars = ax41.bar(['Low', 'Medium', 'High'], risk_counts,
                color=['#2ecc71', '#f39c12', '#e74c3c'],
                edgecolor='black', linewidth=1.2)
ax41.set_ylabel('Number of clients')
ax41.set_title('Figure 4.1: Risk label distribution (n = 5,000)', fontweight='bold')
ax41.grid(axis='y', alpha=0.3, linestyle='--')
for bar, count, prop in zip(bars, risk_counts, risk_props):
    ax41.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
              f'{count}\n({prop:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/fig_4_1_risk_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.1 saved → data/figures/fig_4_1_risk_distribution.png")

# --- Figure 4.4 : Confusion matrices RF vs Tuned XGBoost ---
fig44, axes44 = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_pred, title in zip(
    axes44,
    [y_pred_rf, y_pred_best],
    ['Random Forest', 'Tuned XGBoost']
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'], ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.suptitle('Figure 4.4: Confusion matrices', fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/fig_4_4_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.4 saved → data/figures/fig_4_4_confusion_matrices.png")

# --- Figure 4.5 : Feature importance comparison RF vs SHAP XGBoost ---
fig45, axes45 = plt.subplots(1, 2, figsize=(14, 6))
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
rf_importance.plot(kind='barh', ax=axes45[0], color='steelblue', edgecolor='black', linewidth=0.8)
axes45[0].set_title('Random Forest: Mean decrease impurity', fontweight='bold')
axes45[0].set_xlabel('Importance score')
axes45[0].grid(axis='x', alpha=0.3, linestyle='--')

shap_importance = np.abs(shap_high_risk).mean(axis=0)
shap_imp_series = pd.Series(shap_importance, index=X.columns).sort_values()
shap_imp_series.plot(kind='barh', ax=axes45[1], color='coral', edgecolor='black', linewidth=0.8)
axes45[1].set_title('XGBoost: Mean |SHAP| (High Risk)', fontweight='bold')
axes45[1].set_xlabel('Mean absolute SHAP value')
axes45[1].grid(axis='x', alpha=0.3, linestyle='--')

plt.suptitle('Figure 4.5: Feature importance comparison', fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/fig_4_5_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.5 saved → data/figures/fig_4_5_importance_comparison.png")

# --- Figure 4.6 : XGBoost mean |SHAP| bar plot ---
fig46, ax46 = plt.subplots(figsize=(10, 7))
shap_imp_series.sort_values(ascending=True).plot(kind='barh', ax=ax46,
                                                 color='coral', edgecolor='black', linewidth=0.8)
ax46.set_title('Figure 4.6: XGBoost mean |SHAP| values for High Risk class', fontweight='bold')
ax46.set_xlabel('Mean absolute SHAP value')
ax46.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('data/figures/fig_4_6_shap_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.6 saved → data/figures/fig_4_6_shap_importance.png")

# --- Figure 4.7 : SHAP beeswarm summary plot ---
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_high_risk, X_test_shap, show=False)
plt.title('Figure 4.7: SHAP summary beeswarm plot (High Risk class)', fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/fig_4_7_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.7 saved → data/figures/fig_4_7_beeswarm.png")

# --- Figure 4.8 : SHAP dependence plot ---
plt.figure(figsize=(10, 7))
shap.dependence_plot(
    'adverse_media_score',
    shap_high_risk,
    X_test_shap,
    interaction_index='nb_countries_involved',
    show=False
)
plt.title('Figure 4.8: SHAP dependence plot', fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/fig_4_8_dependence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.8 saved → data/figures/fig_4_8_dependence.png")

# --- Figure 4.9 : SHAP waterfall chart (client High Risk) ---
idx_high = y_test[y_test == 2].index[0]
X_client = X_test.loc[[idx_high]]
client_shap_values = explainer(X_client)

# Compatibilité format : Explanation object (nouvelle API) ou array (ancienne)
plt.figure(figsize=(10, 8))
try:
    # Nouvelle API : Explanation object avec 3 dimensions (obs, features, classes)
    shap.plots.waterfall(client_shap_values[0, :, 2], show=False)
except (IndexError, TypeError):
    # Ancienne API : liste de matrices
    shap_exp = shap.Explanation(
        values=shap_values[2][X_test.index.get_loc(idx_high)],
        base_values=explainer.expected_value[2],
        data=X_client.values[0],
        feature_names=list(X.columns)
    )
    shap.plots.waterfall(shap_exp, show=False)
plt.title('Figure 4.9: SHAP waterfall chart (High Risk client)', fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/fig_4_9_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.9 saved → data/figures/fig_4_9_waterfall.png")

# --- Figure 4.10 : Learning curve ---
train_sizes, train_scores, test_scores = learning_curve(
    xgb_best,
    X,
    y,
    cv=5,
    scoring='f1_weighted',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)
noise_ceiling = 1 - 0.14

fig410, ax410 = plt.subplots(figsize=(10, 6))
ax410.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training F1')
ax410.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                   alpha=0.15, color='steelblue')
ax410.plot(train_sizes, test_mean, 's-', color='coral', label='Validation F1')
ax410.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                   alpha=0.15, color='coral')
ax410.axhline(y=noise_ceiling, color='gray', linestyle='--', linewidth=2,
              label='Noise ceiling (~0.86)')
ax410.set_xlabel('Training set size')
ax410.set_ylabel('Weighted F1-score')
ax410.set_title('Figure 4.10: Learning curve of tuned XGBoost', fontweight='bold')
ax410.legend()
ax410.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('data/figures/fig_4_10_learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.10 saved → data/figures/fig_4_10_learning_curve.png")

# --- Figure V.3 : Pearson correlation matrix ---
# Note: df_model contient les variables label-encodées (country, sector en entiers)
figV3, axV3 = plt.subplots(figsize=(12, 10))
df_corr = df_model.corr(numeric_only=True)
mask = np.triu(np.ones_like(df_corr, dtype=bool))
sns.heatmap(df_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, ax=axV3, cbar_kws={'shrink': 0.8})
axV3.set_title('Figure V.3: Pearson correlation matrix (label-encoded features)', fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/fig_v3_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure V.3 saved → data/figures/fig_v3_correlation_matrix.png")

# =============================================
# STEP 8 — TABLES
# =============================================
print("\n" + "="*60)
print("STEP 8 — Generating Tables")
print("="*60)

table_dir = 'data/tables'
os.makedirs(table_dir, exist_ok=True)

def save_table(df_t, name):
    df_t.to_csv(f"{table_dir}/{name}.csv", index=False)
    print(f"Table saved → {table_dir}/{name}.csv")

# Table 4.1 — Descriptive statistics
table_4_1 = pd.DataFrame({
    'Variable': [
        'transaction_volume (CHF)',
        'account_age_years',
        'nb_countries_involved',
        'cash_ratio',
        'adverse_media_score',
        'beneficial_owner_complexity'
    ],
    'Mean': [
        round(df['transaction_volume'].mean(), 0),
        round(df['account_age_years'].mean(), 1),
        round(df['nb_countries_involved'].mean(), 1),
        round(df['cash_ratio'].mean(), 3),
        round(df['adverse_media_score'].mean(), 2),
        round(df['beneficial_owner_complexity'].mean(), 2)
    ],
    'Median': [
        round(df['transaction_volume'].median(), 0),
        round(df['account_age_years'].median(), 0),
        round(df['nb_countries_involved'].median(), 0),
        round(df['cash_ratio'].median(), 3),
        round(df['adverse_media_score'].median(), 0),
        round(df['beneficial_owner_complexity'].median(), 0)
    ],
    'Std Dev': [
        round(df['transaction_volume'].std(), 0),
        round(df['account_age_years'].std(), 1),
        round(df['nb_countries_involved'].std(), 1),
        round(df['cash_ratio'].std(), 3),
        round(df['adverse_media_score'].std(), 2),
        round(df['beneficial_owner_complexity'].std(), 2)
    ],
    'Min': [
        round(df['transaction_volume'].min(), 0),
        int(df['account_age_years'].min()),
        int(df['nb_countries_involved'].min()),
        round(df['cash_ratio'].min(), 3),
        int(df['adverse_media_score'].min()),
        int(df['beneficial_owner_complexity'].min())
    ],
    'Max': [
        round(df['transaction_volume'].max(), 0),
        int(df['account_age_years'].max()),
        int(df['nb_countries_involved'].max()),
        round(df['cash_ratio'].max(), 3),
        int(df['adverse_media_score'].max()),
        int(df['beneficial_owner_complexity'].max())
    ],
    'Key Observation': [
        'Log-normal wealth distribution',
        'Established private banking base',
        'High variance; driver for High Risk',
        'Beta(2,5); wire-transfer dominant',
        '70% zero; heavily right-skewed',
        'Majority simple structures'
    ]
})
save_table(table_4_1, 'table_4_1_descriptive_stats')

# Table 4.2 — Model performance comparison
best_idx = random_search.best_index_
cv_std_tuned = random_search.cv_results_['std_test_score'][best_idx]

table_4_2 = pd.DataFrame({
    'Metric': [
        'Accuracy',
        'F1-Score (weighted)',
        'AUC-ROC (one-vs-rest)',
        'Cross-validated F1 mean (5-fold)',
        'High Risk Recall'
    ],
    'Random Forest': [
        f"{accuracy_score(y_test, y_pred_rf)*100:.1f}%",
        f"{f1_score(y_test, y_pred_rf, average='weighted'):.3f}",
        f"{roc_auc_score(y_test, y_proba_rf, multi_class='ovr'):.3f}",
        f"{cv_scores_rf.mean():.3f} ± {cv_scores_rf.std():.3f}",
        f"{classification_report(y_test, y_pred_rf, output_dict=True)['2']['recall']:.3f}"
    ],
    'XGBoost (Baseline)': [
        f"{accuracy_score(y_test, y_pred_xgb)*100:.1f}%",
        f"{f1_score(y_test, y_pred_xgb, average='weighted'):.3f}",
        f"{roc_auc_score(y_test, y_proba_xgb, multi_class='ovr'):.3f}",
        f"{cv_scores_xgb.mean():.3f} ± {cv_scores_xgb.std():.3f}",
        f"{classification_report(y_test, y_pred_xgb, output_dict=True)['2']['recall']:.3f}"
    ],
    'XGBoost (Tuned)': [
        f"{accuracy_score(y_test, y_pred_best)*100:.1f}%",
        f"{f1_score(y_test, y_pred_best, average='weighted'):.3f}",
        f"{roc_auc_score(y_test, y_proba_best, multi_class='ovr'):.3f}",
        f"{random_search.best_score_:.3f} ± {cv_std_tuned:.3f}",
        f"{classification_report(y_test, y_pred_best, output_dict=True)['2']['recall']:.3f}"
    ],
    'XGBoost (Cost-Sensitive)': [
        f"{accuracy_score(y_test, y_pred_cs)*100:.1f}%",
        f"{f1_score(y_test, y_pred_cs, average='weighted'):.3f}",
        f"{roc_auc_score(y_test, y_proba_cs, multi_class='ovr'):.3f}",
        f"{cv_scores_cs.mean():.3f} ± {cv_scores_cs.std():.3f}",
        f"{classification_report(y_test, y_pred_cs, output_dict=True)['2']['recall']:.3f}"
    ]
})
save_table(table_4_2, 'table_4_2_model_comparison')

# Table 4.3 — Per-class report tuned XGBoost
rep = classification_report(y_test, y_pred_best, output_dict=True)
table_4_3 = pd.DataFrame({
    'Class': ['Low Risk', 'Medium Risk', 'High Risk', 'Weighted Average'],
    'Precision': [
        round(rep['0']['precision'], 2),
        round(rep['1']['precision'], 2),
        round(rep['2']['precision'], 2),
        round(rep['weighted avg']['precision'], 2)
    ],
    'Recall': [
        round(rep['0']['recall'], 2),
        round(rep['1']['recall'], 2),
        round(rep['2']['recall'], 2),
        round(rep['weighted avg']['recall'], 2)
    ],
    'F1-Score': [
        round(rep['0']['f1-score'], 2),
        round(rep['1']['f1-score'], 2),
        round(rep['2']['f1-score'], 2),
        round(rep['weighted avg']['f1-score'], 3)
    ],
    'Support (n)': [
        int(rep['0']['support']),
        int(rep['1']['support']),
        int(rep['2']['support']),
        int(rep['weighted avg']['support'])
    ]
})
save_table(table_4_3, 'table_4_3_classification_report_tuned_xgb')

print("\n" + "="*60)
print("ALL FIGURES AND TABLES GENERATED")
print("  Figures → data/figures/")
print("  Tables  → data/tables/")
print("="*60)
print("\nDone.")
