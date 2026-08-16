import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import shap
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, learning_curve
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve, auc
)
from xgboost import XGBClassifier

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.dpi': 150,
})
PALETTE = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
BLUE = '#2c5f8a'
CORAL = '#c0392b'
GRAY = '#7f8c8d'


def cv_f1_weighted_cost_sensitive(estimator, X, y, cv):
    """Five-fold weighted-F1 cross-validation for the cost-sensitive model."""
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


os.makedirs('data/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

# =============================================
# 1. LOAD DATA
# =============================================
print("=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

df = pd.read_csv('data/kyc_synthetic.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['risk_label'].value_counts().sort_index()}")

# =============================================
# 2. PREPROCESSING
# =============================================
print("\n" + "=" * 60)
print("STEP 2 — Preprocessing")
print("=" * 60)

df_model = df.drop(columns=['client_id', 'country_risk']).copy()

le_country = LabelEncoder()
le_sector = LabelEncoder()
df_model['country'] = le_country.fit_transform(df_model['country'])
df_model['sector'] = le_sector.fit_transform(df_model['sector'])

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
print("\n" + "=" * 60)
print("STEP 3 — Training Random Forest")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
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
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_rf, target_names=['Low', 'Medium', 'High'])}")

# =============================================
# 4. XGBOOST BASELINE
# =============================================
print("=" * 60)
print("STEP 4 — Training XGBoost baseline")
print("=" * 60)

xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
    random_state=42, n_jobs=-1
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
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_xgb, target_names=['Low', 'Medium', 'High'])}")

# =============================================
# 4b. XGBOOST COST-SENSITIVE
# =============================================
print("=" * 60)
print("STEP 4b — XGBoost Cost-Sensitive")
print("=" * 60)

sample_weights = np.ones(len(y_train))
sample_weights[y_train == 2] = 3.0
sample_weights[y_train == 1] = 1.5

xgb_cs = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
    random_state=42, n_jobs=-1
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
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_cs, target_names=['Low', 'Medium', 'High'])}")

cm_cs = confusion_matrix(y_test, y_pred_cs)
cm_std_xgb = confusion_matrix(y_test, y_pred_xgb)
print(f"\nHigh Risk Recall — Standard XGBoost: {cm_std_xgb[2,2]/cm_std_xgb[2].sum():.4f}")
print(f"High Risk Recall — Cost-Sensitive:    {cm_cs[2,2]/cm_cs[2].sum():.4f}")

joblib.dump(xgb_cs, 'models/xgboost_cost_sensitive.pkl')

# =============================================
# 5. HYPERPARAMETER TUNING
# =============================================
print("\n" + "=" * 60)
print("STEP 5 — Hyperparameter Tuning (RandomizedSearchCV)")
print("=" * 60)

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
y_pred_best = xgb_best.predict(X_test)
y_proba_best = xgb_best.predict_proba(X_test)

print(f"\nTuned XGBoost Results:")
print(f"  Accuracy:              {accuracy_score(y_test, y_pred_best):.4f}")
print(f"  F1-Score (weighted):   {f1_score(y_test, y_pred_best, average='weighted'):.4f}")
print(f"  AUC-ROC (OvR):         {roc_auc_score(y_test, y_proba_best, multi_class='ovr'):.4f}")
print(f"\nClassification Report (Tuned):\n{classification_report(y_test, y_pred_best, target_names=['Low', 'Medium', 'High'])}")

joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(xgb, 'models/xgboost.pkl')
joblib.dump(xgb_best, 'models/xgboost_tuned.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')
print("\nModels saved.")

# =============================================
# 6. SHAP
# =============================================
print("\n" + "=" * 60)
print("STEP 6 — SHAP Analysis")
print("=" * 60)

explainer = shap.TreeExplainer(xgb_best)
X_test_shap = X_test.copy()
shap_values = explainer.shap_values(X_test_shap)

if isinstance(shap_values, list):
    shap_high_risk = shap_values[2]
else:
    shap_high_risk = shap_values[:, :, 2]

shap_importance = np.abs(shap_high_risk).mean(axis=0)
shap_imp_series = pd.Series(shap_importance, index=X.columns).sort_values()

# =============================================
# 7. FIGURES
# =============================================
print("\n" + "=" * 60)
print("STEP 7 — Generating Figures")
print("=" * 60)

# ── Figure 4.2 : Mean feature profile by risk class ──────────
num_vars = [
    'transaction_volume', 'account_age_years', 'nb_countries_involved',
    'cash_ratio', 'adverse_media_score', 'beneficial_owner_complexity'
]
df_raw = df.copy()
profile = df_raw.groupby('risk_label')[num_vars].mean()

# Normalise 0-1 per variable so all variables are on the same scale
profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
profile_norm.index = ['Low Risk', 'Medium Risk', 'High Risk']

fig42, ax42 = plt.subplots(figsize=(10, 5))
im = ax42.imshow(profile_norm.T.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

ax42.set_xticks(range(3))
ax42.set_xticklabels(['Low Risk', 'Medium Risk', 'High Risk'], fontsize=12)
ax42.set_yticks(range(len(num_vars)))
ax42.set_yticklabels([v.replace('_', ' ') for v in num_vars], fontsize=11)

for i in range(len(num_vars)):
    for j in range(3):
        raw_val = profile.iloc[j, i]
        label = f"{raw_val:.2f}" if raw_val < 100 else f"{raw_val:,.0f}"
        ax42.text(j, i, label, ha='center', va='center',
                  fontsize=9, fontweight='bold',
                  color='white' if profile_norm.iloc[j, i] > 0.6 else 'black')

cbar = plt.colorbar(im, ax=ax42, shrink=0.8, pad=0.02)
cbar.set_label('Normalised mean value (0 = min, 1 = max)', fontsize=9)
ax42.set_title('Figure 4.2: Mean feature profile by risk class (n = 5,000)',
               fontweight='bold', fontsize=13, pad=12)
ax42.spines[:].set_visible(False)
plt.tight_layout()
plt.savefig('data/figures/fig_4_2_feature_profile_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.2 saved → data/figures/fig_4_2_feature_profile_heatmap.png")

# ── Figure 4.3 : ROC curves one-vs-rest (tuned XGBoost) ──────
classes = [0, 1, 2]
class_names = ['Low Risk', 'Medium Risk', 'High Risk']
colors = [PALETTE['Low'], PALETTE['Medium'], PALETTE['High']]

y_test_bin = label_binarize(y_test, classes=classes)

fig43, ax43 = plt.subplots(figsize=(8, 6))
for i, (name, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_best[:, i])
    roc_auc = auc(fpr, tpr)
    ax43.plot(fpr, tpr, color=color, lw=2.5,
              label=f'{name}  (AUC = {roc_auc:.3f})')

ax43.plot([0, 1], [0, 1], color=GRAY, lw=1.5, linestyle='--', label='Random classifier')
ax43.set_xlabel('False Positive Rate', fontsize=12)
ax43.set_ylabel('True Positive Rate', fontsize=12)
ax43.set_title('Figure 4.3: ROC curves — tuned XGBoost (one-vs-rest)',
               fontweight='bold', fontsize=13, pad=12)
ax43.legend(loc='lower right', fontsize=11)
ax43.set_xlim([0, 1])
ax43.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig('data/figures/fig_4_3_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.3 saved → data/figures/fig_4_3_roc_curves.png")

# ── Figure 4.4 : Confusion matrices RF vs Tuned XGBoost ──────
fig44, axes44 = plt.subplots(1, 2, figsize=(13, 5))
labels = ['Low', 'Medium', 'High']

for ax, y_pred, title in zip(axes44, [y_pred_rf, y_pred_best], ['Random Forest', 'Tuned XGBoost']):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=11)
    ax.set_ylabel('True label', fontsize=11)

fig44.suptitle('Figure 4.4: Confusion matrices — Random Forest vs Tuned XGBoost',
               fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('data/figures/fig_4_4_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.4 saved → data/figures/fig_4_4_confusion_matrices.png")

# ── Figure 4.5 : Feature importance RF (MDI) vs XGBoost (SHAP) ──
rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

fig45, axes45 = plt.subplots(1, 2, figsize=(14, 6))

axes45[0].barh(range(len(rf_importance)), rf_importance.values,
               color=BLUE, alpha=0.85, edgecolor='white', linewidth=0.5)
axes45[0].set_yticks(range(len(rf_importance)))
axes45[0].set_yticklabels([v.replace('_', ' ') for v in rf_importance.index], fontsize=10)
axes45[0].set_title('Random Forest\nMean decrease in impurity', fontweight='bold', fontsize=11)
axes45[0].set_xlabel('Importance score')

shap_sorted = shap_imp_series.sort_values()
axes45[1].barh(range(len(shap_sorted)), shap_sorted.values,
               color=CORAL, alpha=0.85, edgecolor='white', linewidth=0.5)
axes45[1].set_yticks(range(len(shap_sorted)))
axes45[1].set_yticklabels([v.replace('_', ' ') for v in shap_sorted.index], fontsize=10)
axes45[1].set_title('XGBoost (Tuned)\nMean |SHAP| — High Risk class', fontweight='bold', fontsize=11)
axes45[1].set_xlabel('Mean absolute SHAP value')

fig45.suptitle('Figure 4.5: Feature importance comparison across model architectures',
               fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('data/figures/fig_4_5_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.5 saved → data/figures/fig_4_5_importance_comparison.png")

# ── Figure 4.7 : SHAP beeswarm ───────────────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_high_risk, X_test_shap, show=False, plot_size=None)
plt.title('Figure 4.7: SHAP summary plot — High Risk class (tuned XGBoost)',
          fontweight='bold', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig('data/figures/fig_4_7_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.7 saved → data/figures/fig_4_7_beeswarm.png")

# ── Figure 4.8 : SHAP dependence plot ────────────────────────
plt.figure(figsize=(9, 6))
shap.dependence_plot(
    'adverse_media_score', shap_high_risk, X_test_shap,
    interaction_index='nb_countries_involved', show=False
)
plt.title('Figure 4.8: SHAP dependence — adverse_media_score × nb_countries_involved',
          fontweight='bold', fontsize=12, pad=12)
plt.tight_layout()
plt.savefig('data/figures/fig_4_8_dependence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.8 saved → data/figures/fig_4_8_dependence.png")

# ── Figure 4.9 : SHAP waterfall — highest-confidence High Risk client in the test set (index 3390) ──
idx_high = 3390
X_client = X_test.loc[[idx_high]]
client_shap_values = explainer(X_client)

plt.figure(figsize=(10, 8))
try:
    shap.plots.waterfall(client_shap_values[0, :, 2], show=False)
except (IndexError, TypeError):
    shap_exp = shap.Explanation(
        values=shap_values[2][X_test.index.get_loc(idx_high)],
        base_values=explainer.expected_value[2],
        data=X_client.values[0],
        feature_names=list(X.columns)
    )
    shap.plots.waterfall(shap_exp, show=False)
plt.title('Figure 4.9: SHAP waterfall — High Risk client (tuned XGBoost)',
          fontweight='bold', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig('data/figures/fig_4_9_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.9 saved → data/figures/fig_4_9_waterfall.png")

# Scores bruts (log-odds)
raw_scores = xgb_best.predict(X_client, output_margin=True)
print(f"Raw scores — Low: {raw_scores[0][0]:.3f}, Medium: {raw_scores[0][1]:.3f}, High: {raw_scores[0][2]:.3f}")

# Probabilités softmax
proba = xgb_best.predict_proba(X_client)
print(f"Probabilities — Low: {proba[0][0]:.3f}, Medium: {proba[0][1]:.3f}, High: {proba[0][2]:.3f}")

print(f"predict: {xgb_best.predict(X_client)}")
print(f"argmax proba: {xgb_best.predict_proba(X_client).argmax()}")

proba_all = xgb_best.predict_proba(X_test)
y_pred_test = xgb_best.predict(X_test)
high_risk_proba = proba_all[:, 2]

# ── Figure 4.10 : Learning curve ─────────────────────────────
train_sizes, train_scores, test_scores = learning_curve(
    xgb_best, X, y, cv=5, scoring='f1_weighted',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
test_mean  = test_scores.mean(axis=1)
test_std   = test_scores.std(axis=1)

# Measured empirically in robustness.py:noise_ceiling() by reconstructing the
# deterministic part of the generating score and counting labels re-assigned
# by the injected noise (see thesis Section 3.5.6). Hardcoded here to keep
# this script's only dependency on generate_data.py's label logic explicit.
noise_ceiling = 0.86

fig410, ax410 = plt.subplots(figsize=(10, 6))
ax410.plot(train_sizes, train_mean, 'o-', color=BLUE, lw=2, label='Training F1')
ax410.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                   alpha=0.12, color=BLUE)
ax410.plot(train_sizes, test_mean, 's-', color=CORAL, lw=2, label='Validation F1 (5-fold)')
ax410.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                   alpha=0.12, color=CORAL)
ax410.axhline(y=noise_ceiling, color=GRAY, linestyle='--', lw=2,
              label=f'Noise ceiling ({noise_ceiling})')
ax410.set_xlabel('Training set size', fontsize=12)
ax410.set_ylabel('Weighted F1-score', fontsize=12)
ax410.set_title('Figure 4.10: Learning curve — tuned XGBoost vs noise ceiling',
                fontweight='bold', fontsize=13, pad=12)
ax410.legend(fontsize=11)
ax410.set_ylim([0.5, 1.05])
plt.tight_layout()
plt.savefig('data/figures/fig_4_10_learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4.10 saved → data/figures/fig_4_10_learning_curve.png")

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED → data/figures/")
print("=" * 60)
print("\nDone.")
