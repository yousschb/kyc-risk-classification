import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

# =============================================
# 1. LOAD MODELS & DATA
# =============================================
print("="*60)
print("STEP 1 — Loading models and data")
print("="*60)

rf  = joblib.load('models/random_forest.pkl')
xgb = joblib.load('models/xgboost.pkl')
le_country = joblib.load('models/le_country.pkl')
le_sector  = joblib.load('models/le_sector.pkl')
feature_names = joblib.load('models/feature_names.pkl')

df = pd.read_csv('data/kyc_synthetic.csv')
df_model = df.drop(columns=['client_id', 'country_risk'])
df_model['country'] = le_country.transform(df_model['country'])
df_model['sector']  = le_sector.transform(df_model['sector'])

X = df_model.drop(columns=['risk_label'])
y = df_model['risk_label']

os.makedirs('data/figures', exist_ok=True)
print(f"Data loaded: {X.shape[0]} clients, {X.shape[1]} features")

# =============================================
# 2. SHAP — RANDOM FOREST
# =============================================
print("\n" + "="*60)
print("STEP 2 — SHAP Analysis: Random Forest")
print("="*60)

explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X)

# Global summary plot — RF
plt.figure()
shap.summary_plot(
    shap_values_rf,
    X,
    class_names=['Low Risk', 'Medium Risk', 'High Risk'],
    show=False
)
plt.title('Random Forest — SHAP Summary Plot (Global)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/shap_rf_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: data/figures/shap_rf_summary.png")

# Feature importance bar — RF (High Risk class = index 2)
# shap_values_rf is a list of arrays [n_samples, n_features] per class
shap_rf_high = shap_values_rf[2] if isinstance(shap_values_rf, list) else shap_values_rf[:, :, 2]
plt.figure()
shap.summary_plot(
    shap_rf_high,
    X,
    plot_type='bar',
    show=False
)
plt.title('Random Forest — Feature Importance for High Risk (SHAP)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/shap_rf_bar_high.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: data/figures/shap_rf_bar_high.png")

# =============================================
# 3. SHAP — XGBOOST
# =============================================
print("\n" + "="*60)
print("STEP 3 — SHAP Analysis: XGBoost")
print("="*60)

explainer_xgb = shap.TreeExplainer(xgb)
shap_values_xgb = explainer_xgb.shap_values(X)

# Normalize shape — XGBoost may return [n_samples, n_features, n_classes] or list
if isinstance(shap_values_xgb, list):
    shap_xgb_high = shap_values_xgb[2]
else:
    shap_xgb_high = shap_values_xgb[:, :, 2]

# Global summary plot — XGBoost (High Risk class)
plt.figure()
shap.summary_plot(
    shap_xgb_high,
    X,
    show=False
)
plt.title('XGBoost — SHAP Summary Plot: High Risk class', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/shap_xgb_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: data/figures/shap_xgb_summary.png")

# Feature importance bar — XGBoost (High Risk)
plt.figure()
shap.summary_plot(
    shap_xgb_high,
    X,
    plot_type='bar',
    show=False
)
plt.title('XGBoost — Feature Importance for High Risk (SHAP)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/shap_xgb_bar_high.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: data/figures/shap_xgb_bar_high.png")

# =============================================
# 4. LOCAL EXPLANATION — WATERFALL (single client)
# =============================================
print("\n" + "="*60)
print("STEP 4 — Local Explanation: Waterfall for a High Risk client")
print("="*60)

# Pick a High Risk client
high_risk_idx = df[df['risk_label'] == 2].index[0]
client = X.iloc[[high_risk_idx]]
client_raw = df.iloc[high_risk_idx]

print(f"\nClient: {client_raw['client_id']}")
print(f"  Country:      {client_raw['country']}")
print(f"  PEP:          {client_raw['is_pep']}")
print(f"  Sector:       {client_raw['sector']}")
print(f"  Risk Label:   {client_raw['risk_label']} (High)")

# XGBoost waterfall — High Risk class
explainer_xgb2 = shap.TreeExplainer(xgb)
shap_explanation = explainer_xgb2(client)

# Handle both 2D and 3D SHAP output
if len(shap_explanation.values.shape) == 3:
    waterfall_values = shap.Explanation(
        values=shap_explanation.values[0, :, 2],
        base_values=shap_explanation.base_values[0, 2],
        data=shap_explanation.data[0],
        feature_names=feature_names
    )
else:
    waterfall_values = shap_explanation[0]

plt.figure()
shap.plots.waterfall(waterfall_values, show=False)
plt.title(f'XGBoost — Waterfall: {client_raw["client_id"]} (High Risk)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/shap_waterfall_example.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: data/figures/shap_waterfall_example.png")

# =============================================
# 5. DEPENDENCE PLOT — Top feature
# =============================================
print("\n" + "="*60)
print("STEP 5 — Dependence Plot: adverse_media_score")
print("="*60)

plt.figure()
shap.dependence_plot(
    'adverse_media_score',
    shap_xgb_high,
    X,
    show=False
)
plt.title('XGBoost — SHAP Dependence: adverse_media_score → High Risk', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/shap_dependence_media.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: data/figures/shap_dependence_media.png")

# =============================================
# 6. SAVE EXPLAINERS
# =============================================
joblib.dump(explainer_rf,  'models/explainer_rf.pkl')
joblib.dump(explainer_xgb, 'models/explainer_xgb.pkl')

print("\n" + "="*60)
print("SHAP ANALYSIS COMPLETE")
print("Figures saved in: data/figures/")
print("Explainers saved in: models/")
print("="*60)