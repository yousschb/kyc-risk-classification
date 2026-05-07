import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 5)

# =============================================
# 1. LOAD DATA
# =============================================
print("="*60)
print("STEP 1 — Loading data")
print("="*60)

df = pd.read_csv('data/kyc_synthetic.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

# =============================================
# 2. DESCRIPTIVE STATISTICS
# =============================================
print("\n" + "="*60)
print("STEP 2 — Descriptive Statistics")
print("="*60)
print(df.describe().round(2))
print("\nMissing values:")
print(df.isnull().sum())

# =============================================
# 3. TARGET VARIABLE DISTRIBUTION
# =============================================
print("\n" + "="*60)
print("STEP 3 — Risk Label Distribution")
print("="*60)

label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
df['risk_label_str'] = df['risk_label'].map(label_map)
colors = ['#2ecc71', '#f39c12', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts = df['risk_label_str'].value_counts().reindex(['Low', 'Medium', 'High'])

axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Risk Label Distribution — Counts', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of clients')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

axes[1].pie(counts.values, labels=counts.index, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Risk Label Distribution — Proportions', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('data/figures/eda_label_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_label_distribution.png")
print(counts)

# =============================================
# 4. CATEGORICAL VARIABLES
# =============================================
print("\n" + "="*60)
print("STEP 4 — Categorical Variables")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

country_risk = df.groupby(['country_risk', 'risk_label_str']).size().unstack(fill_value=0)
country_risk = country_risk.reindex(columns=['Low', 'Medium', 'High'])
country_risk.plot(kind='bar', ax=axes[0], color=colors, edgecolor='white')
axes[0].set_title('Country Risk vs Client Risk Label', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Country Risk Category')
axes[0].set_ylabel('Number of clients')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='Risk Label')

pep = df.groupby(['is_pep', 'risk_label_str']).size().unstack(fill_value=0)
pep = pep.reindex(columns=['Low', 'Medium', 'High'])
pep.index = ['Non-PEP', 'PEP']
pep.plot(kind='bar', ax=axes[1], color=colors, edgecolor='white')
axes[1].set_title('PEP Status vs Client Risk Label', fontsize=13, fontweight='bold')
axes[1].set_xlabel('PEP Status')
axes[1].set_ylabel('Number of clients')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='Risk Label')

plt.tight_layout()
plt.savefig('data/figures/eda_categorical.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_categorical.png")

# Sector
fig, ax = plt.subplots(figsize=(14, 6))
sector_risk = df.groupby(['sector', 'risk_label_str']).size().unstack(fill_value=0)
sector_risk = sector_risk.reindex(columns=['Low', 'Medium', 'High'])
sector_risk.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
ax.set_title('Business Sector vs Client Risk Label', fontsize=13, fontweight='bold')
ax.set_xlabel('Sector')
ax.set_ylabel('Number of clients')
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Risk Label')
plt.tight_layout()
plt.savefig('data/figures/eda_sector.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_sector.png")

# =============================================
# 5. NUMERICAL DISTRIBUTIONS
# =============================================
print("\n" + "="*60)
print("STEP 5 — Numerical Distributions")
print("="*60)

numerical_vars = ['transaction_volume', 'account_age_years', 'nb_transactions_30d',
                  'avg_transaction_amount', 'nb_countries_involved', 'cash_ratio']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, var in enumerate(numerical_vars):
    for label, color in zip(['Low', 'Medium', 'High'], colors):
        subset = df[df['risk_label_str'] == label][var]
        axes[i].hist(subset, bins=30, alpha=0.5, color=color, label=label, edgecolor='none')
    axes[i].set_title(var, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')
    axes[i].legend()

plt.suptitle('Numerical Variable Distributions by Risk Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/eda_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_distributions.png")

# =============================================
# 6. CORRELATION MATRIX
# =============================================
print("\n" + "="*60)
print("STEP 6 — Correlation Matrix")
print("="*60)

numerical_cols = ['is_pep', 'transaction_volume', 'account_age_years', 'nb_transactions_30d',
                  'avg_transaction_amount', 'nb_countries_involved', 'cash_ratio',
                  'adverse_media_score', 'beneficial_owner_complexity',
                  'source_of_wealth_verified', 'risk_label']

corr = df[numerical_cols].corr().round(2)

plt.figure(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix — KYC Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/eda_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_correlation.png")
print("\nTop correlations with risk_label:")
print(corr['risk_label'].drop('risk_label').sort_values(ascending=False))

# =============================================
# 7. BOXPLOTS
# =============================================
print("\n" + "="*60)
print("STEP 7 — Boxplots")
print("="*60)

key_vars = ['adverse_media_score', 'nb_countries_involved', 'cash_ratio', 'beneficial_owner_complexity']

fig, axes = plt.subplots(1, 4, figsize=(18, 6))

for i, var in enumerate(key_vars):
    data_by_class = [df[df['risk_label_str'] == label][var].values
                     for label in ['Low', 'Medium', 'High']]
    bp = axes[i].boxplot(data_by_class, tick_labels=['Low', 'Medium', 'High'],
                         patch_artist=True, medianprops={'color': 'black', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[i].set_title(var, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Risk Class')

plt.suptitle('Key Variables Distribution by Risk Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/figures/eda_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_boxplots.png")

print("\n" + "="*60)
print("EDA COMPLETE — 6 figures saved in data/figures/")
print("="*60)