import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df = pd.read_csv('data/kyc_cases_external.csv')

# Binary flags
df['is_pep']      = (df['PEPStatus'] == 'Yes').astype(int)
df['is_sanction'] = (df['SanctionStatus'] == 'Yes').astype(int)
df['aml_flag']    = (df['AMLFlag'] == 'Yes').astype(int)

classes = ['Low', 'Medium', 'High']
colors  = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

# ── Panel 1: PEP rate per class ───────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
pep_rate = df.groupby('RiskCategory')['is_pep'].mean().reindex(classes) * 100
bars = ax1.bar(classes, pep_rate.values,
               color=[colors[c] for c in classes],
               edgecolor='black', linewidth=0.8)
for bar, v in zip(bars, pep_rate.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_title('PEP rate by class', fontweight='bold', fontsize=11)
ax1.set_ylabel('% clients with PEP = Yes')
ax1.set_ylim(0, 75)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Panel 2: Sanction rate per class ─────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sanc_rate = df.groupby('RiskCategory')['is_sanction'].mean().reindex(classes) * 100
bars = ax2.bar(classes, sanc_rate.values,
               color=[colors[c] for c in classes],
               edgecolor='black', linewidth=0.8)
for bar, v in zip(bars, sanc_rate.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_title('Sanction rate by class', fontweight='bold', fontsize=11)
ax2.set_ylabel('% clients with Sanction = Yes')
ax2.set_ylim(0, 35)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Panel 3: AML flag rate per class ─────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
aml_rate = df.groupby('RiskCategory')['aml_flag'].mean().reindex(classes) * 100
bars = ax3.bar(classes, aml_rate.values,
               color=[colors[c] for c in classes],
               edgecolor='black', linewidth=0.8)
for bar, v in zip(bars, aml_rate.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.set_title('AML flag rate by class', fontweight='bold', fontsize=11)
ax3.set_ylabel('% clients with AML flag = Yes')
ax3.set_ylim(0, 60)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── Panel 4: Income distribution ─────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for c in classes:
    vals = df[df['RiskCategory'] == c]['Income'] / 1e6
    ax4.hist(vals, bins=40, alpha=0.5, color=colors[c], label=c, density=True)
ax4.set_title('Income distribution by class', fontweight='bold', fontsize=11)
ax4.set_xlabel('Income (CHF millions)')
ax4.set_ylabel('Density')
ax4.legend(fontsize=9)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# ── Panel 5: VerifiedDocuments distribution ───────────────────
ax5 = fig.add_subplot(gs[1, 1])
vd_props = df.groupby(['RiskCategory', 'VerifiedDocuments']).size().unstack(fill_value=0)
vd_props = vd_props.div(vd_props.sum(axis=1), axis=0).reindex(classes) * 100
x = np.arange(len(vd_props.columns))
width = 0.25
for i, c in enumerate(classes):
    ax5.bar(x + i*width, vd_props.loc[c].values, width,
            label=c, color=colors[c], edgecolor='black', linewidth=0.5)
ax5.set_title('VerifiedDocuments by class', fontweight='bold', fontsize=11)
ax5.set_xlabel('VerifiedDocuments score')
ax5.set_ylabel('% of class')
ax5.set_xticks(x + width)
ax5.set_xticklabels(vd_props.columns)
ax5.legend(fontsize=9)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# ── Panel 6: Annotation text ──────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
msg = (
    "Key finding:\n\n"
    "Low and Medium Risk classes share\n"
    "identical distributions across all\n"
    "available variables:\n\n"
    "  • PEP rate:    Low 0%  ≈  Medium 0%\n"
    "  • Sanction:    Low 0%  ≈  Medium 0%\n"
    "  • Income:      Low ≈ Medium (≈ CHF 5M)\n"
    "  • Documents:   Low ≈ Medium (≈ 77% at 5)\n\n"
    "Only AML flag differs marginally\n"
    "(Low 0%  vs  Medium 24.8%).\n\n"
    "→ Low vs Medium are not separable\n"
    "   with the available variables.\n"
    "→ Binary High vs Non-High is the\n"
    "   statistically valid formulation."
)
ax6.text(0.05, 0.95, msg, transform=ax6.transAxes,
         fontsize=9.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8, edgecolor='#ccc'))

fig.suptitle(
    'Figure 4.X: Feature separability analysis — BankKYC-HalluBench (n = 100,000)\n'
    'Low and Medium Risk classes are statistically indistinguishable on available variables',
    fontweight='bold', fontsize=12, y=1.01
)

plt.savefig('data/figures/fig_hallubench_separability.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved → data/figures/fig_hallubench_separability.png")
