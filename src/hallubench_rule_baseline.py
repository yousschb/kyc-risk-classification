"""
hallubench_rule_baseline.py
============================
Deterministic rule baseline for the three-class BankKYC-HalluBench setup
(Section 4.6.1). Assigns High Risk to any client flagged as PEP or
sanctioned, Medium Risk to any remaining client carrying an AML flag, and
Low Risk to the rest. This is the most favourable separation the available
variables allow, and is used to confirm that the confusion between Low and
Medium Risk observed with the trained XGBoost model (hallubench_train.py)
is a property of the data rather than of the classifier.

Run from the repository root:
    python src/hallubench_rule_baseline.py

Evaluated on the full 100,000-row dataset (no train/test split, since the
rule has no parameters to fit).
"""

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv('data/kyc_cases_external.csv')
print(f"HalluBench dataset: {df.shape}")

def rule(row):
    if row['PEPStatus'] == 'Yes' or row['SanctionStatus'] == 'Yes':
        return 'High'
    elif row['AMLFlag'] == 'Yes':
        return 'Medium'
    else:
        return 'Low'

df['pred_rule'] = df.apply(rule, axis=1)

label_map = {'Low': 0, 'Medium': 1, 'High': 2}
y_true = df['RiskCategory'].map(label_map)
y_pred = df['pred_rule'].map(label_map)

print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
print(classification_report(y_true, y_pred, target_names=['Low', 'Medium', 'High']))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
