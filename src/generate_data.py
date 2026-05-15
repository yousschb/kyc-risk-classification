import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000

# --- Variables de base ---
countries = [
    # Low risk
    'Switzerland', 'Germany', 'France', 'UK', 'USA', 'Luxembourg',
    'Netherlands', 'Sweden', 'Norway', 'Denmark', 'Austria', 'Canada',
    'Australia', 'New Zealand', 'Japan',
    # Medium risk
    'Singapore', 'China', 'UAE', 'Hong Kong', 'Brazil', 'India',
    'South Africa', 'Mexico', 'Turkey', 'Saudi Arabia',
    # High risk
    'Russia', 'Panama', 'Cayman Islands', 'British Virgin Islands',
    'Nigeria', 'Afghanistan', 'Iran', 'North Korea', 'Myanmar', 'Venezuela'
]

country_risk_map = {
    'Switzerland': 'Low', 'Germany': 'Low', 'France': 'Low', 'UK': 'Low',
    'USA': 'Low', 'Luxembourg': 'Low', 'Netherlands': 'Low', 'Sweden': 'Low',
    'Norway': 'Low', 'Denmark': 'Low', 'Austria': 'Low', 'Canada': 'Low',
    'Australia': 'Low', 'New Zealand': 'Low', 'Japan': 'Low',
    'Singapore': 'Medium', 'China': 'Medium', 'UAE': 'Medium', 'Hong Kong': 'Medium',
    'Brazil': 'Medium', 'India': 'Medium', 'South Africa': 'Medium',
    'Mexico': 'Medium', 'Turkey': 'Medium', 'Saudi Arabia': 'Medium',
    'Russia': 'High', 'Panama': 'High', 'Cayman Islands': 'High',
    'British Virgin Islands': 'High', 'Nigeria': 'High', 'Afghanistan': 'High',
    'Iran': 'High', 'North Korea': 'High', 'Myanmar': 'High', 'Venezuela': 'High'
}

country_risk_score = {'Low': 0, 'Medium': 1, 'High': 2}

sectors = ['Real Estate', 'Finance', 'Technology', 'Trading', 'Legal',
           'Healthcare', 'Construction', 'Retail', 'Energy', 'Mining']
sector_risk_map = {
    'Real Estate': 2, 'Trading': 2, 'Mining': 2, 'Construction': 1,
    'Energy': 1, 'Finance': 1, 'Legal': 1,
    'Technology': 0, 'Healthcare': 0, 'Retail': 0
}

# --- Generation ---
country = np.random.choice(countries, N)
country_risk = [country_risk_map[c] for c in country]
country_risk_num = np.array([country_risk_score[r] for r in country_risk])

is_pep = np.random.choice([0, 1], N, p=[0.85, 0.15])
sector = np.random.choice(sectors, N)
sector_risk_num = np.array([sector_risk_map[s] for s in sector])

transaction_volume = np.random.lognormal(mean=11, sigma=1.5, size=N).round(2)
account_age_years = np.random.randint(1, 31, N)
nb_transactions_30d = np.random.randint(1, 100, N)
avg_transaction_amount = (transaction_volume / (nb_transactions_30d * 12)).round(2)
nb_countries_involved = np.random.randint(1, 15, N)
cash_ratio = np.random.beta(2, 5, N).round(3)
adverse_media_score = np.random.choice([0, 1, 2, 3], N, p=[0.70, 0.15, 0.10, 0.05])
beneficial_owner_complexity = np.random.choice([0, 1, 2], N, p=[0.60, 0.30, 0.10])
source_of_wealth_verified = np.random.choice([0, 1], N, p=[0.25, 0.75])

# --- Risk score ---
risk_score = (
    country_risk_num * 2.5 +
    is_pep * 3.0 +
    sector_risk_num * 1.5 +
    np.log1p(transaction_volume) * 0.3 +
    nb_countries_involved * 0.4 +
    cash_ratio * 2.0 +
    adverse_media_score * 2.5 +
    beneficial_owner_complexity * 1.5 +
    (1 - source_of_wealth_verified) * 2.0 -
    account_age_years * 0.1 +
    np.random.normal(0, 1, N)
)

low = np.percentile(risk_score, 40)
high = np.percentile(risk_score, 75)

def assign_label(score):
    if score < low:
        return 0
    elif score < high:
        return 1
    else:
        return 2

risk_label = [assign_label(s) for s in risk_score]

df = pd.DataFrame({
    'client_id': [f'CLT{str(i).zfill(4)}' for i in range(1, N+1)],
    'country': country,
    'country_risk': country_risk,
    'is_pep': is_pep,
    'sector': sector,
    'transaction_volume': transaction_volume,
    'account_age_years': account_age_years,
    'nb_transactions_30d': nb_transactions_30d,
    'avg_transaction_amount': avg_transaction_amount,
    'nb_countries_involved': nb_countries_involved,
    'cash_ratio': cash_ratio,
    'adverse_media_score': adverse_media_score,
    'beneficial_owner_complexity': beneficial_owner_complexity,
    'source_of_wealth_verified': source_of_wealth_verified,
    'risk_label': risk_label
})

df.to_csv('data/kyc_synthetic.csv', index=False)
print(f"Dataset generated: {len(df)} clients")
print(f"Countries: {df['country'].nunique()}")
print(f"\nLabel distribution:")
print(df['risk_label'].value_counts().sort_index().rename({0: 'Low', 1: 'Medium', 2: 'High'}))
print(f"\nPreview:")
print(df.head())