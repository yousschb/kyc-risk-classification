"""
robustness.py
=============
Reproduces every robustness result reported in the thesis (v23):

  - Noise ceiling on the synthetic labels            -> Section 3.3.4 / 4.3.4
  - Train/test gap of the tuned XGBoost              -> Section 4.3.4
  - Confidence intervals over repeated seeds         -> Section 4.3.4
  - Sensitivity to distribution shift                -> Section 4.3.4
  - Fairly re-tuned Random Forest                    -> Section 4.5 / Table 4.5
  - Interpretable benchmark: logistic regression     -> Section 4.5 / Table 4.5
  - Encoding effect: XGBoost with one-hot            -> Section 4.5 / Table 4.5
  - Local stability of SHAP explanations             -> Section 4.4.4

Run from the repository root:

    python src/robustness.py

Outputs:
    reports/robustness_results.json      (all numbers)


Note on reproducibility: absolute figures can move by a few tenths of a point
across library versions (xgboost / scikit-learn). The relative conclusions are
stable: the logistic regression reaches the noise ceiling, the re-tuned Random
Forest stays well behind XGBoost, and the model collapses under distribution
shift.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_ITER = 30        # randomised-search budget (same for XGBoost and Random Forest)
N_SEEDS = 8        # repeated splits for confidence intervals
TARGET = "risk_label"
DROP = ["client_id", "country_risk"]   # country_risk is redundant with country
CATEGORICAL = ["country", "sector"]

# Locate the dataset whether the script is run from the repo root or from src/.
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "..", "data", "kyc_synthetic.csv"),
    os.path.join(HERE, "data", "kyc_synthetic.csv"),
    "data/kyc_synthetic.csv",
]
DATA_PATH = next((p for p in CANDIDATES if os.path.exists(p)), CANDIDATES[0])
REPORTS = os.path.join(HERE, "..", "reports")
FIGDIR = os.path.join(REPORTS, "figures")

# ---------------------------------------------------------------------------
# Generating-rule constants.
# These MIRROR src/generate_data.py and must be kept in sync with it. They are
# needed only to compute the theoretical noise ceiling and the distribution-
# shift dataset; they are never used to train the production models.
# ---------------------------------------------------------------------------
COUNTRIES = [
    "Switzerland", "Germany", "France", "UK", "USA", "Luxembourg",
    "Netherlands", "Sweden", "Norway", "Denmark", "Austria", "Canada",
    "Australia", "New Zealand", "Japan",
    "Singapore", "China", "UAE", "Hong Kong", "Brazil", "India",
    "South Africa", "Mexico", "Turkey", "Saudi Arabia",
    "Russia", "Panama", "Cayman Islands", "British Virgin Islands",
    "Nigeria", "Afghanistan", "Iran", "North Korea", "Myanmar", "Venezuela",
]
COUNTRY_TIER = {c: ("Low" if i < 15 else "Medium" if i < 25 else "High")
                for i, c in enumerate(COUNTRIES)}
SECTORS = ["Real Estate", "Finance", "Technology", "Trading", "Legal",
           "Healthcare", "Construction", "Retail", "Energy", "Mining"]
SECTOR_RISK = {"Real Estate": 2, "Trading": 2, "Mining": 2, "Construction": 1,
               "Energy": 1, "Finance": 1, "Legal": 1, "Technology": 0,
               "Healthcare": 0, "Retail": 0}
TIER = {"Low": 0, "Medium": 1, "High": 2}
# Weights of the additive risk score (see src/generate_data.py).
W = {"country": 2.5, "pep": 3.0, "sector": 1.5, "media": 2.5}
# Percentile thresholds used to cut the score into Low / Medium / High.
LOW_Q, HIGH_Q = 40, 75


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load and encode the synthetic dataset exactly as train_model.py does."""
    df = pd.read_csv(DATA_PATH).drop(columns=DROP)
    for col in CATEGORICAL:
        df[col] = LabelEncoder().fit_transform(df[col])
    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = df[TARGET].values
    return X, y, features


def split(X, y, seed=SEED):
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def metrics(model, X_test, y_test, X_train=None, y_train=None):
    """Return a dict of the metrics reported in the thesis."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    out = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1_weighted": round(f1_score(y_test, pred, average="weighted"), 4),
        "auc_roc_ovr": round(roc_auc_score(y_test, proba, multi_class="ovr"), 4),
        "high_risk_recall": round(recall_score(y_test, pred, labels=[2],
                                                average="macro"), 4),
    }
    if X_train is not None:
        out["train_accuracy"] = round(accuracy_score(y_train, model.predict(X_train)), 4)
        out["train_test_gap"] = round(out["train_accuracy"] - out["accuracy"], 4)
    return out


# ---------------------------------------------------------------------------
# Reference models: tuned XGBoost and baseline Random Forest
# ---------------------------------------------------------------------------
def tuned_xgboost(X_train, y_train, cv):
    """Reproduce the tuned XGBoost via the same randomised search as train_model.py."""
    grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2, 0.5],
    }
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="mlogloss", random_state=SEED, n_jobs=-1),
        grid, n_iter=N_ITER, scoring="f1_weighted", cv=cv,
        random_state=SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def baseline_random_forest(X_train, y_train):
    """Reproduce the exact Random Forest configuration of train_model.py
    (Table 4.2), so that cross_model_spearman() is computed on the same
    model reported in the thesis rather than a differently-tuned variant."""
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                min_samples_split=5, min_samples_leaf=2,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

# ---------------------------------------------------------------------------
# Section 1 -- Noise ceiling (Section 3.3.4 / 4.3.4)
# ---------------------------------------------------------------------------
def noise_ceiling():
    """
    Recompute the deterministic part of the generating score from the raw data
    and measure how often the stochastic term re-assigns the label. An oracle
    that knows the deterministic score perfectly would still miss those cases;
    the resulting accuracy is the achievable ceiling.
    """
    raw = pd.read_csv(DATA_PATH)
    crn = raw["country_risk"].map(TIER).values
    srn = raw["sector"].map(SECTOR_RISK).values
    deterministic = (
        crn * W["country"] + raw["is_pep"].values * W["pep"]
        + srn * W["sector"] + np.log1p(raw["transaction_volume"].values) * 0.3
        + raw["nb_countries_involved"].values * 0.4 + raw["cash_ratio"].values * 2.0
        + raw["adverse_media_score"].values * W["media"]
        + raw["beneficial_owner_complexity"].values * 1.5
        + (1 - raw["source_of_wealth_verified"].values) * 2.0
        - raw["account_age_years"].values * 0.1
    )
    y = raw[TARGET].values
    # Thresholds are those of the realised (noisy) score, reconstructed here as
    # the empirical percentiles of the labels' underlying score.
    # We approximate them from the deterministic score's own percentiles, which
    # is sufficient to bound the ceiling.
    lo, hi = np.percentile(deterministic, LOW_Q), np.percentile(deterministic, HIGH_Q)
    y_oracle = np.where(deterministic < lo, 0, np.where(deterministic < hi, 1, 2))
    ceiling = (y_oracle == y).mean()
    return {
        "deterministic_score_std": round(float(deterministic.std()), 3),
        "labels_flipped_by_noise": round(float((y_oracle != y).mean()), 3),
        "accuracy_ceiling": round(float(ceiling), 3),
        "majority_baseline": round(float(pd.Series(y).value_counts(normalize=True).max()), 3),
    }


# ---------------------------------------------------------------------------
# Section 2 -- Fairly re-tuned Random Forest (Section 4.5 / Table 4.5)
# ---------------------------------------------------------------------------
def retuned_random_forest(X_train, y_train, X_test, y_test, cv):
    grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 8, 10, 12],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        grid, n_iter=N_ITER, scoring="f1_weighted", cv=cv,
        random_state=SEED, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    rf = search.best_estimator_
    res = metrics(rf, X_test, y_test, X_train, y_train)
    res["best_params"] = search.best_params_
    return rf, res


# ---------------------------------------------------------------------------
# Section 3 -- Interpretable benchmark and encoding effect (Section 4.5)
# ---------------------------------------------------------------------------
def logistic_regression(X_train, y_train, X_test, y_test, features):
    """Multinomial logistic regression with one-hot encoding of the nominal vars."""
    num = [c for c in features if c not in CATEGORICAL]
    pre = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("sc", StandardScaler(), num),
    ])
    lr = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, C=1.0))])
    lr.fit(X_train, y_train)
    return lr, metrics(lr, X_test, y_test, X_train, y_train)


def xgboost_one_hot(X_train, y_train, X_test, y_test, features, xgb_params):
    """Same tuned XGBoost hyper-parameters but with one-hot encoding."""
    num = [c for c in features if c not in CATEGORICAL]
    pre = ColumnTransformer([
        ("oh", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("pass", "passthrough", num),
    ])
    keep = ["n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "gamma"]
    clf = XGBClassifier(**{k: xgb_params[k] for k in keep if k in xgb_params},
                        eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
    model = Pipeline([("pre", pre), ("clf", clf)])
    model.fit(X_train, y_train)
    return model, metrics(model, X_test, y_test, X_train, y_train)


# ---------------------------------------------------------------------------
# Section 4 -- Confidence intervals and learning curve (Section 4.3.4)
# ---------------------------------------------------------------------------
def confidence_intervals(build_model, X, y, n=N_SEEDS):
    accs, aucs = [], []
    for s in range(n):
        xa, xb, ya, yb = train_test_split(X, y, test_size=0.2,
                                          random_state=s, stratify=y)
        m = build_model()
        m.fit(xa, ya)
        accs.append(accuracy_score(yb, m.predict(xb)))
        aucs.append(roc_auc_score(yb, m.predict_proba(xb), multi_class="ovr"))
    a, u = np.array(accs), np.array(aucs)
    return {
        "acc_mean": round(float(a.mean()), 4),
        "acc_ci95": round(float(1.96 * a.std() / np.sqrt(n)), 4),
        "auc_mean": round(float(u.mean()), 4),
        "auc_ci95": round(float(1.96 * u.std() / np.sqrt(n)), 4),
    }



# ---------------------------------------------------------------------------
# Section 5 -- Distribution shift (Section 4.3.4)
# ---------------------------------------------------------------------------
def _generate(seed, tier_map, weights, n=5000):
    """Parameterised copy of the generating process (mirrors generate_data.py)."""
    rng = np.random.RandomState(seed)
    country = rng.choice(COUNTRIES, n)
    crn = np.array([TIER[tier_map[c]] for c in country])
    is_pep = rng.choice([0, 1], n, p=[0.85, 0.15])
    sector = rng.choice(SECTORS, n)
    srn = np.array([SECTOR_RISK[s] for s in sector])
    tv = rng.lognormal(11, 1.5, n).round(2)
    age = rng.randint(1, 31, n)
    nt = rng.randint(1, 100, n)
    ata = (tv / (nt * 12)).round(2)
    nc = rng.randint(1, 15, n)
    cr = rng.beta(2, 5, n).round(3)
    ams = rng.choice([0, 1, 2, 3], n, p=[.70, .15, .10, .05])
    boc = rng.choice([0, 1, 2], n, p=[.60, .30, .10])
    sowv = rng.choice([0, 1], n, p=[.25, .75])
    score = (crn * weights["country"] + is_pep * weights["pep"]
             + srn * weights["sector"] + np.log1p(tv) * 0.3 + nc * 0.4
             + cr * 2.0 + ams * weights["media"] + boc * 1.5
             + (1 - sowv) * 2.0 - age * 0.1 + rng.normal(0, 1, n))
    lo, hi = np.percentile(score, LOW_Q), np.percentile(score, HIGH_Q)
    label = np.where(score < lo, 0, np.where(score < hi, 1, 2))
    return pd.DataFrame({
        "country": country, "is_pep": is_pep, "sector": sector,
        "transaction_volume": tv, "account_age_years": age,
        "nb_transactions_30d": nt, "avg_transaction_amount": ata,
        "nb_countries_involved": nc, "cash_ratio": cr,
        "adverse_media_score": ams, "beneficial_owner_complexity": boc,
        "source_of_wealth_verified": sowv, TARGET: label,
    })


def distribution_shift(model, features):
    """Apply the trained model to data drawn from a different risk regime."""
    lec = LabelEncoder().fit(COUNTRIES)
    les = LabelEncoder().fit(SECTORS)

    def prep(df):
        df = df.copy()
        df["country"] = lec.transform(df["country"])
        df["sector"] = les.transform(df["sector"])
        return df[features], df[TARGET].values

    # In-distribution reference (same rule, new seed).
    Xr, yr = prep(_generate(7, COUNTRY_TIER, W))
    acc_in = accuracy_score(yr, model.predict(Xr))
    auc_in = roc_auc_score(yr, model.predict_proba(Xr), multi_class="ovr")
    # Concept shift: rotate country tiers and re-balance the weights.
    shifted_map = {c: ("High" if COUNTRY_TIER[c] == "Low"
                        else "Low" if COUNTRY_TIER[c] == "High" else "Medium")
                   for c in COUNTRIES}
    shifted_w = {"country": 2.5, "pep": 1.5, "sector": 3.0, "media": 1.0}
    Xs, ys = prep(_generate(7, shifted_map, shifted_w))
    acc_shift = accuracy_score(ys, model.predict(Xs))
    auc_shift = roc_auc_score(ys, model.predict_proba(Xs), multi_class="ovr")
    return {
        "accuracy_in_distribution": round(float(acc_in), 4),
        "auc_in_distribution": round(float(auc_in), 4),
        "accuracy_concept_shift": round(float(acc_shift), 4),
        "auc_concept_shift": round(float(auc_shift), 4),
    }


# ---------------------------------------------------------------------------
# Section 4b -- Significance of the cost-sensitive gain (Section 4.3.2)
# ---------------------------------------------------------------------------
def mcnemar_high_recall(tuned_model, X_train, y_train, X_test, y_test):
    """
    Paired McNemar test on the detection of High-risk clients: does the
    cost-sensitive weighting improve High-risk recall beyond sampling noise?
    Both models share the tuned hyper-parameters; only the sample weights
    differ. The test uses the discordant pairs (clients caught by one model
    and missed by the other) on the true High-risk clients of the test set.
    """
    from scipy.stats import binomtest
    keep = ["n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "gamma"]
    params = {k: tuned_model.get_params()[k] for k in keep
              if k in tuned_model.get_params()}
    sw = np.ones(len(y_train))
    sw[y_train == 2] = 3.0
    sw[y_train == 1] = 1.5
    cs = XGBClassifier(**params, eval_metric="mlogloss", random_state=SEED, n_jobs=-1)
    cs.fit(X_train, y_train, sample_weight=sw)

    high = (y_test == 2)
    caught_tuned = tuned_model.predict(X_test)[high] == 2
    caught_cs = cs.predict(X_test)[high] == 2
    tuned_only = int((caught_tuned & ~caught_cs).sum())
    cs_only = int((~caught_tuned & caught_cs).sum())
    n = tuned_only + cs_only
    p = binomtest(min(tuned_only, cs_only), n, 0.5).pvalue if n > 0 else 1.0
    return {
        "recall_tuned": round(float(caught_tuned.mean()), 3),
        "recall_cost_sensitive": round(float(caught_cs.mean()), 3),
        "discordant_tuned_only": tuned_only,
        "discordant_cost_sensitive_only": cs_only,
        "mcnemar_p_value": float(f"{p:.3g}"),
        "significant_at_5pct": bool(p < 0.05),
    }


# ---------------------------------------------------------------------------
# Section 5b -- Cross-model agreement of SHAP importances (Section 4.4.2)
# ---------------------------------------------------------------------------
def cross_model_spearman(rf_model, xgb_model, X_eval):
    """
    Spearman rank correlation between the two models' global SHAP importances,
    measured on the High-risk class. For each model, the mean absolute SHAP
    value of every feature is computed over X_eval for class 2 (High), giving a
    twelve-element importance vector; the two vectors are then rank-correlated.
    This is the rho reported in the abstract, Section 4.4.2, Table 6.1 and the
    conclusion.
    """
    try:
        import shap
    except ImportError:
        return {"skipped": "shap not installed (pip install shap)"}
    from scipy.stats import spearmanr

    def high_class_importance(model):
        vals = shap.TreeExplainer(model).shap_values(X_eval)
        if isinstance(vals, list):          # Random Forest -> list per class
            arr = np.abs(vals[2])
        elif getattr(vals, "ndim", 2) == 3:  # XGBoost -> (n, features, classes)
            arr = np.abs(vals[:, :, 2])
        else:
            arr = np.abs(vals)
        return arr.mean(axis=0)

    rf_imp = high_class_importance(rf_model)
    xgb_imp = high_class_importance(xgb_model)
    rho, p = spearmanr(rf_imp, xgb_imp)
    return {
        "spearman_rho": round(float(rho), 3),
        "p_value": round(float(p), 4),
        "n_features": int(len(rf_imp)),
        "class": "High",
    }


# ---------------------------------------------------------------------------
# Section 6 -- Local stability of SHAP explanations (Section 4.4.4)
# ---------------------------------------------------------------------------
def local_shap_stability(model, X_test, y_test, features, n_clients=40, pct=0.05):
    try:
        import shap
    except ImportError:
        return {"skipped": "shap not installed (pip install shap)"}
    from scipy.stats import spearmanr

    explainer = shap.TreeExplainer(model)
    continuous = ["transaction_volume", "account_age_years", "nb_transactions_30d",
                  "avg_transaction_amount", "nb_countries_involved", "cash_ratio"]
    high = X_test[y_test == 2].head(n_clients).reset_index(drop=True)

    def shap_high(frame):
        vals = explainer.shap_values(frame)
        if isinstance(vals, list):
            vals = vals[2]
        elif getattr(vals, "ndim", 2) == 3:
            vals = vals[:, :, 2]
        return np.asarray(vals)

    base = shap_high(high)
    rng = np.random.default_rng(0)
    rhos, rel = [], []
    for _ in range(5):
        pert = high.copy()
        for c in continuous:
            pert[c] = pert[c] * (1 + rng.normal(0, pct, len(pert)))
        pv = shap_high(pert)
        for i in range(len(high)):
            rho = spearmanr(np.abs(base[i]), np.abs(pv[i])).correlation
            if not np.isnan(rho):
                rhos.append(rho)
            denom = np.abs(base[i]).sum()
            if denom > 0:
                rel.append(np.abs(pv[i] - base[i]).sum() / denom)
    return {
        "n_clients": n_clients,
        "perturbation": f"+/-{int(pct * 100)}% continuous",
        "rank_corr_mean": round(float(np.mean(rhos)), 3),
        "rank_corr_min": round(float(np.min(rhos)), 3),
        "rel_L1_change_mean": round(float(np.mean(rel)), 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data from:", os.path.relpath(DATA_PATH))
    X, y, features = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    print("Fitting reference models (tuned XGBoost, baseline Random Forest)...")
    xgb, xgb_params = tuned_xgboost(X_train, y_train, cv)
    rf_baseline = baseline_random_forest(X_train, y_train)  # metrics already in Table 4.2

    results = {}
    results["noise_ceiling"] = noise_ceiling()

    print("Computing cross-model SHAP agreement (Spearman rho)...")
    results["cross_model_spearman"] = cross_model_spearman(rf_baseline, xgb, X_test)

    print("Testing significance of the cost-sensitive gain (McNemar)...")
    results["cost_sensitive_mcnemar"] = mcnemar_high_recall(
        xgb, X_train, y_train, X_test, y_test)

    print("Re-tuning Random Forest with a comparable budget...")
    _, results["random_forest_retuned"] = retuned_random_forest(
        X_train, y_train, X_test, y_test, cv)

    print("Training interpretable benchmark and one-hot XGBoost...")
    _, results["logistic_regression"] = logistic_regression(
        X_train, y_train, X_test, y_test, features)
    _, results["xgboost_one_hot"] = xgboost_one_hot(
        X_train, y_train, X_test, y_test, features, xgb_params)

    results["xgboost_tuned_gap"] = metrics(xgb, X_test, y_test, X_train, y_train)

    print("Estimating confidence intervals over repeated seeds...")
    keep = ["n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "gamma"]
    results["ci_xgboost"] = confidence_intervals(
        lambda: XGBClassifier(**{k: xgb_params[k] for k in keep if k in xgb_params},
                              eval_metric="mlogloss", random_state=SEED, n_jobs=-1), X, y)

    print("Testing sensitivity to distribution shift...")
    results["distribution_shift"] = distribution_shift(xgb, features)

    print("Measuring local stability of SHAP explanations...")
    results["shap_local_stability"] = local_shap_stability(xgb, X_test, y_test, features)

    os.makedirs(REPORTS, exist_ok=True)
    out_path = os.path.join(REPORTS, "robustness_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 64)
    print("ROBUSTNESS SUMMARY (thesis v23)")
    print("=" * 64)
    nc = results["noise_ceiling"]
    print(f"Noise ceiling            : {nc['accuracy_ceiling']}  "
          f"(labels flipped {nc['labels_flipped_by_noise']}, baseline {nc['majority_baseline']})")
    lr = results["logistic_regression"]
    print(f"Logistic (one-hot)       : acc {lr['accuracy']}  AUC {lr['auc_roc_ovr']}  "
          f"gap {lr['train_test_gap']}")
    xo = results["xgboost_one_hot"]
    print(f"XGBoost (one-hot)        : acc {xo['accuracy']}  AUC {xo['auc_roc_ovr']}  "
          f"gap {xo['train_test_gap']}")
    xt = results["xgboost_tuned_gap"]
    print(f"XGBoost (tuned, label)   : acc {xt['accuracy']}  AUC {xt['auc_roc_ovr']}  "
          f"gap {xt['train_test_gap']}")
    rf = results["random_forest_retuned"]
    print(f"Random Forest (re-tuned) : acc {rf['accuracy']}  AUC {rf['auc_roc_ovr']}  "
          f"gap {rf['train_test_gap']}")
    ci = results["ci_xgboost"]
    print(f"XGBoost CI (8 seeds)     : acc {ci['acc_mean']} +/- {ci['acc_ci95']}  "
          f"AUC {ci['auc_mean']} +/- {ci['auc_ci95']}")
    ds = results["distribution_shift"]
    print(f"Distribution shift (acc) : in-dist {ds['accuracy_in_distribution']} -> "
          f"shifted {ds['accuracy_concept_shift']}")
    print(f"Distribution shift (AUC) : in-dist {ds['auc_in_distribution']} -> "
          f"shifted {ds['auc_concept_shift']}")
    mc = results["cost_sensitive_mcnemar"]
    print(f"Cost-sensitive McNemar   : p {mc['mcnemar_p_value']} "
          f"(discordant {mc['discordant_cost_sensitive_only']} vs "
          f"{mc['discordant_tuned_only']}) -> "
          f"{'significant' if mc['significant_at_5pct'] else 'not significant'}")
    cm = results["cross_model_spearman"]
    if "spearman_rho" in cm:
        print(f"Cross-model SHAP rho     : {cm['spearman_rho']} "
              f"(p {cm['p_value']}, High class, {cm['n_features']} features)")
    ss = results["shap_local_stability"]
    if "rank_corr_mean" in ss:
        print(f"SHAP local stability     : rank corr {ss['rank_corr_mean']} "
              f"(min {ss['rank_corr_min']}), L1 change {ss['rel_L1_change_mean']}")
    print("=" * 64)
    print(f"Saved: {os.path.relpath(out_path)}")
 


if __name__ == "__main__":
    main()
