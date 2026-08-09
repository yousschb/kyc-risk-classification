# kyc-risk-classification

Explainable client risk classification for private banking, developed as part of an MScIS master's thesis at HEC Lausanne: *Explainability and Regulatory Compliance in Private Banking: Designing and Evaluating a Client Risk Classification System*.

The project trains machine learning models to classify clients into Low, Medium and High money-laundering risk, and uses SHAP to explain every decision in a form a compliance officer can read. All work is performed on a synthetic dataset; no confidential client data is used.

**Live prototype:** https://kyc-risk-classification.streamlit.app

## What it does

- Classifies clients into three risk tiers with Random Forest and XGBoost models
- Explains each prediction with SHAP (global importance and per-client attributions)
- Produces a regulatory PDF report with ranked SHAP contributions and a model hash for traceability
- Ships a Streamlit app with an individual mode and a portfolio mode

## Repository structure

    src/generate_data.py     Synthetic dataset generation (5,000 profiles, seed 42)
    src/train_model.py       Random Forest and XGBoost training, tuning, cost-sensitive variant
    src/explain.py           SHAP analysis, global and local explanations
    src/robustness.py        Diagnostics: noise ceiling, generalisation, distribution shift,
                             interpretable benchmark, cross-model agreement, SHAP stability
    app.py                   Streamlit prototype (individual and portfolio modes, PDF export)
    data/kyc_synthetic.csv   Generated dataset (5,000 rows)
    models/                  Trained models and encoders
    reports/                 robustness_results.json and figures

## Requirements

    Python 3.9
    pip install -r requirements.txt

Main libraries: scikit-learn, xgboost, shap, streamlit, pandas, numpy, matplotlib, reportlab, scipy.

## Reproduce the results

    python src/generate_data.py     # writes data/kyc_synthetic.csv
    python src/train_model.py       # trains and tunes the models
    python src/explain.py           # SHAP analyses
    python src/robustness.py        # robustness diagnostics and Figure 4.10

Run the app locally:

    streamlit run app.py

## Note on reproducibility

Figures may vary by up to about two points across platforms and library versions, because the randomised hyperparameter search can select slightly different settings when cross-validation scores shift by small amounts. The relative conclusions are unaffected.

## Author

Youssouf Chaib — MScIS, HEC Lausanne. Internship at Banque Audi Suisse SA, Business Risk Management, Geneva.
