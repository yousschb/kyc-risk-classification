import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="KYC Risk Classification",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CUSTOM CSS
# =============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #e8eaed;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2d3142;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Risk badges */
    .risk-high {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 16px 32px;
        border-radius: 8px;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: 0 4px 20px rgba(231,76,60,0.4);
    }
    .risk-medium {
        background: linear-gradient(135deg, #d35400, #e67e22);
        color: white;
        padding: 16px 32px;
        border-radius: 8px;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: 0 4px 20px rgba(230,126,34,0.4);
    }
    .risk-low {
        background: linear-gradient(135deg, #1e8449, #27ae60);
        color: white;
        padding: 16px 32px;
        border-radius: 8px;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 2px;
        box-shadow: 0 4px 20px rgba(39,174,96,0.4);
    }

    /* Regulatory box */
    .reg-box {
        background-color: #1a1d27;
        border-left: 4px solid #3498db;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin-top: 16px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #a8b2c1;
    }

    /* Metric cards */
    .metric-card {
        background-color: #1a1d27;
        border: 1px solid #2d3142;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }

    /* Section divider */
    .section-title {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6b7a99;
        margin-bottom: 12px;
        margin-top: 24px;
    }

    /* Selectbox and inputs */
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: #1a1d27;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #2980b9, #3498db);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 12px 32px;
        font-weight: 600;
        font-family: 'IBM Plex Sans', sans-serif;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(52,152,219,0.4);
    }

    /* Dataframe */
    .stDataFrame {
        border: 1px solid #2d3142;
        border-radius: 8px;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================
# LOAD MODELS (CACHED)
# =============================================
@st.cache_resource
def load_models():
    rf = joblib.load('models/random_forest.pkl')
    xgb = joblib.load('models/xgboost.pkl')
    le_country = joblib.load('models/le_country.pkl')
    le_sector = joblib.load('models/le_sector.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    explainer_rf = joblib.load('models/explainer_rf.pkl')
    explainer_xgb = joblib.load('models/explainer_xgb.pkl')
    return rf, xgb, le_country, le_sector, feature_names, explainer_rf, explainer_xgb

rf, xgb, le_country, le_sector, feature_names, explainer_rf, explainer_xgb = load_models()

COUNTRIES = ['Switzerland', 'Germany', 'France', 'UAE', 'Panama', 'Cayman Islands',
             'Singapore', 'UK', 'Russia', 'China', 'USA', 'Luxembourg']
SECTORS = ['Real Estate', 'Finance', 'Technology', 'Trading', 'Legal',
           'Healthcare', 'Construction', 'Retail', 'Energy', 'Mining']
LABEL_MAP = {0: 'LOW RISK', 1: 'MEDIUM RISK', 2: 'HIGH RISK'}
COLOR_MAP = {0: 'risk-low', 1: 'risk-medium', 2: 'risk-high'}

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.markdown("## 🏦 KYC Risk System")
    st.markdown("---")

    mode = st.radio(
        "Interface",
        ["Individual Analysis", "Portfolio Analysis"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    model_choice = st.selectbox(
        "Model",
        ["XGBoost", "Random Forest"],
        help="XGBoost performs better (AUC 0.87 vs 0.82)"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#6b7a99; line-height:1.6;'>
    <b>Model Performance</b><br>
    XGBoost — Acc: 71% | AUC: 0.87<br>
    Random Forest — Acc: 61% | AUC: 0.82<br><br>
    <b>Methodology</b><br>
    Design Science Research<br>
    Peffers et al. (2007)<br><br>
    <b>Regulatory Framework</b><br>
    FINMA Circular 2016/7<br>
    AMLA (LBA) Art. 3–6
    </div>
    """, unsafe_allow_html=True)

# =============================================
# HELPER FUNCTIONS
# =============================================
def prepare_input(country, is_pep, sector, transaction_volume, account_age_years,
                  nb_transactions_30d, nb_countries_involved, cash_ratio,
                  adverse_media_score, beneficial_owner_complexity, source_of_wealth_verified):
    avg_transaction_amount = transaction_volume / (nb_transactions_30d * 12) if nb_transactions_30d > 0 else 0
    country_enc = le_country.transform([country])[0]
    sector_enc = le_sector.transform([sector])[0]
    data = {
        'country': country_enc,
        'is_pep': is_pep,
        'sector': sector_enc,
        'transaction_volume': transaction_volume,
        'account_age_years': account_age_years,
        'nb_transactions_30d': nb_transactions_30d,
        'avg_transaction_amount': avg_transaction_amount,
        'nb_countries_involved': nb_countries_involved,
        'cash_ratio': cash_ratio,
        'adverse_media_score': adverse_media_score,
        'beneficial_owner_complexity': beneficial_owner_complexity,
        'source_of_wealth_verified': source_of_wealth_verified
    }
    return pd.DataFrame([data])[feature_names]

def get_regulatory_explanation(label, shap_vals, feature_names, client_data):
    top_features = pd.Series(shap_vals, index=feature_names).abs().sort_values(ascending=False).head(3)
    feature_labels = {
        'adverse_media_score': 'adverse media exposure',
        'is_pep': 'PEP status',
        'nb_countries_involved': 'multi-jurisdiction exposure',
        'cash_ratio': 'cash transaction ratio',
        'beneficial_owner_complexity': 'beneficial ownership complexity',
        'country': 'country of domicile risk',
        'account_age_years': 'account tenure',
        'source_of_wealth_verified': 'source of wealth verification',
        'transaction_volume': 'transaction volume',
        'sector': 'business sector risk'
    }
    top_str = ', '.join([feature_labels.get(f, f) for f in top_features.index])
    risk_labels = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
    finma_refs = {
        0: 'Standard CDD applicable (FINMA Circ. 2016/7 §32)',
        1: 'Enhanced monitoring recommended (FINMA Circ. 2016/7 §44)',
        2: 'Enhanced Due Diligence required (FINMA Circ. 2016/7 §52 — AMLA Art. 6)'
    }
    return f"""CLASSIFICATION: {risk_labels[label]} RISK
PRIMARY DRIVERS: {top_str}
REGULATORY REFERENCE: {finma_refs[label]}
RECOMMENDED ACTION: {'Standard onboarding' if label == 0 else 'Senior review required' if label == 1 else 'EDD mandatory — escalate to compliance'}
GENERATED: Automated KYC Risk Assessment System v1.0"""

def plot_waterfall(explainer, X_input, label_idx, feature_names):
    explanation = explainer(X_input)
    if len(explanation.values.shape) == 3:
        waterfall_exp = shap.Explanation(
            values=explanation.values[0, :, label_idx],
            base_values=explanation.base_values[0, label_idx],
            data=explanation.data[0],
            feature_names=feature_names
        )
    else:
        waterfall_exp = explanation[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#1a1d27')
    ax.set_facecolor('#1a1d27')
    shap.plots.waterfall(waterfall_exp, show=False)
    plt.tight_layout()
    return fig

# =============================================
# INTERFACE 1 — INDIVIDUAL ANALYSIS
# =============================================
if mode == "Individual Analysis":
    st.markdown("# Individual Client Risk Assessment")
    st.markdown('<p class="section-title">Client Profile</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        country = st.selectbox("Country of Domicile", COUNTRIES)
        sector = st.selectbox("Business Sector", SECTORS)
        is_pep = st.selectbox("PEP Status", [0, 1], format_func=lambda x: "Yes — PEP" if x == 1 else "No")

    with col2:
        transaction_volume = st.number_input("Annual Transaction Volume (CHF)", min_value=1000, max_value=10000000, value=150000, step=10000)
        account_age_years = st.slider("Account Age (years)", 1, 30, 5)
        nb_transactions_30d = st.slider("Transactions (last 30 days)", 1, 100, 20)

    with col3:
        nb_countries_involved = st.slider("Countries Involved", 1, 15, 3)
        cash_ratio = st.slider("Cash Ratio", 0.0, 1.0, 0.2, step=0.05)
        adverse_media_score = st.selectbox("Adverse Media Score", [0, 1, 2, 3],
                                           format_func=lambda x: f"{x} — {'None' if x==0 else 'Low' if x==1 else 'Medium' if x==2 else 'High'}")
        beneficial_owner_complexity = st.selectbox("Beneficial Owner Complexity", [0, 1, 2],
                                                    format_func=lambda x: "Simple" if x==0 else "Holding" if x==1 else "Complex offshore")
        source_of_wealth_verified = st.selectbox("Source of Wealth Verified", [0, 1],
                                                   format_func=lambda x: "Yes" if x==1 else "No")

    st.markdown("---")

    if st.button("RUN RISK ASSESSMENT"):
        X_input = prepare_input(country, is_pep, sector, transaction_volume, account_age_years,
                                nb_transactions_30d, nb_countries_involved, cash_ratio,
                                adverse_media_score, beneficial_owner_complexity, source_of_wealth_verified)

        model = xgb if model_choice == "XGBoost" else rf
        explainer = explainer_xgb if model_choice == "XGBoost" else explainer_rf

        label = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

        explanation = explainer(X_input)
        if len(explanation.values.shape) == 3:
            shap_vals = explanation.values[0, :, label]
        else:
            shap_vals = explanation.values[0]

        reg_text = get_regulatory_explanation(label, shap_vals, feature_names, X_input)

        # Store in session state
        st.session_state['results'] = {
            'label': label, 'proba': proba, 'X_input': X_input,
            'reg_text': reg_text, 'shap_vals': shap_vals,
            'country': country, 'sector': sector, 'is_pep': is_pep,
            'transaction_volume': transaction_volume, 'account_age_years': account_age_years,
            'nb_transactions_30d': nb_transactions_30d, 'nb_countries_involved': nb_countries_involved,
            'cash_ratio': cash_ratio, 'adverse_media_score': adverse_media_score,
            'beneficial_owner_complexity': beneficial_owner_complexity,
            'source_of_wealth_verified': source_of_wealth_verified,
            'model_choice': model_choice, 'explainer': explainer
        }

    # Show results from session state
    if 'results' in st.session_state:
        r = st.session_state['results']
        label = r['label']
        proba = r['proba']
        X_input = r['X_input']
        reg_text = r['reg_text']
        explainer = r['explainer']

        st.markdown("---")
        st.markdown('<p class="section-title">Risk Classification Result</p>', unsafe_allow_html=True)

        col_result, col_proba = st.columns([1, 2])

        with col_result:
            st.markdown(f'<div class="{COLOR_MAP[label]}">{LABEL_MAP[label]}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style='margin-top:12px; font-size:0.8rem; color:#6b7a99;'>
            Model: {r['model_choice']}<br>
            Confidence: {proba[label]*100:.1f}%
            </div>
            """, unsafe_allow_html=True)

        with col_proba:
            fig_proba, ax = plt.subplots(figsize=(6, 2))
            fig_proba.patch.set_facecolor('#1a1d27')
            ax.set_facecolor('#1a1d27')
            colors_list = ['#27ae60', '#e67e22', '#e74c3c']
            bars = ax.barh(['Low', 'Medium', 'High'], proba, color=colors_list, height=0.5)
            for bar, p in zip(bars, proba):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{p*100:.1f}%', va='center', color='white', fontsize=10)
            ax.set_xlim(0, 1.15)
            ax.tick_params(colors='white')
            ax.spines[:].set_visible(False)
            ax.set_title('Class Probabilities', color='white', fontsize=11, pad=8)
            plt.tight_layout()
            st.pyplot(fig_proba)
            plt.close()

        st.markdown('<p class="section-title">SHAP Explanation — Why this classification?</p>', unsafe_allow_html=True)
        fig_waterfall = plot_waterfall(explainer, X_input, label, feature_names)
        st.pyplot(fig_waterfall)
        plt.close()

        st.markdown(f'<div class="reg-box">{reg_text}</div>', unsafe_allow_html=True)

        # PDF EXPORT
        st.markdown('<p class="section-title">Export Report</p>', unsafe_allow_html=True)

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm, mm
        from reportlab.lib import colors as rl_colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io, datetime

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                topMargin=1.8*cm, bottomMargin=1.8*cm,
                                leftMargin=2*cm, rightMargin=2*cm)

        story = []
        blue      = rl_colors.HexColor('#2980b9')
        lightgrey = rl_colors.HexColor('#f5f7fa')
        midgrey   = rl_colors.HexColor('#dee2e6')
        darktext  = rl_colors.HexColor('#2c3e50')
        reg_bg    = rl_colors.HexColor('#eef2f7')
        risk_col  = {0: rl_colors.HexColor('#27ae60'),
                     1: rl_colors.HexColor('#e67e22'),
                     2: rl_colors.HexColor('#e74c3c')}

        # ── Styles ──────────────────────────────────────────────
        title_s = ParagraphStyle('title_s', fontName='Helvetica-Bold', fontSize=15,
                                  textColor=blue, alignment=TA_CENTER,
                                  spaceAfter=2*mm)
        meta_s  = ParagraphStyle('meta_s',  fontName='Helvetica', fontSize=8,
                                  textColor=rl_colors.grey, alignment=TA_CENTER,
                                  spaceAfter=3*mm)
        badge_s = ParagraphStyle('badge_s', fontName='Helvetica-Bold', fontSize=17,
                                  textColor=rl_colors.white, alignment=TA_CENTER,
                                  backColor=risk_col[label],
                                  borderPadding=(4*mm, 0, 4*mm, 0),
                                  spaceAfter=1*mm)
        conf_s  = ParagraphStyle('conf_s',  fontName='Helvetica', fontSize=8,
                                  textColor=rl_colors.grey, alignment=TA_CENTER,
                                  spaceAfter=4*mm)
        sec_s   = ParagraphStyle('sec_s',   fontName='Helvetica-Bold', fontSize=9,
                                  textColor=blue, alignment=TA_LEFT,
                                  spaceBefore=4*mm, spaceAfter=2*mm)
        cell_s  = ParagraphStyle('cell_s',  fontName='Helvetica', fontSize=8,
                                  textColor=darktext, leading=11)
        reg_line_s = ParagraphStyle('reg_line_s', fontName='Courier', fontSize=8,
                                     textColor=darktext, leading=13,
                                     spaceAfter=1.5*mm)
        foot_s  = ParagraphStyle('foot_s',  fontName='Helvetica', fontSize=7,
                                  textColor=rl_colors.grey, alignment=TA_CENTER,
                                  spaceBefore=2*mm)

        # ── Header ──────────────────────────────────────────────
        story.append(Paragraph("KYC Risk Classification Report", title_s))
        story.append(Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  Model: {r['model_choice']}",
            meta_s))
        story.append(HRFlowable(width="100%", thickness=1.2, color=blue, spaceAfter=5*mm))

        # ── Risk Badge ──────────────────────────────────────────
        story.append(Paragraph(LABEL_MAP[label], badge_s))
        story.append(Spacer(1, 2*mm))
        conf_s2 = ParagraphStyle('conf_s2', fontName='Helvetica-Oblique', fontSize=9,
                                  textColor=darktext, alignment=TA_CENTER, spaceAfter=0)
        story.append(Paragraph(f"Confidence: {proba[label]*100:.1f}%", conf_s2))
        story.append(Spacer(1, 5*mm))

        # ── Tables side by side ─────────────────────────────────
        ts = TableStyle([
            ('BACKGROUND',    (0, 0), (-1, 0), blue),
            ('TEXTCOLOR',     (0, 0), (-1, 0), rl_colors.white),
            ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1,-1), 8),
            ('ROWBACKGROUNDS',(0, 1), (-1,-1), [lightgrey, rl_colors.white]),
            ('GRID',          (0, 0), (-1,-1), 0.4, midgrey),
            ('LEFTPADDING',   (0, 0), (-1,-1), 5),
            ('RIGHTPADDING',  (0, 0), (-1,-1), 5),
            ('TOPPADDING',    (0, 0), (-1,-1), 3),
            ('BOTTOMPADDING', (0, 0), (-1,-1), 3),
            ('VALIGN',        (0, 0), (-1,-1), 'MIDDLE'),
        ])

        profile_rows = [
            ['Parameter', 'Value'],
            ['Country',        r['country']],
            ['Sector',         r['sector']],
            ['PEP Status',     'Yes' if r['is_pep']==1 else 'No'],
            ['Volume (CHF)',    f"{r['transaction_volume']:,.0f}"],
            ['Account Age',    f"{r['account_age_years']} years"],
            ['Countries',      str(r['nb_countries_involved'])],
            ['Cash Ratio',     f"{r['cash_ratio']:.2f}"],
            ['Adverse Media',  f"{r['adverse_media_score']} / 3"],
            ['BO Complexity',  ['Simple','Holding','Offshore'][r['beneficial_owner_complexity']]],
            ['SoW Verified',   'Yes' if r['source_of_wealth_verified']==1 else 'No'],
        ]
        proba_rows = [
            ['Risk Class',   'Probability'],
            ['Low Risk',     f'{proba[0]*100:.1f}%'],
            ['Medium Risk',  f'{proba[1]*100:.1f}%'],
            ['High Risk',    f'{proba[2]*100:.1f}%'],
        ]

        t_profile = Table(profile_rows, colWidths=[4.8*cm, 3.8*cm])
        t_profile.setStyle(ts)
        t_proba   = Table(proba_rows,   colWidths=[4.2*cm, 3.2*cm])
        t_proba.setStyle(ts)

        story.append(Paragraph("CLIENT PROFILE", sec_s))
        side = Table(
            [[t_profile, '', t_proba]],
            colWidths=[8.6*cm, 0.7*cm, 7.4*cm]
        )
        side.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP')]))
        story.append(side)
        story.append(Spacer(1, 5*mm))

        # ── Regulatory Box ──────────────────────────────────────
        story.append(Paragraph("REGULATORY EXPLANATION (FINMA)", sec_s))

        reg_lines = [l.strip() for l in reg_text.split('\n') if l.strip()]
        reg_table_data = [[Paragraph(line, reg_line_s)] for line in reg_lines]
        reg_table = Table(reg_table_data, colWidths=[16.5*cm])
        reg_table.setStyle(TableStyle([
            ('BACKGROUND',   (0,0), (-1,-1), reg_bg),
            ('LEFTPADDING',  (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING',   (0,0), (-1,-1), 4),
            ('BOTTOMPADDING',(0,0), (-1,-1), 4),
            ('LINEABOVE',    (0,0), (-1, 0), 1, blue),
            ('LINEBELOW',    (0,-1),(-1,-1), 0.5, midgrey),
        ]))
        story.append(reg_table)

        # ── Footer ──────────────────────────────────────────────
        story.append(Spacer(1, 4*mm))
        story.append(HRFlowable(width="100%", thickness=0.4, color=midgrey))
        story.append(Paragraph(
            "KYC Risk Classification System  ·  HEC Lausanne MScIS Thesis 2026  ·  "
            "Youssouf Chaib  ·  FINMA Circular 2016/7 & AMLA  ·  DSR Peffers et al. (2007)",
            foot_s))

        doc.build(story)
        buffer.seek(0)

        st.download_button(
            label="DOWNLOAD PDF",
            data=buffer,
            file_name=f"KYC_Report_{r['country']}_{LABEL_MAP[label].replace(' ', '_')}.pdf",
            mime="application/pdf",
            key="pdf_download_btn"
        )


# =============================================
# INTERFACE 2 — PORTFOLIO ANALYSIS
# =============================================
else:
    st.markdown("# Portfolio Risk Analysis")
    st.markdown('<p class="section-title">Upload Client Portfolio</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV file (same format as kyc_synthetic.csv)",
        type=['csv']
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"{len(df)} clients loaded.")

        # Preprocess
        df_model = df.copy()
        if 'country_risk' in df_model.columns:
            df_model = df_model.drop(columns=['client_id', 'country_risk'], errors='ignore')
        else:
            df_model = df_model.drop(columns=['client_id'], errors='ignore')

        df_model['country'] = le_country.transform(df_model['country'])
        df_model['sector'] = le_sector.transform(df_model['sector'])

        if 'risk_label' in df_model.columns:
            df_model = df_model.drop(columns=['risk_label'])
        if 'risk_label_str' in df_model.columns:
            df_model = df_model.drop(columns=['risk_label_str'])

        X_portfolio = df_model[feature_names]
        model = xgb if model_choice == "XGBoost" else rf
        predictions = model.predict(X_portfolio)
        probas = model.predict_proba(X_portfolio)

        df['predicted_risk'] = predictions
        df['predicted_risk_label'] = pd.Series(predictions).map({0: 'Low', 1: 'Medium', 2: 'High'})
        df['confidence'] = [probas[i][predictions[i]] for i in range(len(predictions))]

        # Dashboard metrics
        st.markdown("---")
        st.markdown('<p class="section-title">Portfolio Overview</p>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        counts = df['predicted_risk_label'].value_counts()

        with col1:
            st.metric("Total Clients", len(df))
        with col2:
            st.metric("Low Risk", counts.get('Low', 0), delta=f"{counts.get('Low',0)/len(df)*100:.0f}%")
        with col3:
            st.metric("Medium Risk", counts.get('Medium', 0), delta=f"{counts.get('Medium',0)/len(df)*100:.0f}%")
        with col4:
            st.metric("High Risk", counts.get('High', 0), delta=f"{counts.get('High',0)/len(df)*100:.0f}%")

        # Distribution chart
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig_dist, ax = plt.subplots(figsize=(6, 4))
            fig_dist.patch.set_facecolor('#1a1d27')
            ax.set_facecolor('#1a1d27')
            labels_order = ['Low', 'Medium', 'High']
            values = [counts.get(l, 0) for l in labels_order]
            colors = ['#27ae60', '#e67e22', '#e74c3c']
            wedges, texts, autotexts = ax.pie(values, labels=labels_order, colors=colors,
                                               autopct='%1.1f%%', startangle=90)
            for text in texts + autotexts:
                text.set_color('white')
            ax.set_title('Risk Distribution', color='white', fontsize=12, pad=12)
            st.pyplot(fig_dist)
            plt.close()

        with col_chart2:
            fig_bar, ax = plt.subplots(figsize=(6, 4))
            fig_bar.patch.set_facecolor('#1a1d27')
            ax.set_facecolor('#1a1d27')
            country_risk = df.groupby(['country', 'predicted_risk_label']).size().unstack(fill_value=0)
            if 'High' in country_risk.columns:
                country_risk['High'].sort_values(ascending=True).tail(8).plot(
                    kind='barh', ax=ax, color='#e74c3c', alpha=0.85)
            ax.set_title('Top Countries — High Risk Clients', color='white', fontsize=11)
            ax.tick_params(colors='white')
            ax.spines[:].set_visible(False)
            ax.set_xlabel('Count', color='white')
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close()

        # Filterable table
        st.markdown('<p class="section-title">Client Table</p>', unsafe_allow_html=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_risk = st.multiselect("Filter by Risk Level", ['Low', 'Medium', 'High'],
                                          default=['High', 'Medium'])
        with col_f2:
            filter_country = st.multiselect("Filter by Country", sorted(df['country'].unique()),
                                             default=[])

        df_filtered = df[df['predicted_risk_label'].isin(filter_risk)]
        if filter_country:
            df_filtered = df_filtered[df_filtered['country'].isin(filter_country)]

        display_cols = ['client_id', 'country', 'sector', 'is_pep', 'adverse_media_score',
                        'predicted_risk_label', 'confidence']
        st.dataframe(
            df_filtered[display_cols].sort_values('confidence', ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # SHAP on selected client
        st.markdown('<p class="section-title">Individual SHAP Explanation</p>', unsafe_allow_html=True)
        selected_client = st.selectbox(
            "Select a client for SHAP analysis",
            df_filtered['client_id'].tolist() if len(df_filtered) > 0 else df['client_id'].tolist()
        )

        if selected_client:
            client_idx = df[df['client_id'] == selected_client].index[0]
            X_client = X_portfolio.iloc[[client_idx - df.index[0]]]
            label_client = predictions[client_idx - df.index[0]]
            explainer = explainer_xgb if model_choice == "XGBoost" else explainer_rf
            fig_w = plot_waterfall(explainer, X_client, label_client, feature_names)
            st.pyplot(fig_w)
            plt.close()
    else:
        st.info("Upload a CSV file to analyze your portfolio. Use `data/kyc_synthetic.csv` as reference format.")
        st.markdown("""
        **Expected columns:**
        `client_id, country, country_risk, is_pep, sector, transaction_volume, account_age_years,
        nb_transactions_30d, avg_transaction_amount, nb_countries_involved, cash_ratio,
        adverse_media_score, beneficial_owner_complexity, source_of_wealth_verified`
        """)