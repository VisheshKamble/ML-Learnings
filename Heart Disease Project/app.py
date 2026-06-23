import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan · Heart Risk Predictor",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    errors = []
    model, scaler, columns = None, None, None

    for name, path in [("model", "./KNN_heart.pkl"), ("scaler", "./scaler.pkl"), ("columns", "./columns.pkl")]:
        if not os.path.exists(path):
            errors.append(f"Missing file: `{path}`")
            continue
        try:
            obj = joblib.load(path)
            if name == "model":
                model = obj
            elif name == "scaler":
                scaler = obj
            else:
                columns = obj
        except Exception as e:
            errors.append(f"Failed to load `{path}`: {e}")

    return model, scaler, columns, errors

model, scaler, columns, load_errors = load_artifacts()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #060810 !important;
    color: #ECEDF2;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

[data-testid="stMainBlockContainer"] {
    padding: 0 24px 80px !important;
    max-width: 860px !important;
    margin: 0 auto !important;
}

.block-container {
    padding: 0 !important;
    max-width: 860px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #060810; }
::-webkit-scrollbar-thumb { background: #FF2D55; border-radius: 2px; }

/* ─── HERO ─────────────────────────────────────────────────────────── */
.hero {
    padding: 72px 0 56px;
    text-align: center;
    position: relative;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(255,45,85,0.08);
    border: 1px solid rgba(255,45,85,0.2);
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #FF2D55;
    margin-bottom: 28px;
}

.hero-badge-dot {
    width: 5px; height: 5px;
    background: #FF2D55;
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}

.hero-title {
    font-size: clamp(36px, 6vw, 62px);
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.05;
    color: #F0F1F8;
    margin-bottom: 18px;
}

.hero-title em {
    font-style: normal;
    background: linear-gradient(135deg, #FF2D55 0%, #FF8FA3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 16px;
    font-weight: 400;
    color: #6B7280;
    line-height: 1.7;
    max-width: 480px;
    margin: 0 auto 40px;
}

/* EKG line decoration */
.hero-ekg {
    width: 100%;
    max-width: 320px;
    margin: 0 auto 0;
    opacity: 0.25;
}

/* ─── STATS ROW ─────────────────────────────────────────────────────── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 40px;
}

.stat-cell {
    background: #0A0D18;
    padding: 20px 16px;
    text-align: center;
}

.stat-val {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #F0F1F8;
    margin-bottom: 4px;
}

.stat-lbl {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4B5060;
}

/* ─── SECTION LABELS ─────────────────────────────────────────────────── */
.sec-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4B5060;
    margin: 32px 0 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.04);
}

/* ─── STREAMLIT WIDGET OVERRIDES ─────────────────────────────────────── */
/* Number input */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stRadio"] > label {
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #5A6070 !important;
    margin-bottom: 6px !important;
}

[data-testid="stNumberInput"] input {
    background: #0D1120 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #E8EAF0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 11px 14px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

[data-testid="stNumberInput"] input:focus {
    border-color: rgba(255,45,85,0.45) !important;
    box-shadow: 0 0 0 3px rgba(255,45,85,0.07) !important;
    outline: none !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #0D1120 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #E8EAF0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: border-color 0.15s ease !important;
}

[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(255,45,85,0.35) !important;
}

/* Slider */
[data-testid="stSlider"] [role="slider"] {
    background: #FF2D55 !important;
    border: 2px solid #FF2D55 !important;
    box-shadow: 0 0 12px rgba(255,45,85,0.5) !important;
}

/* ─── PREDICT BUTTON ─────────────────────────────────────────────────── */
.stButton > button {
    width: 100% !important;
    background: #FF2D55 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 16px 0 !important;
    height: auto !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(255,45,85,0.3) !important;
    margin-top: 12px !important;
}

.stButton > button:hover {
    background: #E8193F !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(255,45,85,0.45) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 12px rgba(255,45,85,0.3) !important;
}

/* ─── RESULT CARD ─────────────────────────────────────────────────────── */
.result-wrap {
    margin-top: 40px;
    animation: rise-in 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}

@keyframes rise-in {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-card {
    border-radius: 20px;
    padding: 36px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    padding: 1px;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}

.result-card.danger {
    background: radial-gradient(ellipse 100% 100% at 50% 0%, rgba(255,45,85,0.12) 0%, rgba(13,17,32,0.98) 70%);
    border: 1px solid rgba(255,45,85,0.25);
}

.result-card.safe {
    background: radial-gradient(ellipse 100% 100% at 50% 0%, rgba(52,211,153,0.1) 0%, rgba(13,17,32,0.98) 70%);
    border: 1px solid rgba(52,211,153,0.22);
}

.result-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 24px;
}

.result-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4B5060;
    margin-bottom: 10px;
}

.result-status {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.025em;
    line-height: 1.1;
}

.result-status.danger { color: #FF2D55; }
.result-status.safe   { color: #34D399; }

.result-desc {
    font-size: 14px;
    font-weight: 400;
    color: #6B7280;
    line-height: 1.65;
    max-width: 480px;
}

/* Confidence bar */
.conf-wrap { margin-top: 28px; }

.conf-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.conf-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4B5060;
}

.conf-pct {
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.conf-pct.danger { color: #FF2D55; }
.conf-pct.safe   { color: #34D399; }

.conf-track {
    width: 100%;
    height: 5px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
}

.conf-fill {
    height: 100%;
    border-radius: 3px;
}

.conf-fill.danger { background: linear-gradient(90deg, #FF2D55, #FF8FA3); }
.conf-fill.safe   { background: linear-gradient(90deg, #059669, #34D399); }

/* ─── SUMMARY GRID ──────────────────────────────────────────────────── */
.summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 16px;
}

.summary-card {
    background: #0A0D18;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 20px;
}

.summary-card-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4B5060;
    margin-bottom: 14px;
}

.summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.summary-row:last-child { border-bottom: none; }

.summary-key {
    font-size: 12px;
    color: #4B5060;
    font-weight: 400;
}

.summary-val {
    font-size: 13px;
    color: #C0C5D8;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

/* ─── INDICATOR BARS ─────────────────────────────────────────────────── */
.ind-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
}

.ind-name {
    font-size: 11px;
    color: #4B5060;
    font-weight: 500;
    width: 110px;
    flex-shrink: 0;
    letter-spacing: 0.03em;
}

.ind-track {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.05);
    border-radius: 2px;
    overflow: hidden;
}

.ind-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #FF2D55, #FF8FA3);
}

.ind-val {
    font-size: 11px;
    color: #4B5060;
    font-weight: 500;
    width: 32px;
    text-align: right;
    flex-shrink: 0;
}

/* ─── IDLE STATE ────────────────────────────────────────────────────── */
.idle-wrap {
    background: #0A0D18;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 48px 32px;
    text-align: center;
    margin-top: 40px;
}

.idle-icon {
    width: 56px; height: 56px;
    margin: 0 auto 20px;
    background: rgba(255,45,85,0.07);
    border: 1px solid rgba(255,45,85,0.14);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.idle-title {
    font-size: 18px;
    font-weight: 600;
    color: #C0C5D8;
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}

.idle-text {
    font-size: 14px;
    color: #4B5060;
    line-height: 1.65;
}

/* ─── DISCLAIMER ────────────────────────────────────────────────────── */
.disclaimer {
    margin-top: 32px;
    padding: 16px 20px;
    background: rgba(255,180,0,0.04);
    border: 1px solid rgba(255,180,0,0.12);
    border-radius: 12px;
    font-size: 12px;
    color: #7A6A40;
    line-height: 1.6;
    text-align: center;
}

/* ─── ERROR BANNER ───────────────────────────────────────────────────── */
.err-banner {
    background: rgba(255,45,85,0.06);
    border: 1px solid rgba(255,45,85,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 13px;
    color: #FF8FA3;
    line-height: 1.6;
    margin-bottom: 24px;
}

.err-banner code {
    background: rgba(255,45,85,0.1);
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 12px;
}

hr.div { border: none; border-top: 1px solid rgba(255,255,255,0.04); margin: 8px 0 20px; }

</style>
""", unsafe_allow_html=True)

# ── SVG Assets ────────────────────────────────────────────────────────────────
EKG_PATH = "M0 30 L40 30 L55 30 L65 5 L75 55 L82 15 L90 48 L98 30 L140 30 L155 30 L165 3 L175 57 L183 12 L191 50 L200 30 L260 30"

DANGER_ICON = """<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
  <circle cx="14" cy="14" r="13" stroke="#FF2D55" stroke-width="1.5" opacity="0.4"/>
  <path d="M14 17.5C14 17.5 8 13.5 8 9.5C8 7.01 9.79 5 12 5C13.17 5 14.2 5.6 14.88 6.52C15.56 5.6 16.59 5 17.76 5C19.97 5 21.76 7.01 21.76 9.5C21.76 13.5 14 17.5 14 17.5Z"
        fill="rgba(255,45,85,0.12)" stroke="#FF2D55" stroke-width="1.4" stroke-linejoin="round"/>
  <circle cx="14" cy="21.5" r="1.5" fill="#FF2D55"/>
</svg>"""

SAFE_ICON = """<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
  <circle cx="14" cy="14" r="13" stroke="#34D399" stroke-width="1.5" opacity="0.4"/>
  <path d="M9 14L12.5 17.5L19 11" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

PULSE_ICON = """<svg width="26" height="26" viewBox="0 0 26 26" fill="none">
  <path d="M3 13 L8 13 L10 7 L13 19 L16 9 L18 15 L23 15"
        stroke="#FF2D55" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-badge">
    <span class="hero-badge-dot"></span>
    Predictive Diagnostics · CardioScan
  </div>
  <div class="hero-title">Heart Disease<br><em>Risk Assessment</em></div>
  <div class="hero-sub">
    Enter patient vitals and diagnostic markers to receive an AI-powered cardiovascular risk prediction.
  </div>
  <svg class="hero-ekg" viewBox="0 0 260 60" xmlns="http://www.w3.org/2000/svg">
    <path d="{EKG_PATH}" stroke="#FF2D55" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
</div>

<div class="stats-row">
  <div class="stat-cell"><div class="stat-val">KNN</div><div class="stat-lbl">Model</div></div>
  <div class="stat-cell"><div class="stat-val">13</div><div class="stat-lbl">Features</div></div>
  <div class="stat-cell"><div class="stat-val">UCI</div><div class="stat-lbl">Dataset</div></div>
  <div class="stat-cell"><div class="stat-val">Binary</div><div class="stat-lbl">Output</div></div>
</div>
""", unsafe_allow_html=True)

# ── Load errors banner ────────────────────────────────────────────────────────
if load_errors:
    err_html = "<br>".join(f"⚠ {e}" for e in load_errors)
    st.markdown(f'<div class="err-banner">{err_html}<br><br>Place <code>KNN_heart.pkl</code>, <code>scaler.pkl</code>, and <code>columns.pkl</code> in the same directory as this script.</div>', unsafe_allow_html=True)

# ── Helper: safe selectbox parse ──────────────────────────────────────────────
def parse_first_int(val: str) -> int:
    """Extract leading integer from option strings like '0 — Normal'."""
    stripped = val.strip()
    for i, ch in enumerate(stripped):
        if not ch.isdigit():
            return int(stripped[:i]) if i > 0 else 0
    return int(stripped)

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Patient Demographics</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age (years)", min_value=20, max_value=100, value=52, step=1)
with c2:
    sex = st.selectbox("Biological Sex", options=["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0
with c3:
    trestbps = st.number_input("Resting BP (mmHg)", min_value=80, max_value=220, value=120, step=1)

st.markdown('<div class="sec-label">Cardiac Indicators</div>', unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
with c4:
    cp = st.selectbox("Chest Pain Type", options=[
        "0 — Typical Angina",
        "1 — Atypical Angina",
        "2 — Non-Anginal",
        "3 — Asymptomatic",
    ])
    cp_val = parse_first_int(cp)
with c5:
    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=210, step=1)
with c6:
    thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=158, step=1)

c7, c8, c9 = st.columns(3)
with c7:
    fbs = st.selectbox("Fasting Blood Sugar", options=["< 120 mg/dL (Normal)", "> 120 mg/dL (High)"])
    fbs_val = 0 if "Normal" in fbs else 1
with c8:
    restecg = st.selectbox("Resting ECG", options=[
        "0 — Normal",
        "1 — ST-T Abnormality",
        "2 — LV Hypertrophy",
    ])
    restecg_val = parse_first_int(restecg)
with c9:
    exang = st.selectbox("Exercise Angina", options=["No", "Yes"])
    exang_val = 1 if exang == "Yes" else 0

st.markdown('<div class="sec-label">Diagnostic Measurements</div>', unsafe_allow_html=True)

c10, c11, c12 = st.columns(3)
with c10:
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=8.0, value=0.8, step=0.1, format="%.1f")
with c11:
    slope = st.selectbox("ST Slope", options=[
        "0 — Upsloping",
        "1 — Flat",
        "2 — Downsloping",
    ])
    slope_val = parse_first_int(slope)
with c12:
    ca = st.selectbox("Major Vessels (CA)", options=["0", "1", "2", "3"])
    ca_val = int(ca)

thal_col, _ = st.columns([1, 2])
with thal_col:
    thal = st.selectbox("Thalassemia", options=[
        "1 — Normal",
        "2 — Fixed Defect",
        "3 — Reversable Defect",
    ])
    thal_val = parse_first_int(thal)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

predict_btn = st.button("Run Cardiac Risk Analysis", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if model is None or scaler is None:
        st.markdown("""
        <div class="err-banner">
          ⚠ Model files not loaded. Ensure <code>KNN_heart.pkl</code> and <code>scaler.pkl</code>
          are present in the app directory and are valid joblib files.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build input
        raw_input = {
            "age": age,
            "sex": sex_val,
            "cp": cp_val,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs_val,
            "restecg": restecg_val,
            "thalach": thalach,
            "exang": exang_val,
            "oldpeak": oldpeak,
            "slope": slope_val,
            "ca": ca_val,
            "thal": thal_val,
        }

        input_df = pd.DataFrame([raw_input])

        # Align columns to training order
        if columns is not None:
            for col in columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[list(columns)]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = int(model.predict(input_scaled)[0])

        # Probability
        try:
            proba = model.predict_proba(input_scaled)[0]
            risk_pct = round(float(proba[1]) * 100, 1)
            safe_pct = round(float(proba[0]) * 100, 1)
        except Exception:
            risk_pct = 75.0 if prediction == 1 else 25.0
            safe_pct = 100.0 - risk_pct

        # ── Result card ────────────────────────────────────────────────────
        if prediction == 1:
            cls = "danger"
            icon = DANGER_ICON
            status_text = "High Risk Detected"
            desc = "Significant cardiovascular risk indicators found. Immediate clinical consultation and further diagnostic workup are recommended."
            conf_label = "Risk Probability"
            conf_val = risk_pct
        else:
            cls = "safe"
            icon = SAFE_ICON
            status_text = "Low Risk Indicated"
            desc = "No significant cardiovascular risk markers detected. Continue routine checkups and maintain a heart-healthy lifestyle."
            conf_label = "Healthy Confidence"
            conf_val = safe_pct

        st.markdown(f"""
        <div class="result-wrap">
          <div class="result-card {cls}">
            <div class="result-top">
              <div>
                <div class="result-label">Prediction Result</div>
                <div class="result-status {cls}">{status_text}</div>
              </div>
              {icon}
            </div>
            <div class="result-desc">{desc}</div>
            <div class="conf-wrap">
              <div class="conf-header">
                <span class="conf-label">{conf_label}</span>
                <span class="conf-pct {cls}">{conf_val}%</span>
              </div>
              <div class="conf-track">
                <div class="conf-fill {cls}" style="width:{conf_val}%"></div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Summary grid ───────────────────────────────────────────────────
        st.markdown(f"""
        <div class="summary-grid">
          <div class="summary-card">
            <div class="summary-card-title">Patient Summary</div>
            <div class="summary-row">
              <span class="summary-key">Age / Sex</span>
              <span class="summary-val">{age}y / {sex}</span>
            </div>
            <div class="summary-row">
              <span class="summary-key">Resting BP</span>
              <span class="summary-val">{trestbps} mmHg</span>
            </div>
            <div class="summary-row">
              <span class="summary-key">Cholesterol</span>
              <span class="summary-val">{chol} mg/dL</span>
            </div>
            <div class="summary-row">
              <span class="summary-key">Max HR</span>
              <span class="summary-val">{thalach} bpm</span>
            </div>
            <div class="summary-row">
              <span class="summary-key">ST Depression</span>
              <span class="summary-val">{oldpeak:.1f}</span>
            </div>
            <div class="summary-row">
              <span class="summary-key">Major Vessels</span>
              <span class="summary-val">{ca_val}</span>
            </div>
          </div>

          <div class="summary-card">
            <div class="summary-card-title">Key Indicators</div>
            {"".join(f'''
            <div class="ind-row">
              <span class="ind-name">{name}</span>
              <div class="ind-track"><div class="ind-fill" style="width:{pct:.0f}%"></div></div>
              <span class="ind-val">{pct:.0f}%</span>
            </div>''' for name, pct in [
                ("Age", min(age / 80.0, 1.0) * 100),
                ("Cholesterol", min((chol - 100) / 400.0, 1.0) * 100),
                ("ST Depression", min(oldpeak / 6.0, 1.0) * 100),
                ("HR (inverse)", (1 - min(thalach / 200.0, 1.0)) * 100),
                ("Vessels (CA)", (ca_val / 3.0) * 100),
            ])}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Disclaimer ─────────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer">
          ⚕ For educational and research use only. This tool does not constitute medical advice.
          Always consult a licensed healthcare professional for diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)

else:
    # Idle state
    st.markdown(f"""
    <div class="idle-wrap">
      <div class="idle-icon">{PULSE_ICON}</div>
      <div class="idle-title">Awaiting Patient Data</div>
      <div class="idle-text">
        Fill in the vitals and diagnostic fields above,<br>
        then click <strong style="color:#C0C5D8;">Run Cardiac Risk Analysis</strong>.
      </div>
      <hr class="div" style="margin-top:28px">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:4px">
        <div><div class="stat-lbl" style="margin-bottom:4px">Demographics</div><div style="font-size:12px;color:#4B5060">Age, Sex, BP</div></div>
        <div><div class="stat-lbl" style="margin-bottom:4px">Cardiac</div><div style="font-size:12px;color:#4B5060">CP, Chol, HR, ECG</div></div>
        <div><div class="stat-lbl" style="margin-bottom:4px">Diagnostic</div><div style="font-size:12px;color:#4B5060">ST, Vessels, Thal</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)