import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils.navbar import inject_navbar

st.set_page_config(
    page_title="Manufacturing Sector Forecasting — Kenya",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_navbar("home")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, var(--navy) 0%, var(--slate) 60%, var(--teal) 100%);
    border-radius: 8px; padding: 3rem 3.5rem; margin-bottom: 2.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 220px; height: 220px; border-radius: 50%;
    background: rgba(200,151,58,0.08); border: 1px solid rgba(200,151,58,0.15);
}
.hero-eyebrow {
    font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--gold); font-weight: 600; margin-bottom: 0.75rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif; font-size: 2.5rem; font-weight: 700;
    color: #ffffff; line-height: 1.18; margin: 0 0 0.9rem 0;
}
.hero p {
    font-size: 1.05rem; color: #a8b8cc; max-width: 640px;
    line-height: 1.65; font-weight: 300; margin: 0;
}
</style>

<div class="hero">
    <div class="hero-eyebrow">Strathmore University — Dissertation Project</div>
    <h1>Manufacturing Sector<br>Forecasting in Kenya</h1>
    <p>A comparative study of ARIMA, MLP, and LSTM models for forecasting Kenya's manufacturing 
    sector activity using high-frequency production indicators from the Kenya National Bureau of Statistics.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.markdown('<div class="section-label">Research Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Study Objectives</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card card-accent">
<p style="margin:0;font-size:0.95rem;line-height:1.75;color:#2a2a3e;">
This study develops and evaluates three forecasting approaches — a classical statistical 
method (ARIMA), a feed-forward neural network (MLP), and a recurrent deep learning model 
(LSTM) — to predict monthly manufacturing sector output.
</p>
</div>
<div class="card" style="margin-top:1rem;">
<table class="styled-table">
<thead><tr><th>Objective</th><th>Method</th></tr></thead>
<tbody>
<tr><td>Establish a statistical baseline</td><td>ARIMA(2,1,2)</td></tr>
<tr><td>Capture non-linear dynamics</td><td>Multi-Layer Perceptron</td></tr>
<tr><td>Model long-term dependencies</td><td>Long Short-Term Memory</td></tr>
<tr><td>Interpret model behaviour</td><td>SHAP Value Analysis</td></tr>
</tbody>
</table>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-label">Data Coverage</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">KNBS Indicators</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="card card-gold">
<table class="styled-table">
<thead><tr><th>Indicator</th><th>Unit</th></tr></thead>
<tbody>
<tr><td>Cement Production</td><td>MT</td></tr>
<tr><td>Sugar Production</td><td>MT</td></tr>
<tr><td>Milk Intake</td><td>Mn Litres</td></tr>
<tr><td>Soft Drinks Production</td><td>'000 Litres</td></tr>
<tr><td>Assembled Vehicles</td><td>Units</td></tr>
<tr><td>Galvanized Sheets</td><td>MT</td></tr>
</tbody>
</table>
</div>
<div class="card" style="margin-top:1rem;padding:1.2rem 1.5rem;">
    <div style="font-size:0.66rem;letter-spacing:0.12em;text-transform:uppercase;color:#5a6475;font-weight:600;margin-bottom:0.3rem;">Period Covered</div>
    <div style="font-family:'Playfair Display',serif;font-size:1.3rem;color:#0d1b2a;font-weight:600;">Jan 2010 — Dec 2025</div>
    <div style="font-size:0.8rem;color:#1a6b72;margin-top:0.2rem;">192 monthly observations · No missing values</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Methodology</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Train / Validation / Test Split</div>', unsafe_allow_html=True)

st.markdown("""
<div class="metrics-row">
    <div class="metric-pill">
        <div class="label">Training Set</div>
        <div class="value">168</div>
        <div class="sub">Jan 2010 – Dec 2023</div>
    </div>
    <div class="metric-pill">
        <div class="label">Validation Set</div>
        <div class="value">6</div>
        <div class="sub">Jan 2024 – Jun 2024</div>
    </div>
    <div class="metric-pill">
        <div class="label">Test Set</div>
        <div class="value">18</div>
        <div class="sub">Jul 2024 – Dec 2025</div>
    </div>
    <div class="metric-pill">
        <div class="label">Look-back Window</div>
        <div class="value">12</div>
        <div class="sub">months of history</div>
    </div>
</div>
<div class="notice">
    Navigate to <strong>Forecasting</strong> in the top bar to generate model predictions, 
    or <strong>Explainable AI</strong> for SHAP feature attribution analysis.
    Forecasts are intended for research and policy analysis — interpret with domain expertise.
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.site-footer {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: var(--navy);
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 0.6rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    z-index: 9980;
}
.footer-name {
    font-family: 'Playfair Display', serif;
    font-size: 0.82rem;
    font-weight: 600;
    color: #ffffff;   /* White */
    letter-spacing: 0.02em;
}
.footer-institution {
    font-size: 0.72rem;
    color: #ffffff;   /* Changed to white */
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.footer-divider {
    width: 1px; height: 18px;
    background: rgba(255,255,255,0.1);
    margin: 0 1rem;
    display: inline-block;
    vertical-align: middle;
}
.footer-right {
    font-size: 0.68rem;
    color: #ffffff;   /* Changed to white */
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
/* Push page content above the footer */
.block-container { padding-bottom: 4rem !important; }
</style>

<div class="site-footer">
    <div>
        <span class="footer-name">Judith Jepkoech</span>
        <span class="footer-divider"></span>
        <span class="footer-institution">Strathmore University &nbsp;·&nbsp; Dissertation Project</span>
    </div>
    <div class="footer-right">Manufacturing Sector Forecasting &nbsp;·&nbsp; 2026</div>
</div>
""", unsafe_allow_html=True)