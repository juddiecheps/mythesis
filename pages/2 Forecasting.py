import streamlit as st
import shap
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.navbar import inject_navbar
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(
    page_title="Explainable AI — Manufacturing Kenya",
    layout="wide",
    initial_sidebar_state="collapsed"
)

inject_navbar("xai")

# ── Load ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    model  = load_model("models/mlp_cement_model.keras")
    scaler = joblib.load("models/minmax_scaler.pkl")
    meta   = joblib.load("models/metadata.pkl")
    data   = pd.read_csv("data/clean_data.csv", index_col=0)
    return model, scaler, meta, data

model, scaler, meta, data = load_model_and_data()
columns   = meta["columns"]
look_back = meta["look_back"]

# ── Inline controls ──────────────────────────────────────────────────────────
st.markdown('<style>.ctrl-lbl{font-size:0.66rem;letter-spacing:0.15em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:0.4rem;}</style>', unsafe_allow_html=True)
xctrl1, xctrl2 = st.columns([1, 1])
with xctrl1:
    st.markdown('<div class="ctrl-lbl">Samples to Explain</div>', unsafe_allow_html=True)
    n_samples = st.slider("Samples", 5, 15, 10, label_visibility="collapsed")
with xctrl2:
    st.markdown('<div class="ctrl-lbl">Waterfall Sample Index</div>', unsafe_allow_html=True)
    sample_idx = st.slider("Sample index", 0, n_samples - 1, 0, label_visibility="collapsed")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="eyebrow">Interpretability</div>
    <h1>Explainable AI — SHAP Analysis</h1>
    <p>Feature attribution analysis using SHAP values to understand model behaviour and predictor importance</p>
</div>
""", unsafe_allow_html=True)

# ── Prepare ───────────────────────────────────────────────────────────────────
def make_windows(df, cols, look_back):
    X = []
    for i in range(len(df) - look_back):
        window = scaler.transform(df[cols].iloc[i:i + look_back])
        X.append(window.flatten())
    return np.array(X)

X_all        = make_windows(data, columns, look_back)
X_background = X_all[:50]
X_test       = X_all[-n_samples:]
feature_names = [
    f"{col}_t-{look_back - i - 1}"
    for i in range(look_back)
    for col in columns
]

# ── SHAP ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice" style="margin-bottom:1.5rem;">
    Computing SHAP values — this typically takes 15–30 seconds on first load.
</div>
""", unsafe_allow_html=True)

with st.spinner("Running SHAP attribution analysis..."):
    try:
        explainer      = shap.DeepExplainer(model, X_background)
        shap_values    = explainer.shap_values(X_test)
        expected_value = explainer.expected_value
        method         = "DeepExplainer"
    except Exception:
        explainer      = shap.KernelExplainer(model.predict, X_background[:20])
        shap_values    = explainer.shap_values(X_test, nsamples=100)
        expected_value = explainer.expected_value
        method         = "KernelExplainer"

# Normalise shap_values to plain 2-D float array
if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.array(shap_values, dtype=float)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]
shap_values = shap_values[:len(X_test)]

# Safely convert expected_value to a plain Python float.
# DeepExplainer can return a TF tensor; calling float() on it directly
# raises "Scalar tensor has no len()" inside shap.Explanation.
def _safe_float(v):
    try:
        if hasattr(v, "numpy"):          # TF/Keras tensor
            v = v.numpy()
        arr = np.array(v, dtype=float).flatten()
        return float(arr[0]) if len(arr) > 0 else 0.0
    except Exception:
        return 0.0

if isinstance(expected_value, (list, tuple)):
    expected_value = _safe_float(expected_value[0])
else:
    expected_value = _safe_float(expected_value)

NAVY = "#0d1b2a"; TEAL = "#1a6b72"; GOLD = "#c8973a"; LIGHT = "#f5f0e8"; RED = "#c0392b"

# ── Aggregated importance ─────────────────────────────────────────────────────
grouped = {}
for i, fname in enumerate(feature_names):
    base = fname.rsplit("_t-", 1)[0]
    grouped.setdefault(base, []).append(i)

agg_importance = {
    feat: np.mean(np.abs(shap_values[:, idxs]))
    for feat, idxs in grouped.items()
}
agg_df = (pd.DataFrame.from_dict(agg_importance, orient="index", columns=["Importance"])
          .sort_values("Importance", ascending=False))

col1, col2 = st.columns([2, 3], gap="large")

with col1:
    st.markdown('<div class="section-label">Aggregate View</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Indicator Importance</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:0.84rem;color:var(--muted);margin-bottom:1rem;line-height:1.6;">
Mean absolute SHAP across all 12 time lags — which indicators drive the model's forecasts most.
</div>""", unsafe_allow_html=True)

    max_val = agg_df["Importance"].max()
    rows_html = ""
    for feat, row in agg_df.iterrows():
        pct   = row["Importance"] / max_val * 100
        clean = feat.replace("_", " ")
        rows_html += f"""
<div style="padding:0.55rem 0;border-bottom:1px solid var(--border);">
    <div style="font-size:0.82rem;color:var(--navy);margin-bottom:0.28rem;">{clean}</div>
    <div style="display:flex;align-items:center;gap:0.6rem;">
        <div style="flex:1;height:7px;background:#e8e4de;border-radius:4px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:100%;background:{TEAL};border-radius:4px;"></div>
        </div>
        <span style="font-size:0.78rem;font-variant-numeric:tabular-nums;color:{TEAL};font-weight:600;width:52px;text-align:right;">{row['Importance']:.4f}</span>
    </div>
</div>"""
    st.markdown(f'<div class="card card-teal">{rows_html}</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-label">Distribution Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">SHAP Summary Plot</div>', unsafe_allow_html=True)

    fig1, _ = plt.subplots(figsize=(7.5, 6))
    fig1.patch.set_facecolor(LIGHT)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, max_display=18, show=False, plot_size=None)
    ax1 = plt.gca()
    ax1.set_facecolor(LIGHT); fig1.patch.set_facecolor(LIGHT)
    ax1.tick_params(labelsize=7.5, colors="#5a6475")
    for sp in ["top","right"]: ax1.spines[sp].set_visible(False)
    ax1.spines["left"].set_color("#d6cfc4"); ax1.spines["bottom"].set_color("#d6cfc4")
    ax1.set_xlabel("SHAP value (impact on model output)", fontsize=8.5, color="#5a6475")
    plt.tight_layout(pad=1.2)
    st.pyplot(fig1); plt.close(fig1)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Waterfall ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Single Prediction Breakdown</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">Waterfall Explanation — Sample {sample_idx}</div>', unsafe_allow_html=True)

single_shap = np.array(shap_values[sample_idx]).reshape(-1)
single_data = np.array(X_test[sample_idx]).reshape(-1)

col3, col4 = st.columns([3, 2], gap="large")

with col3:
    explanation = shap.Explanation(
        values=single_shap, base_values=expected_value,
        data=single_data, feature_names=feature_names,
    )
    fig2, _ = plt.subplots(figsize=(7.5, 6))
    fig2.patch.set_facecolor(LIGHT)
    shap.plots.waterfall(explanation, max_display=12, show=False)
    ax2 = plt.gca()
    ax2.set_facecolor(LIGHT); fig2.patch.set_facecolor(LIGHT)
    ax2.tick_params(labelsize=7.5, colors="#5a6475")
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
    ax2.spines["left"].set_color("#d6cfc4"); ax2.spines["bottom"].set_color("#d6cfc4")
    plt.tight_layout(pad=1.2)
    st.pyplot(fig2); plt.close(fig2)

with col4:
    st.markdown('<div class="section-label">Top Drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Key Feature Impacts</div>', unsafe_allow_html=True)

    top_k   = 8
    top_idx = np.argsort(np.abs(single_shap))[::-1][:top_k]
    max_abs = np.abs(single_shap[top_idx]).max()

    driver_rows = ""
    for idx in top_idx:
        val   = single_shap[idx]
        pct   = abs(val) / max_abs * 100
        color = TEAL if val > 0 else RED
        sign  = "+" if val > 0 else ""
        clean = feature_names[idx].replace("_", " ")
        driver_rows += f"""
<div style="padding:0.5rem 0;border-bottom:1px solid var(--border);">
    <div style="font-size:0.78rem;color:var(--muted);margin-bottom:0.22rem;">{clean}</div>
    <div style="display:flex;align-items:center;gap:0.6rem;">
        <div style="flex:1;height:6px;background:#e8e4de;border-radius:3px;overflow:hidden;">
            <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:3px;"></div>
        </div>
        <span style="font-size:0.8rem;font-weight:700;color:{color};width:60px;text-align:right;">{sign}{val:.4f}</span>
    </div>
</div>"""

    st.markdown(f"""
<div class="card card-gold">
{driver_rows}
<p style="font-size:0.74rem;color:var(--muted);margin:0.8rem 0 0 0;">
Teal = positive impact (raises forecast). Red = negative impact (lowers forecast). Values in MinMax-scaled units.
</p>
</div>
<div class="card" style="margin-top:1rem;padding:1rem 1.2rem;">
    <div style="font-size:0.66rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:0.4rem;">Method</div>
    <div style="font-size:0.85rem;color:var(--navy);font-weight:600;margin-bottom:0.3rem;">{method}</div>
    <div style="font-size:0.8rem;color:var(--muted);line-height:1.6;">
        {n_samples} test samples · {len(X_background)} background samples · Model: MLP
    </div>
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
    color: #ffffff;
    letter-spacing: 0.02em;
}
.footer-institution {
    font-size: 0.72rem;
    color: #4a6278;
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
    color: #2e4055;
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
    <div class="footer-right">Manufacturing Sector Forecasting &nbsp;·&nbsp; 2025</div>
</div>
""", unsafe_allow_html=True)