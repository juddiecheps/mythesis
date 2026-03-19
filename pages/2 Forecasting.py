import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys, os, warnings, logging

# ── Suppress TF/Keras warnings ─────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.navbar import inject_navbar
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Forecasting — Manufacturing Kenya", layout="wide", initial_sidebar_state="collapsed")
inject_navbar("forecasting")

@st.cache_resource
def load_assets():
    arima  = joblib.load("models/arima_cement.pkl")
    mlp    = load_model("models/mlp_cement_model.keras")
    lstm   = load_model("models/lstm_cement_model.keras")
    scaler = joblib.load("models/minmax_scaler.pkl")
    meta   = joblib.load("models/metadata.pkl")
    data   = pd.read_csv("data/clean_data.csv", parse_dates=True, index_col=0)
    return arima, mlp, lstm, scaler, meta, data

arima, mlp, lstm, scaler, meta, data = load_assets()
TARGET     = meta["target"]
columns    = meta["columns"]
target_idx = columns.index(TARGET)
look_back  = meta["look_back"]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="eyebrow">Model Predictions</div>
    <h1>Manufacturing Sector Forecasting</h1>
    <p>Forecasting Kenya manufacturing sector output · Indicator: Production Index (MT) · Source: KNBS</p>
</div>
""", unsafe_allow_html=True)

# ── Controls (AFTER header, well below navbar) ────────────────────────────────
badge_map = {"ARIMA": "badge-arima", "MLP": "badge-mlp", "LSTM": "badge-lstm"}
ctrl1, ctrl2, ctrl3, _ = st.columns([1, 2, 1.2, 1])
with ctrl1:
    st.markdown('<div class="ctrl-lbl">Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Model", ["ARIMA", "MLP", "LSTM"], label_visibility="collapsed")
with ctrl2:
    st.markdown('<div class="ctrl-lbl">Forecast Horizon (months)</div>', unsafe_allow_html=True)
    forecast_steps = st.slider("Horizon", 1, 36, 12, label_visibility="collapsed")
with ctrl3:
    st.markdown('<div class="ctrl-lbl">History</div>', unsafe_allow_html=True)
    show_all_hist = st.checkbox("Show full history", value=False)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Forecast ──────────────────────────────────────────────────────────────────
def recursive_forecast(model, window, steps, tidx, is_lstm):
    preds = []; w = window.copy()
    for _ in range(steps):
        p = model.predict(w[np.newaxis,:,:] if is_lstm else w.reshape(1,-1), verbose=0)[0,0]
        preds.append(p); w = np.roll(w,-1,axis=0); w[-1,tidx]=p
    return np.array(preds)

def inverse_target(sp):
    d = np.zeros((len(sp),len(columns))); d[:,target_idx]=sp
    return scaler.inverse_transform(d)[:,target_idx]

last_window    = scaler.transform(data[columns].iloc[-look_back:])
last_date      = data.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_steps+1, freq='MS')[1:]

if model_choice == "ARIMA":
    preds = arima.forecast(steps=forecast_steps).values
else:
    preds = inverse_target(recursive_forecast(
        lstm if model_choice=="LSTM" else mlp,
        last_window, forecast_steps, target_idx, model_choice=="LSTM"
    ))

forecast_df = pd.DataFrame({'Predicted (MT)': preds}, index=forecast_dates)
forecast_df.index.name = 'Date'

last_12_avg = data[TARGET].iloc[-12:].mean()
pct_chg     = (preds.mean() - last_12_avg) / last_12_avg * 100
d_cls = "up" if pct_chg > 0 else "down"; d_sign = "+" if pct_chg > 0 else ""

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi"><div class="lbl">Mean Forecast</div><div class="val">{preds.mean():,.0f}</div><div class="delta neutral">MT / month</div></div>
    <div class="kpi"><div class="lbl">Min Projection</div><div class="val">{preds.min():,.0f}</div><div class="delta neutral">MT</div></div>
    <div class="kpi"><div class="lbl">Max Projection</div><div class="val">{preds.max():,.0f}</div><div class="delta neutral">MT</div></div>
    <div class="kpi"><div class="lbl">vs Prior 12-Month Avg</div><div class="val">{d_sign}{pct_chg:.1f}%</div><div class="delta {d_cls}">{d_sign}{preds.mean()-last_12_avg:,.0f} MT</div></div>
    <div class="kpi"><div class="lbl">Horizon</div><div class="val">{forecast_steps}</div><div class="delta neutral">months ahead</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Time Series</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">Historical Data & {model_choice} Projections</div>', unsafe_allow_html=True)

hist_n = len(data) if show_all_hist else min(84, len(data))
NAVY="#0d1b2a"; TEAL="#1a6b72"; GOLD="#c8973a"; LIGHT="#f5f0e8"

fig, ax = plt.subplots(figsize=(13, 5.2))
fig.patch.set_facecolor(LIGHT); ax.set_facecolor(LIGHT)
ax.plot(data.index[-hist_n:], data[TARGET].iloc[-hist_n:], color=NAVY, linewidth=1.8, zorder=3, label="Historical data")
ax.fill_between(data.index[-hist_n:], data[TARGET].iloc[-hist_n:], alpha=0.06, color=NAVY)
ax.plot(forecast_dates, preds, color=GOLD, linewidth=2, zorder=4, label=f"{model_choice} forecast")
ax.scatter(forecast_dates, preds, color=GOLD, s=28, zorder=5)
ax.axvspan(forecast_dates[0], forecast_dates[-1], alpha=0.06, color=GOLD, zorder=1)
ax.axvline(x=last_date, color=TEAL, linewidth=1, linestyle="--", alpha=0.7, zorder=2)
y_min = min(data[TARGET].iloc[-hist_n:].min(), preds.min()) * 0.97
ax.text(last_date, y_min, "  Forecast start", fontsize=8, color=TEAL, va="bottom", fontstyle="italic")
ax.set_xlabel("Date", fontsize=9, color="#5a6475", labelpad=8)
ax.set_ylabel("Manufacturing Output (MT)", fontsize=9, color="#5a6475", labelpad=8)
ax.set_title(f"Kenya Manufacturing Sector Output — {model_choice} Projections", fontsize=11, color=NAVY, fontweight="bold", pad=14)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{x:,.0f}"))
ax.tick_params(axis="both", labelsize=8, colors="#5a6475")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#d6cfc4"); ax.spines["bottom"].set_color("#d6cfc4")
ax.grid(axis="y", color="#d6cfc4", linewidth=0.6, alpha=0.7)
ax.legend(fontsize=8.5, framealpha=0.9, facecolor=LIGHT, edgecolor="#d6cfc4", loc="upper left")
plt.tight_layout(pad=1.5)
st.pyplot(fig); plt.close(fig)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_t, col_p = st.columns([3, 2], gap="large")
with col_t:
    st.markdown('<div class="section-label">Projection Table</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Monthly Forecast Values</div>', unsafe_allow_html=True)
    rows = "".join(f"<tr><td>{d.strftime('%b %Y')}</td><td>{v:,.0f}</td></tr>" for d,v in zip(forecast_dates,preds))
    st.markdown(f"""
<div class="card">
<table class="styled-table">
<thead><tr><th>Month</th><th>Manufacturing Output (MT)</th></tr></thead>
<tbody>{rows}</tbody>
</table></div>
""", unsafe_allow_html=True)
    st.download_button("Download Forecast CSV", forecast_df.to_csv(),
        f"{model_choice}_forecast_{forecast_steps}m.csv", "text/csv")

with col_p:
    st.markdown('<div class="section-label">Model Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Test Set Performance</div>', unsafe_allow_html=True)
    perf = meta.get('model_performance', {})
    if perf:
        ar=perf.get('arima',{}); ml=perf.get('mlp',{}); ls=perf.get('lstm',{})
        st.markdown(f"""
<div class="card card-teal">
<table class="styled-table">
<thead><tr><th>Model</th><th>RMSE</th><th>MAE</th></tr></thead>
<tbody>
<tr><td><span class="model-badge badge-arima">ARIMA</span></td><td style="text-align:right">{ar.get('rmse',0):,.0f}</td><td style="text-align:right">{ar.get('mae',0):,.0f}</td></tr>
<tr><td><span class="model-badge badge-mlp">MLP</span></td><td style="text-align:right">{ml.get('rmse',0):.4f}</td><td style="text-align:right">{ml.get('mae',0):.4f}</td></tr>
<tr><td><span class="model-badge badge-lstm">LSTM</span></td><td style="text-align:right">{ls.get('rmse',0):.4f}</td><td style="text-align:right">{ls.get('mae',0):.4f}</td></tr>
</tbody></table>
<p style="font-size:0.74rem;color:var(--muted);margin:0.75rem 0 0 0;">Manufacturing Output · ARIMA in MT · MLP/LSTM in scaled units · Test: Jul 2024 – Dec 2025</p>
</div>
""", unsafe_allow_html=True)
    st.markdown(f"""
<div class="card card-gold" style="margin-top:1rem;">
    <div style="font-size:0.66rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:0.6rem;">Configuration</div>
    <div style="font-size:0.86rem;line-height:1.85;color:var(--text);">
        <strong>Target:</strong> Manufacturing Output (MT)<br>
        <strong>Features:</strong> {len(columns)} indicators<br>
        <strong>Look-back:</strong> {look_back} months<br>
        <strong>Training:</strong> Jan 2010 – Dec 2023<br>
        <strong>Source:</strong> KNBS Monthly Production Data
    </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.site-footer {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #0d1b2a;
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
    color: #ffffff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.footer-divider {
    width: 1px; height: 18px;
    background: rgba(255,255,255,0.2);
    margin: 0 1rem;
    display: inline-block;
    vertical-align: middle;
}
.footer-right {
    font-size: 0.68rem;
    color: #ffffff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.block-container { padding-bottom: 4rem !important; }
</style>

<div class="site-footer">
    <div>
        <span class="footer-name">Judith Jepkoech</span>
        <span class="footer-divider"></span>
        <span class="footer-institution">Strathmore University &nbsp;&middot;&nbsp; Dissertation Project</span>
    </div>
    <div class="footer-right">Manufacturing Sector Forecasting &nbsp;&middot;&nbsp; 2026</div>
</div>
""", unsafe_allow_html=True)