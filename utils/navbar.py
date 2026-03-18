"""
Shared navbar using st.columns + st.button with st.switch_page.
inject_navbar(active_page) at top of each page.
active_page: "home" | "forecasting" | "xai"
"""
import streamlit as st

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root {
    --navy:    #0d1b2a;
    --slate:   #1b2d40;
    --teal:    #1a6b72;
    --gold:    #c8973a;
    --text:    #1a1a2e;
    --muted:   #5a6475;
    --border:  #d6cfc4;
    --card-bg: #fdfcfa;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; color: var(--text); }

/* ── Push content below navbar ── */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 3rem;
    max-width: 1140px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* General text in sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] li {
    color: #c8d0dc;
    font-family: 'Source Sans 3', sans-serif;
}

/* Section headers */
[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1rem !important;
    color: #e8ecf0 !important;
    font-weight: 600 !important;
}

/* Widget labels */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] label {
    color: #7a90a8 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
}

/* Selectbox — the visible selected value box */
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 5px !important;
}
/* Selected text inside the box */
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div[class*="ValueContainer"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div[class*="singleValue"] {
    color: #ffffff !important;
    font-size: 0.88rem !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
/* Dropdown arrow */
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #7a90a8 !important;
}
/* Dropdown menu (the popup list) */
[data-testid="stSidebar"] [data-baseweb="popover"] ul,
[data-testid="stSidebar"] [data-baseweb="menu"] {
    background: #162231 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebar"] [data-baseweb="menu"] li,
[data-testid="stSidebar"] [data-baseweb="menu"] [role="option"] {
    color: #c8d0dc !important;
    font-size: 0.88rem !important;
    background: transparent !important;
}
[data-testid="stSidebar"] [data-baseweb="menu"] li:hover,
[data-testid="stSidebar"] [data-baseweb="menu"] [role="option"]:hover {
    background: rgba(26,107,114,0.3) !important;
    color: #ffffff !important;
}

/* Slider track and thumb */
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div {
    background: rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: var(--teal) !important;
    border-color: var(--teal) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSidebar"] .stSlider p {
    color: #7a90a8 !important;
    font-size: 0.78rem !important;
}

/* Checkbox */
[data-testid="stSidebar"] [data-testid="stCheckbox"] span {
    color: #c8d0dc !important;
    font-size: 0.85rem !important;
}

/* Horizontal rule */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 0.8rem 0 !important;
}

/* Download button in sidebar (if any) */
[data-testid="stSidebar"] .stDownloadButton > button {
    background: rgba(26,107,114,0.4) !important;
    color: #ffffff !important;
    border: 1px solid rgba(26,107,114,0.6) !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
}

/* Sidebar collapse handle */
[data-testid="collapsedControl"] {
    top: 80px !important;
    background: var(--navy) !important;
    border-right: 2px solid var(--teal) !important;
    border-radius: 0 5px 5px 0 !important;
    width: 20px !important;
    z-index: 9999 !important;
    box-shadow: 3px 0 10px rgba(0,0,0,0.3) !important;
    color: #ffffff !important;
}
[data-testid="collapsedControl"]:hover { background: var(--teal) !important; }

/* ── Navbar row container ── */
.stNavbar {
    background: var(--navy) !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    box-shadow: 0 2px 18px rgba(0,0,0,0.35) !important;
    margin: -0.5rem -4rem 1.8rem -4rem !important;
    padding: 0 2rem !important;
    min-height: 62px !important;
}

/* Style ALL columns inside .stNavbar */
.stNavbar > div > div > div[data-testid="stHorizontalBlock"] {
    align-items: center !important;
    min-height: 62px !important;
    gap: 0 !important;
}

/* Brand column (first) */
.nav-brand {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 0 1.5rem 0 0;
    border-right: 1px solid rgba(255,255,255,0.1);
    white-space: nowrap;
}
.nav-brand-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.9rem; font-weight: 700;
    color: #ffffff; letter-spacing: 0.03em; line-height: 1.1;
}
.nav-brand-sub {
    font-size: 0.6rem; letter-spacing: 0.22em;
    text-transform: uppercase; color: var(--gold); font-weight: 600; margin-top: 2px;
}
.nav-tag {
    font-size: 0.65rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #2e4055; white-space: nowrap;
}

/* ── Nav buttons ── */
.stNavbar button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #7a8fa6 !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.13em !important;
    text-transform: uppercase !important;
    padding: 0 1rem !important;
    height: 62px !important;
    min-height: 62px !important;
    line-height: 62px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s !important;
    white-space: nowrap !important;
    width: 100% !important;
    box-shadow: none !important;
}
.stNavbar button:hover {
    color: #ffffff !important;
    background: rgba(255,255,255,0.06) !important;
}
.stNavbar button[kind="primary"] {
    color: #ffffff !important;
    background: rgba(26,107,114,0.28) !important;
    border-bottom: 2px solid var(--teal) !important;
}
.stNavbar button[kind="primary"]:hover {
    background: rgba(26,107,114,0.4) !important;
}
/* Remove Streamlit's button focus outline inside navbar */
.stNavbar button:focus { box-shadow: none !important; outline: none !important; }

/* ── Page components ── */
.page-header { margin-bottom: 2rem; padding-bottom: 1.2rem; border-bottom: 1px solid var(--border); }
.page-header .eyebrow { font-size: 0.68rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--teal); font-weight: 600; margin-bottom: 0.4rem; }
.page-header h1 { font-family: 'Playfair Display', serif; font-size: 1.9rem; color: var(--navy); font-weight: 700; margin: 0; }
.page-header p { font-size: 0.94rem; color: var(--muted); margin: 0.4rem 0 0 0; line-height: 1.55; }

.section-label { font-size: 0.66rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--teal); font-weight: 600; margin-bottom: 0.4rem; }
.section-title { font-family: 'Playfair Display', serif; font-size: 1.25rem; color: var(--navy); margin: 0 0 0.9rem 0; font-weight: 600; }
.divider { height: 1px; background: linear-gradient(to right, var(--border), transparent); margin: 1.8rem 0; }
.card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }
.card-teal { border-top: 3px solid var(--teal); }
.card-gold { border-top: 3px solid var(--gold); }
.card-accent { border-left: 3px solid var(--teal); }

.styled-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.styled-table th { background: var(--navy); color: #c8d0dc; font-size: 0.69rem; letter-spacing: 0.11em; text-transform: uppercase; padding: 0.65rem 0.95rem; text-align: left; font-weight: 600; }
.styled-table td { padding: 0.55rem 0.95rem; border-bottom: 1px solid var(--border); color: var(--text); }
.styled-table tr:last-child td { border-bottom: none; }
.styled-table tr:hover td { background: #f7f3ec; }

.kpi-row { display: flex; gap: 0.85rem; margin: 1.5rem 0; flex-wrap: wrap; }
.kpi { flex: 1; min-width: 130px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 5px; padding: 1rem 1.2rem; }
.kpi .lbl { font-size: 0.65rem; letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin-bottom: 0.25rem; }
.kpi .val { font-family: 'Playfair Display', serif; font-size: 1.35rem; color: var(--navy); font-weight: 600; }
.kpi .delta { font-size: 0.78rem; margin-top: 0.15rem; font-weight: 600; }
.kpi .delta.up { color: #1a7a42; }
.kpi .delta.down { color: #a63020; }
.kpi .delta.neutral { color: var(--teal); }

.model-badge { display: inline-block; padding: 0.18rem 0.65rem; border-radius: 3px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; }
.badge-arima { background: #e8f0f5; color: #1b3a5a; }
.badge-mlp   { background: #e8f5f0; color: #1a4a35; }
.badge-lstm  { background: #f5eee8; color: #4a2a10; }

.notice { background: #f0f6f6; border: 1px solid #b8d9dc; border-left: 3px solid var(--teal); border-radius: 4px; padding: 0.85rem 1.2rem; font-size: 0.87rem; color: var(--slate); margin-top: 1.5rem; }
.metrics-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
.metric-pill { background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 1rem 1.4rem; flex: 1; min-width: 140px; }
.metric-pill .label { font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin-bottom: 0.35rem; }
.metric-pill .value { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: var(--navy); font-weight: 600; }
.metric-pill .sub { font-size: 0.76rem; color: var(--teal); margin-top: 0.15rem; }

.stDownloadButton > button { background: var(--navy) !important; color: #fff !important; border: none !important; border-radius: 4px !important; font-size: 0.8rem !important; letter-spacing: 0.07em !important; padding: 0.45rem 1.1rem !important; font-family: 'Source Sans 3', sans-serif !important; text-transform: uppercase !important; }
.stDownloadButton > button:hover { background: var(--teal) !important; }
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 6px !important; }
</style>
"""

PAGE_FILES = {
    "home":        "app.py",
    "forecasting": "pages/2 Forecasting.py",
    "xai":         "pages/3 Explainable_AI.py",
}

NAV_ITEMS = [
    ("home",        "Overview"),
    ("forecasting", "Forecasting"),
    ("xai",         "Explainable AI"),
]

def inject_navbar(active_page: str = "home"):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # Open navy wrapper div
    st.markdown('<div class="stNavbar">', unsafe_allow_html=True)

    # Brand + nav buttons + tag in one row
    brand_col, *nav_cols, tag_col = st.columns([3, 1.1, 1.1, 1.3, 1.5])

    with brand_col:
        st.markdown("""
<div class="nav-brand">
    <div class="nav-brand-title">MANUFACTURING FORECASTING</div>
    <div class="nav-brand-sub">Kenya &nbsp;·&nbsp; KNBS Production Data</div>
</div>
""", unsafe_allow_html=True)

    for col, (key, label) in zip(nav_cols, NAV_ITEMS):
        with col:
            btn_type = "primary" if key == active_page else "secondary"
            if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                if key != active_page:
                    st.switch_page(PAGE_FILES[key])

    with tag_col:
        st.markdown('<div class="nav-tag" style="text-align:right;padding-right:0.5rem;">Judith Jepkoech Thesis · 2026</div>', unsafe_allow_html=True)

    # Close wrapper
    st.markdown('</div>', unsafe_allow_html=True)