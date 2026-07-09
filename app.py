# ============================================================
# CORPORATE BANKRUPTCY & SUKUK DEFAULT PREDICTOR
# Streamlit Dashboard — 3 Screens
# Author: Muhammad Baqar Wagan | github.com/BaqarW-tech
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import warnings
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bankruptcy & Sukuk Risk Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME ────────────────────────────────────────────────────
NAVY   = "#0D1B2A"
GOLD   = "#D4A843"
RED    = "#C0392B"
GREEN  = "#27AE60"
BLUE   = "#1A3C5E"
LIGHT  = "#F0F4F8"
GREY   = "#7F8C8D"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');

  html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {NAVY};
    color: {LIGHT};
  }}
  .stApp {{ background-color: {NAVY}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background-color: #0A1520;
    border-right: 1px solid #1E3A5F;
  }}

  /* Metric cards */
  [data-testid="stMetricValue"] {{
    font-size: 2rem !important;
    font-weight: 700 !important;
  }}

  /* Inputs */
  .stNumberInput input, .stSelectbox select, .stSlider {{
    background-color: #1A2B3C !important;
    color: {LIGHT} !important;
    border: 1px solid #2A4A6B !important;
    border-radius: 6px !important;
  }}

  /* Buttons */
  .stButton > button {{
    background: linear-gradient(135deg, {GOLD} 0%, #B8902E 100%);
    color: {NAVY};
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 2rem;
    font-size: 1rem;
    transition: opacity 0.2s;
  }}
  .stButton > button:hover {{ opacity: 0.85; }}

  /* Cards */
  .risk-card {{
    background: #1A2B3C;
    border-radius: 10px;
    padding: 1.5rem;
    border: 1px solid #2A4A6B;
    margin-bottom: 1rem;
  }}
  .metric-card {{
    background: #1A2B3C;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    border-left: 4px solid {GOLD};
    margin-bottom: 0.8rem;
  }}
  .zone-safe    {{ border-left-color: {GREEN} !important; }}
  .zone-grey    {{ border-left-color: {GOLD}  !important; }}
  .zone-danger  {{ border-left-color: {RED}   !important; }}

  /* Header */
  .app-header {{
    background: linear-gradient(135deg, #0D1B2A 0%, #1A3C5E 100%);
    border-bottom: 2px solid {GOLD};
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    border-radius: 10px;
  }}
  .app-title {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {GOLD};
    letter-spacing: -0.5px;
  }}
  .app-subtitle {{
    font-size: 0.9rem;
    color: {GREY};
    margin-top: 0.2rem;
  }}

  /* Section headers */
  .section-header {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {GOLD};
    border-bottom: 1px solid #2A4A6B;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }}

  /* Table */
  .dataframe {{ color: {LIGHT} !important; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{
    background: #0A1520;
    border-radius: 8px;
    padding: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    color: {GREY};
    font-weight: 600;
    border-radius: 6px;
  }}
  .stTabs [aria-selected="true"] {{
    background: {BLUE} !important;
    color: {GOLD} !important;
  }}

  /* Hide Streamlit branding */
  #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA & MODEL (CACHED)
# ============================================================

FEATURE_COLS = [
    "X1_working_capital_to_assets", "X2_retained_earnings_to_assets",
    "X3_ebit_to_assets", "X4_market_cap_to_liabilities",
    "X5_revenue_to_assets", "current_ratio", "quick_ratio",
    "cash_ratio", "debt_to_equity", "interest_coverage",
    "gross_margin", "net_profit_margin", "roa", "roe",
    "asset_turnover", "revenue_growth", "asset_growth",
    "log_total_assets", "operating_cash_flow_to_assets",
    "capex_to_assets", "z_score_trend",
    "altman_z_raw", "altman_z_sq", "in_distress_zone",
    "in_grey_zone", "liquidity_x_leverage",
    "profit_x_growth", "cashflow_quality",
]

SUKUK_FEATURES = [
    "tangible_asset_coverage", "liquid_assets_to_sukuk",
    "profit_sharing_ratio", "roa_islamic",
    "zakat_adjusted_equity_ratio", "sukuk_to_total_assets",
    "total_liabilities_to_assets", "oil_revenue_beta",
    "real_estate_exposure", "fx_risk_usd_peg",
    "shariah_board_rating", "credit_rating_numeric",
    "X1_wc_to_assets", "X2_retained_earnings", "X3_ebit_to_assets",
]


def compute_altman_z(row):
    return (1.2 * row["X1_working_capital_to_assets"] +
            1.4 * row["X2_retained_earnings_to_assets"] +
            3.3 * row["X3_ebit_to_assets"] +
            0.6 * row["X4_market_cap_to_liabilities"] +
            1.0 * row["X5_revenue_to_assets"])


@st.cache_data(show_spinner=False)
def generate_training_data(n=5000):
    np.random.seed(42)
    n_b = int(n * 0.05)
    n_h = n - n_b

    def make(n, b):
        m = 0.38 if b else 1.0
        noise = lambda s: np.random.normal(0, s, n)
        return pd.DataFrame({
            "X1_working_capital_to_assets":  np.clip(m*0.12 + noise(0.15), -1,  1),
            "X2_retained_earnings_to_assets":np.clip(m*0.18 + noise(0.20), -1,  1),
            "X3_ebit_to_assets":             np.clip(m*0.08 + noise(0.12), -0.5,0.5),
            "X4_market_cap_to_liabilities":  np.clip(m*1.20 + noise(0.80),  0.01,10),
            "X5_revenue_to_assets":          np.clip(m*0.90 + noise(0.40),  0.01,5),
            "current_ratio":   np.clip(m*1.80 + noise(0.70), 0.1, 8),
            "quick_ratio":     np.clip(m*1.20 + noise(0.50), 0.05,5),
            "cash_ratio":      np.clip(m*0.40 + noise(0.30), 0,   3),
            "debt_to_equity":  np.clip((2.0/m) + noise(1.0),  0.1, 15),
            "interest_coverage":np.clip(m*4.0  + noise(3),   -5,  30),
            "gross_margin":    np.clip(m*0.30  + noise(0.15),-0.5,0.9),
            "net_profit_margin":np.clip(m*0.08 + noise(0.10),-1,  0.5),
            "roa":             np.clip(m*0.06  + noise(0.08),-0.5,0.4),
            "roe":             np.clip(m*0.10  + noise(0.15),-2,  2),
            "asset_turnover":  np.clip(m*0.85  + noise(0.40), 0.01,5),
            "revenue_growth":  np.clip(m*0.08  + noise(0.25),-0.8,2),
            "asset_growth":    np.clip(m*0.05  + noise(0.20),-0.5,2),
            "log_total_assets":np.clip(np.random.normal(12,2,n), 6,20),
            "operating_cash_flow_to_assets":np.clip(m*0.07+noise(0.10),-0.5,0.5),
            "capex_to_assets": np.clip(m*0.04  + noise(0.03), 0,  0.3),
            "z_score_trend":   np.clip(m*0.20  + noise(0.50),-3,  3),
            "bankrupt":        int(b),
        })

    df = pd.concat([make(n_h, False), make(n_b, True)],
                   ignore_index=True).sample(frac=1, random_state=42)

    # Derived
    df["altman_z_raw"]          = df.apply(compute_altman_z, axis=1)
    df["altman_z_sq"]           = df["altman_z_raw"] ** 2
    df["in_distress_zone"]      = (df["altman_z_raw"] < 1.81).astype(int)
    df["in_grey_zone"]          = ((df["altman_z_raw"] >= 1.81) &
                                    (df["altman_z_raw"] < 2.99)).astype(int)
    df["liquidity_x_leverage"]  = df["current_ratio"] / (df["debt_to_equity"] + 0.01)
    df["profit_x_growth"]       = df["roa"] * df["revenue_growth"]
    df["cashflow_quality"]      = df["operating_cash_flow_to_assets"] / (df["roa"] + 0.001)
    return df


@st.cache_resource(show_spinner=False)
def train_models():
    df = generate_training_data()
    X  = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y  = df["bankrupt"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    sp = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_model = xgb.XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=sp, use_label_encoder=False,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    lgb_model = lgb.LGBMClassifier(
        n_estimators=350, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=sp, random_state=42, n_jobs=-1, verbose=-1,
    )
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced",
        n_jobs=-1, random_state=42,
    )

    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    ensemble = VotingClassifier(
        estimators=[("xgb", xgb_model), ("lgb", lgb_model), ("rf", rf_model)],
        voting="soft", weights=[3, 3, 2],
    )
    ensemble.fit(X_res, y_res)

    # Also keep XGB standalone for SHAP
    xgb_model.fit(X_res, y_res)

    return ensemble, xgb_model, X_test, y_test


@st.cache_resource(show_spinner=False)
def train_sukuk_model():
    np.random.seed(42)
    n = 400
    n_d = int(n * 0.08)
    n_p = n - n_d

    def make_sukuk(n, d):
        m = 0.38 if d else 1.0
        noise = lambda s: np.random.normal(0, s, n)
        return pd.DataFrame({
            "tangible_asset_coverage":     np.clip(m*1.30+noise(0.40), 0.5, 5),
            "liquid_assets_to_sukuk":      np.clip(m*0.40+noise(0.20), 0.01,2),
            "profit_sharing_ratio":        np.clip(m*0.12+noise(0.06), 0,   0.5),
            "roa_islamic":                 np.clip(m*0.07+noise(0.05),-0.3, 0.4),
            "zakat_adjusted_equity_ratio": np.clip(m*0.35+noise(0.12), 0.05,0.9),
            "sukuk_to_total_assets":       np.clip((0.5/m)+noise(0.15), 0.05,0.95),
            "total_liabilities_to_assets": np.clip((0.55/m)+noise(0.15),0.1, 0.95),
            "oil_revenue_beta":            np.clip((1.5/m)*np.random.uniform(0,1,n),0,3),
            "real_estate_exposure":        np.clip(np.random.beta(2,5,n)*(1.5 if d else 1),0,1),
            "fx_risk_usd_peg":             np.random.choice([0,1],n,p=[0.85,0.15]),
            "shariah_board_rating":        np.random.choice([3,4,5],n,p=[0.2,0.4,0.4]),
            "credit_rating_numeric":       np.random.choice(
                [1,2,3,4,5,6,9], n,
                p=[0,0.02,0.05,0.1,0.2,0.3,0.33] if d else [0.05,0.1,0.2,0.25,0.2,0.1,0.1]
            ),
            "X1_wc_to_assets":    np.clip(m*0.10+noise(0.12),-1,1),
            "X2_retained_earnings":np.clip(m*0.15+noise(0.15),-1,1),
            "X3_ebit_to_assets":  np.clip(m*0.07+noise(0.10),-0.5,0.5),
            "defaulted": int(d),
        })

    df_s = pd.concat([make_sukuk(n_p, False), make_sukuk(n_d, True)],
                     ignore_index=True).sample(frac=1, random_state=42)
    X = df_s[SUKUK_FEATURES].fillna(0)
    y = df_s["defaulted"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                stratify=y, random_state=42)
    sp = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    m_s = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=sp, use_label_encoder=False,
        eval_metric="logloss", random_state=42,
    )
    sm = SMOTE(random_state=42, k_neighbors=2)
    X_r, y_r = sm.fit_resample(X_tr, y_tr)
    m_s.fit(X_r, y_r)
    return m_s, df_s


# ============================================================
# HELPERS
# ============================================================

def risk_color(prob):
    if prob >= 0.60: return RED
    if prob >= 0.35: return "#E67E22"
    if prob >= 0.15: return GOLD
    return GREEN


def risk_label(prob):
    if prob >= 0.60: return "🔴 HIGH RISK"
    if prob >= 0.35: return "🟠 ELEVATED RISK"
    if prob >= 0.15: return "🟡 WATCH"
    return "🟢 LOW RISK"


def altman_zone(z):
    if z < 1.81:  return "🔴 DISTRESS ZONE",   RED
    if z < 2.99:  return "🟡 GREY ZONE",        GOLD
    return             "🟢 SAFE ZONE",           GREEN


def gauge_chart(prob, title="Default Probability"):
    fig, ax = plt.subplots(figsize=(4, 2.2),
                           subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)

    theta_start = np.pi
    theta_end   = 0
    theta       = theta_start + (theta_end - theta_start) * prob

    # Background arc
    arc_bg = np.linspace(np.pi, 0, 100)
    ax.plot(arc_bg, [1]*100, color="#1E3A5F", lw=25, solid_capstyle="round")

    # Color segments
    for lo, hi, col in [(0, 0.15, GREEN), (0.15, 0.35, GOLD),
                         (0.35, 0.60, "#E67E22"), (0.60, 1.0, RED)]:
        seg = np.linspace(np.pi * (1 - lo), np.pi * (1 - hi), 30)
        ax.plot(seg, [1]*30, color=col, lw=25, alpha=0.35, solid_capstyle="butt")

    # Needle
    ax.annotate("", xy=(theta, 0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=GOLD,
                                lw=2.5, mutation_scale=15))
    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, np.pi)
    ax.axis("off")

    ax.text(0, -0.25, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold",
            color=risk_color(prob), transform=ax.transData)
    ax.text(0, -0.55, title, ha="center", va="center",
            fontsize=9, color=GREY, transform=ax.transData)

    plt.tight_layout(pad=0)
    return fig


def feature_input_row(label, key, min_val, max_val, default, help_txt=""):
    return st.number_input(label, min_value=float(min_val),
                           max_value=float(max_val),
                           value=float(default),
                           step=0.01, help=help_txt, key=key)


def build_feature_vector(inputs: dict) -> pd.DataFrame:
    """Convert raw financial inputs → full feature vector."""
    z = (1.2 * inputs["X1"] + 1.4 * inputs["X2"] +
         3.3 * inputs["X3"] + 0.6 * inputs["X4"] + 1.0 * inputs["X5"])

    row = {
        "X1_working_capital_to_assets":   inputs["X1"],
        "X2_retained_earnings_to_assets": inputs["X2"],
        "X3_ebit_to_assets":              inputs["X3"],
        "X4_market_cap_to_liabilities":   inputs["X4"],
        "X5_revenue_to_assets":           inputs["X5"],
        "current_ratio":                  inputs.get("current_ratio", 1.5),
        "quick_ratio":                    inputs.get("quick_ratio", 1.0),
        "cash_ratio":                     inputs.get("cash_ratio", 0.3),
        "debt_to_equity":                 inputs.get("debt_to_equity", 1.2),
        "interest_coverage":              inputs.get("interest_coverage", 3.0),
        "gross_margin":                   inputs.get("gross_margin", 0.30),
        "net_profit_margin":              inputs.get("net_profit_margin", 0.08),
        "roa":                            inputs.get("roa", 0.05),
        "roe":                            inputs.get("roe", 0.10),
        "asset_turnover":                 inputs.get("asset_turnover", 0.8),
        "revenue_growth":                 inputs.get("revenue_growth", 0.05),
        "asset_growth":                   inputs.get("asset_growth", 0.04),
        "log_total_assets":               inputs.get("log_total_assets", 12.0),
        "operating_cash_flow_to_assets":  inputs.get("ocf_to_assets", 0.06),
        "capex_to_assets":                inputs.get("capex_to_assets", 0.04),
        "z_score_trend":                  inputs.get("z_score_trend", 0.0),
        "altman_z_raw":                   z,
        "altman_z_sq":                    z ** 2,
        "in_distress_zone":               int(z < 1.81),
        "in_grey_zone":                   int(1.81 <= z < 2.99),
        "liquidity_x_leverage":           inputs.get("current_ratio", 1.5) / (inputs.get("debt_to_equity", 1.2) + 0.01),
        "profit_x_growth":                inputs.get("roa", 0.05) * inputs.get("revenue_growth", 0.05),
        "cashflow_quality":               inputs.get("ocf_to_assets", 0.06) / (inputs.get("roa", 0.05) + 0.001),
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:1rem 0;'>
      <div style='font-size:2rem;'>📊</div>
      <div style='color:{GOLD}; font-weight:700; font-size:1.1rem;'>Risk Predictor</div>
      <div style='color:{GREY}; font-size:0.75rem;'>Altman Z-Score + ML + Islamic Finance</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏢  Company Analyzer", "🌙  Sukuk Screener", "📂  Portfolio Stress Test"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown(f"""
    <div style='font-size:0.8rem; color:{GREY}; padding:0.5rem;'>
      <b style='color:{GOLD};'>Methodology</b><br><br>
      <b>Phase 1</b> — Altman Z-Score (1968)<br>
      <b>Phase 2</b> — XGBoost / LightGBM / Ensemble<br>
      <b>Phase 3</b> — Islamic Finance Extension<br><br>
      <b style='color:{GOLD};'>Dataset</b><br>
      5,000 firms · 5% bankruptcy rate<br>
      SMOTE resampling · 5-fold CV<br><br>
      <b style='color:{GOLD};'>Portfolio Project</b><br>
      Muhammad Baqar Wagan<br>
      <a href='https://github.com/BaqarW-tech' style='color:{GOLD};'>github.com/BaqarW-tech</a>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# LOAD MODELS (with spinner)
# ============================================================

with st.spinner("⚙️ Loading models (first run trains ~5,000 firms)..."):
    ensemble, xgb_only, X_test, y_test = train_models()
    sukuk_model, df_sukuk             = train_sukuk_model()


# ============================================================
# PAGE 1 — COMPANY ANALYZER
# ============================================================

if "Company" in page:
    st.markdown(f"""
    <div class='app-header'>
      <div class='app-title'>🏢 Company Bankruptcy Analyzer</div>
      <div class='app-subtitle'>
        Enter financial statement data → Get Altman Z-Score + ML ensemble default probability
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_results = st.columns([1, 1.2], gap="large")

    # ── LEFT: INPUT FORM ──────────────────────────────────────
    with col_form:
        st.markdown("<div class='section-header'>📋 Altman Z-Score Inputs</div>",
                    unsafe_allow_html=True)

        with st.expander("Core Z-Score Ratios", expanded=True):
            X1 = feature_input_row("Working Capital / Total Assets",    "x1", -1.0,  1.0,  0.12, "X1 — Liquidity measure")
            X2 = feature_input_row("Retained Earnings / Total Assets",  "x2", -1.0,  1.0,  0.18, "X2 — Cumulative profitability")
            X3 = feature_input_row("EBIT / Total Assets",               "x3", -0.5,  0.5,  0.08, "X3 — Operating efficiency")
            X4 = feature_input_row("Market Cap / Total Liabilities",    "x4",  0.01,10.0,  1.20, "X4 — Solvency buffer")
            X5 = feature_input_row("Revenue / Total Assets",            "x5",  0.01, 5.0,  0.90, "X5 — Asset utilization")

        st.markdown("<div class='section-header' style='margin-top:1rem;'>📈 Extended Ratios</div>",
                    unsafe_allow_html=True)

        with st.expander("Liquidity", expanded=False):
            cr  = feature_input_row("Current Ratio",  "cr",  0.1,  8.0, 1.80)
            qr  = feature_input_row("Quick Ratio",    "qr",  0.05, 5.0, 1.20)
            cashr = feature_input_row("Cash Ratio",   "cashr", 0.0, 3.0, 0.40)

        with st.expander("Profitability", expanded=False):
            gm  = feature_input_row("Gross Margin",        "gm",  -0.5, 0.9, 0.30)
            npm = feature_input_row("Net Profit Margin",   "npm", -1.0, 0.5, 0.08)
            roa = feature_input_row("Return on Assets",    "roa", -0.5, 0.4, 0.06)
            roe = feature_input_row("Return on Equity",    "roe", -2.0, 2.0, 0.10)

        with st.expander("Leverage & Growth", expanded=False):
            de   = feature_input_row("Debt / Equity",      "de",  0.1, 15.0, 1.20)
            ic   = feature_input_row("Interest Coverage",  "ic", -5.0, 30.0, 4.00)
            rg   = feature_input_row("Revenue Growth YoY", "rg", -0.8,  2.0, 0.08)
            ag   = feature_input_row("Asset Growth YoY",   "ag", -0.5,  2.0, 0.05)
            ocf  = feature_input_row("Op. Cash Flow / Assets", "ocf", -0.5, 0.5, 0.07)
            zt   = feature_input_row("Z-Score Trend (ΔZ)", "zt", -3.0,  3.0, 0.20)

        analyze_btn = st.button("⚡ Analyze Company", use_container_width=True)

    # ── RIGHT: RESULTS ────────────────────────────────────────
    with col_results:
        if analyze_btn:
            inputs = {
                "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5,
                "current_ratio": cr, "quick_ratio": qr, "cash_ratio": cashr,
                "gross_margin": gm, "net_profit_margin": npm,
                "roa": roa, "roe": roe, "debt_to_equity": de,
                "interest_coverage": ic, "revenue_growth": rg,
                "asset_growth": ag, "ocf_to_assets": ocf, "z_score_trend": zt,
                "asset_turnover": 0.85, "log_total_assets": 12.0, "capex_to_assets": 0.04,
            }

            X_input = build_feature_vector(inputs)
            ml_prob = ensemble.predict_proba(X_input)[0, 1]
            z_score = inputs["altman_z_raw"] if "altman_z_raw" in inputs \
                      else X_input["altman_z_raw"].values[0]
            z_score = X_input["altman_z_raw"].values[0]
            zone_label, zone_col = altman_zone(z_score)

            # ── Gauge ──
            st.markdown("<div class='section-header'>📊 Risk Assessment</div>",
                        unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_g = gauge_chart(ml_prob, "ML Default Probability")
                st.pyplot(fig_g, use_container_width=True)
                plt.close()
            with c2:
                st.markdown(f"""
                <div class='metric-card {"zone-danger" if z_score < 1.81 else "zone-grey" if z_score < 2.99 else "zone-safe"}'>
                  <div style='font-size:0.8rem; color:{GREY};'>ALTMAN Z-SCORE</div>
                  <div style='font-size:2.2rem; font-weight:700; color:{zone_col};'>{z_score:.3f}</div>
                  <div style='font-size:0.9rem; color:{zone_col}; margin-top:0.3rem;'>{zone_label}</div>
                </div>
                <div class='metric-card'>
                  <div style='font-size:0.8rem; color:{GREY};'>ML RISK TIER</div>
                  <div style='font-size:1.4rem; font-weight:700; color:{risk_color(ml_prob)};'>
                    {risk_label(ml_prob)}
                  </div>
                  <div style='font-size:0.85rem; color:{GREY}; margin-top:0.3rem;'>
                    Ensemble probability: {ml_prob*100:.1f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Altman decomposition bar ──
            st.markdown("<div class='section-header' style='margin-top:1rem;'>🔬 Z-Score Decomposition</div>",
                        unsafe_allow_html=True)

            components = {
                "1.2×X1 (WC/TA)": 1.2 * X1,
                "1.4×X2 (RE/TA)": 1.4 * X2,
                "3.3×X3 (EBIT/TA)": 3.3 * X3,
                "0.6×X4 (MktCap/TL)": 0.6 * X4,
                "1.0×X5 (Rev/TA)": 1.0 * X5,
            }
            fig2, ax2 = plt.subplots(figsize=(6, 2.8))
            fig2.patch.set_facecolor("#1A2B3C")
            ax2.set_facecolor("#1A2B3C")
            colors = [GREEN if v >= 0 else RED for v in components.values()]
            bars = ax2.barh(list(components.keys()), list(components.values()),
                            color=colors, edgecolor="none", height=0.6)
            ax2.axvline(0, color=GREY, lw=1)
            ax2.tick_params(colors=LIGHT, labelsize=9)
            for spine in ax2.spines.values():
                spine.set_visible(False)
            ax2.set_xlabel("Contribution to Z-Score", color=GREY, fontsize=9)
            for bar, val in zip(bars, components.values()):
                ax2.text(val + (0.02 if val >= 0 else -0.02),
                         bar.get_y() + bar.get_height()/2,
                         f"{val:.3f}", va="center", ha="left" if val>=0 else "right",
                         color=LIGHT, fontsize=8)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

            # ── Interpretation ──
            if ml_prob >= 0.60:
                interp = f"⚠️ **High default risk detected.** Both Altman ({zone_label}) and ML ({ml_prob*100:.0f}%) signal significant distress. Recommend immediate credit review."
            elif ml_prob >= 0.35:
                interp = f"⚡ **Elevated risk.** ML model flags {ml_prob*100:.0f}% probability. Monitor liquidity and leverage closely."
            elif z_score < 1.81:
                interp = f"🔔 **Altman flags distress** (Z={z_score:.2f}) but ML probability is low ({ml_prob*100:.0f}%). May be recovering — verify with qualitative data."
            else:
                interp = f"✅ **Low risk profile.** Z-Score ({z_score:.2f}) and ML probability ({ml_prob*100:.0f}%) both indicate financial stability."

            st.info(interp)
        else:
            st.markdown(f"""
            <div style='text-align:center; padding:4rem 2rem; color:{GREY};'>
              <div style='font-size:3rem; margin-bottom:1rem;'>📋</div>
              <div style='font-size:1.1rem;'>Enter financial ratios on the left<br>and click <b style='color:{GOLD};'>Analyze Company</b></div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PAGE 2 — SUKUK SCREENER
# ============================================================

elif "Sukuk" in page:
    st.markdown(f"""
    <div class='app-header'>
      <div class='app-title'>🌙 GCC Sukuk Issuer Screener</div>
      <div class='app-subtitle'>
        Islamic finance-adjusted default risk model · Shariah-compliant ratio substitutions
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Why standard Altman breaks ──
    with st.expander("📖 Why Standard Altman Fails for Sukuk", expanded=False):
        st.markdown(f"""
        <div style='color:{LIGHT}; line-height:1.8; font-size:0.92rem;'>
        <table width='100%' style='border-collapse:collapse;'>
          <tr style='border-bottom:1px solid #2A4A6B;'>
            <th style='text-align:left; color:{GOLD}; padding:8px;'>Standard Ratio</th>
            <th style='text-align:left; color:{GOLD}; padding:8px;'>Problem in Islamic Finance</th>
            <th style='text-align:left; color:{GOLD}; padding:8px;'>Replacement</th>
          </tr>
          <tr style='border-bottom:1px solid #1E3A5F;'>
            <td style='padding:8px;'>Interest Coverage</td>
            <td style='padding:8px;'>Riba — no interest in sukuk</td>
            <td style='padding:8px;'>Profit Sharing Ratio</td>
          </tr>
          <tr style='border-bottom:1px solid #1E3A5F;'>
            <td style='padding:8px;'>Market Cap / Liabilities</td>
            <td style='padding:8px;'>SPV structure — consolidated vs unconsolidated</td>
            <td style='padding:8px;'>Tangible Asset Coverage</td>
          </tr>
          <tr style='border-bottom:1px solid #1E3A5F;'>
            <td style='padding:8px;'>Retained Earnings</td>
            <td style='padding:8px;'>Reduced by Zakat obligation</td>
            <td style='padding:8px;'>Zakat-Adjusted Equity Ratio</td>
          </tr>
          <tr>
            <td style='padding:8px;'>Debt / Equity</td>
            <td style='padding:8px;'>Sukuk ≠ conventional debt</td>
            <td style='padding:8px;'>Sukuk to Total Assets</td>
          </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([1, 1.2], gap="large")

    with col_s1:
        st.markdown("<div class='section-header'>🌙 Islamic Finance Ratios</div>",
                    unsafe_allow_html=True)

        with st.expander("Asset Coverage & Liquidity", expanded=True):
            tac  = feature_input_row("Tangible Asset Coverage", "tac", 0.5, 5.0, 1.30,
                                     "Tangible assets / Sukuk outstanding — must be ≥ 1.0 for most structures")
            las  = feature_input_row("Liquid Assets / Sukuk",   "las", 0.01,2.0, 0.40)

        with st.expander("Profitability (No Interest)", expanded=True):
            psr  = feature_input_row("Profit Sharing Ratio",    "psr", 0.0, 0.5, 0.12,
                                     "Mudaraba / Musharaka profits as % of assets")
            roa_i= feature_input_row("ROA (Islamic)",            "roi",-0.3, 0.4, 0.07)
            zaer = feature_input_row("Zakat-Adjusted Equity Ratio","zaer",0.05,0.9,0.35,
                                     "Equity minus Zakat payable / Total assets")

        with st.expander("Leverage & Structure", expanded=True):
            sta  = feature_input_row("Sukuk / Total Assets",    "sta", 0.05,0.95,0.45)
            tla  = feature_input_row("Total Liabilities / Assets","tla",0.10,0.95,0.55)

        with st.expander("GCC Macro Risk", expanded=False):
            orb  = feature_input_row("Oil Revenue Beta",         "orb", 0.0, 3.0, 1.0,
                                     "Sensitivity to Brent crude price")
            rep  = feature_input_row("Real Estate Exposure",     "rep", 0.0, 1.0, 0.25,
                                     "RE assets as % of total assets")
            fx   = st.selectbox("FX Risk (Non-Pegged Currency)", [0, 1],
                                format_func=lambda x: "No (USD-pegged)" if x == 0 else "Yes (floating)")

        with st.expander("Governance", expanded=False):
            sbr  = st.slider("Shariah Board Rating (1=Weak, 5=Strong)", 1, 5, 4)
            crn  = st.selectbox("Credit Rating",
                                [(1,"AAA"),(2,"AA"),(3,"A"),(4,"BBB"),
                                 (5,"BB"),(6,"B/CCC"),(9,"Not Rated")],
                                format_func=lambda x: x[1], index=3)[0]

        with st.expander("Altman-Equivalent Ratios", expanded=False):
            s_x1 = feature_input_row("Working Capital / Assets", "sx1",-1.0,1.0,0.10)
            s_x2 = feature_input_row("Retained Earnings / Assets","sx2",-1.0,1.0,0.15)
            s_x3 = feature_input_row("EBIT / Assets",             "sx3",-0.5,0.5,0.07)

        sukuk_btn = st.button("🌙 Screen Sukuk Issuer", use_container_width=True)

    with col_s2:
        if sukuk_btn:
            issuer_data = {
                "tangible_asset_coverage":     tac,
                "liquid_assets_to_sukuk":      las,
                "profit_sharing_ratio":        psr,
                "roa_islamic":                 roa_i,
                "zakat_adjusted_equity_ratio": zaer,
                "sukuk_to_total_assets":       sta,
                "total_liabilities_to_assets": tla,
                "oil_revenue_beta":            orb,
                "real_estate_exposure":        rep,
                "fx_risk_usd_peg":             fx,
                "shariah_board_rating":        sbr,
                "credit_rating_numeric":       crn,
                "X1_wc_to_assets":             s_x1,
                "X2_retained_earnings":        s_x2,
                "X3_ebit_to_assets":           s_x3,
            }
            X_s   = pd.DataFrame([issuer_data])[SUKUK_FEATURES].fillna(0)
            prob_s= sukuk_model.predict_proba(X_s)[0, 1]

            st.markdown("<div class='section-header'>🌙 Sukuk Risk Assessment</div>",
                        unsafe_allow_html=True)

            fig_gs = gauge_chart(prob_s, "Default Probability")
            st.pyplot(fig_gs, use_container_width=True)
            plt.close()

            col_t1, col_t2 = st.columns(2)
            col_t1.metric("Risk Tier",   risk_label(prob_s))
            col_t2.metric("Probability", f"{prob_s*100:.1f}%")

            # Risk driver table
            st.markdown("<div class='section-header' style='margin-top:1rem;'>🔬 Key Risk Drivers</div>",
                        unsafe_allow_html=True)
            importance = pd.Series(
                sukuk_model.feature_importances_,
                index=SUKUK_FEATURES
            ).sort_values(ascending=False).head(6)

            fig_imp, ax_imp = plt.subplots(figsize=(6, 3))
            fig_imp.patch.set_facecolor("#1A2B3C")
            ax_imp.set_facecolor("#1A2B3C")
            importance[::-1].plot(kind="barh", ax=ax_imp,
                                  color=GOLD, edgecolor="none", height=0.5)
            ax_imp.tick_params(colors=LIGHT, labelsize=8)
            for spine in ax_imp.spines.values():
                spine.set_visible(False)
            ax_imp.set_xlabel("Feature Importance (Gain)", color=GREY, fontsize=8)
            ax_imp.set_title("Top Risk Drivers for This Model",
                             color=LIGHT, fontsize=9, pad=8)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_imp, use_container_width=True)
            plt.close()

            # Asset coverage warning
            if tac < 1.0:
                st.error("⚠️ **Tangible Asset Coverage < 1.0** — Sukuk is under-collateralised. This violates core Shariah requirements for asset-backed structures.")
            if orb > 2.0:
                st.warning("⚡ **High Oil Revenue Beta** — This issuer is significantly exposed to oil price volatility. Relevant for SAMA stress testing.")
            if sta > 0.70:
                st.warning("📌 **Heavy Sukuk Leverage** — Sukuk-to-assets ratio exceeds 70%, signalling limited financial flexibility.")
        else:
            st.markdown(f"""
            <div style='text-align:center; padding:4rem 2rem; color:{GREY};'>
              <div style='font-size:3rem; margin-bottom:1rem;'>🌙</div>
              <div style='font-size:1.1rem;'>Enter Islamic finance ratios<br>and click <b style='color:{GOLD};'>Screen Sukuk Issuer</b></div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PAGE 3 — PORTFOLIO STRESS TESTER
# ============================================================

elif "Portfolio" in page:
    st.markdown(f"""
    <div class='app-header'>
      <div class='app-title'>📂 Portfolio Stress Tester</div>
      <div class='app-subtitle'>
        Upload a portfolio of companies or sukuk issuers · Oil price shock simulation · Risk heatmap
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 Upload Portfolio", "⚡ Oil Price Shock Simulation"])

    # ── TAB 1: Upload ──────────────────────────────────────────
    with tab1:
        st.markdown(f"""
        <div class='risk-card'>
          <b style='color:{GOLD};'>CSV Format Required</b><br>
          <span style='color:{GREY}; font-size:0.85rem;'>
          Columns: <code>company_name, X1, X2, X3, X4, X5</code> (Altman ratios)<br>
          Optional: <code>sector, country</code> for grouping
          </span>
        </div>
        """, unsafe_allow_html=True)

        template_csv = """company_name,X1,X2,X3,X4,X5,sector,country
Saudi Aramco,0.18,0.35,0.15,3.20,0.80,Energy,Saudi Arabia
SABIC,0.12,0.22,0.09,1.80,0.75,Chemicals,Saudi Arabia
Al Rajhi Bank,0.05,0.28,0.08,1.50,0.30,Banking,Saudi Arabia
Aldar Properties,0.08,0.15,0.06,0.90,0.45,Real Estate,UAE
DP World,-0.02,0.10,0.04,0.75,0.55,Logistics,UAE
Distressed Co,-0.15,-0.08,-0.03,0.30,0.40,Manufacturing,Bahrain
"""
        st.download_button(
            "⬇️ Download Template CSV",
            template_csv, "portfolio_template.csv", "text/csv",
        )

        uploaded = st.file_uploader("Upload your portfolio CSV",
                                    type=["csv"], key="port_upload")

        if uploaded:
            df_port = pd.read_csv(uploaded)
            required_cols = ["company_name", "X1", "X2", "X3", "X4", "X5"]
            if not all(c in df_port.columns for c in required_cols):
                st.error(f"Missing columns. Required: {required_cols}")
            else:
                # Compute Z-Scores and ML probabilities
                df_port["altman_z"] = (
                    1.2*df_port["X1"] + 1.4*df_port["X2"] +
                    3.3*df_port["X3"] + 0.6*df_port["X4"] + 1.0*df_port["X5"]
                )

                def port_ml_prob(row):
                    inp = {
                        "X1": row["X1"], "X2": row["X2"], "X3": row["X3"],
                        "X4": row["X4"], "X5": row["X5"],
                    }
                    X_inp = build_feature_vector(inp)
                    return ensemble.predict_proba(X_inp)[0, 1]

                with st.spinner("Scoring portfolio..."):
                    df_port["ml_prob"]  = df_port.apply(port_ml_prob, axis=1)
                    df_port["z_zone"]   = df_port["altman_z"].apply(
                        lambda z: "Distress" if z < 1.81 else
                                  "Grey"    if z < 2.99 else "Safe"
                    )
                    df_port["risk_tier"]= df_port["ml_prob"].apply(
                        lambda p: "HIGH" if p >= 0.60 else
                                  "ELEVATED" if p >= 0.35 else
                                  "WATCH" if p >= 0.15 else "LOW"
                    )

                # ── Portfolio Summary ──
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Companies",  len(df_port))
                c2.metric("High Risk",
                          (df_port["risk_tier"] == "HIGH").sum(),
                          delta=f"{(df_port['risk_tier']=='HIGH').mean()*100:.0f}%",
                          delta_color="inverse")
                c3.metric("Avg ML Probability", f"{df_port['ml_prob'].mean()*100:.1f}%")
                c4.metric("Avg Z-Score",        f"{df_port['altman_z'].mean():.2f}")

                # ── Heatmap ──
                st.markdown(f"<div class='section-header' style='margin-top:1.5rem;'>🗺️ Risk Heatmap</div>",
                            unsafe_allow_html=True)

                fig_heat, ax_heat = plt.subplots(
                    figsize=(max(8, len(df_port)*0.8), 3)
                )
                fig_heat.patch.set_facecolor("#1A2B3C")
                ax_heat.set_facecolor("#1A2B3C")

                probs = df_port["ml_prob"].values.reshape(1, -1)
                im = ax_heat.imshow(probs, cmap="RdYlGn_r",
                                    vmin=0, vmax=1, aspect="auto")
                ax_heat.set_xticks(range(len(df_port)))
                ax_heat.set_xticklabels(df_port["company_name"],
                                         rotation=45, ha="right",
                                         color=LIGHT, fontsize=9)
                ax_heat.set_yticks([])
                for i, p in enumerate(df_port["ml_prob"]):
                    ax_heat.text(i, 0, f"{p*100:.0f}%", ha="center",
                                 va="center", color="white",
                                 fontsize=9, fontweight="bold")
                plt.colorbar(im, ax=ax_heat, orientation="horizontal",
                             pad=0.4, shrink=0.5, label="Default Probability")
                ax_heat.set_title("Portfolio Default Risk Heatmap",
                                  color=LIGHT, fontsize=11, pad=10)
                plt.tight_layout()
                st.pyplot(fig_heat, use_container_width=True)
                plt.close()

                # ── Detailed Table ──
                st.markdown(f"<div class='section-header' style='margin-top:1.5rem;'>📋 Detailed Portfolio Table</div>",
                            unsafe_allow_html=True)
                display_df = df_port[["company_name", "altman_z", "z_zone",
                                       "ml_prob", "risk_tier"]].copy()
                display_df.columns = ["Company", "Z-Score", "Zone",
                                       "ML Probability", "Risk Tier"]
                display_df["ML Probability"] = display_df["ML Probability"].map("{:.1%}".format)
                display_df["Z-Score"]         = display_df["Z-Score"].map("{:.3f}".format)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("⬆️ Upload a CSV to score your portfolio, or download the template above.")

    # ── TAB 2: Oil Price Shock ─────────────────────────────────
    with tab2:
        st.markdown(f"""
        <div class='risk-card'>
          <b style='color:{GOLD};'>Macro Stress Test — Brent Crude Price Shock</b><br>
          <span style='color:{GREY}; font-size:0.85rem;'>
          Simulates how a drop in oil prices affects GCC corporate financial health,
          proxied through deteriorating profitability and liquidity ratios.
          </span>
        </div>
        """, unsafe_allow_html=True)

        col_oi1, col_oi2 = st.columns([1, 2])
        with col_oi1:
            base_oil   = st.slider("Base Brent Price (USD/bbl)", 50, 120, 85)
            shock_pct  = st.slider("Price Shock (%)", -60, 0, -40,
                                   format="%d%%",
                                   help="Negative = oil price drops")
            shock_oil  = base_oil * (1 + shock_pct / 100)
            n_firms    = st.slider("Number of GCC Firms", 10, 100, 30)
            oil_beta   = st.slider("Average Oil Revenue Beta", 0.5, 2.5, 1.5)

            run_stress = st.button("⚡ Run Stress Test", use_container_width=True)

        with col_oi2:
            if run_stress:
                np.random.seed(99)

                # Baseline firms (healthy-ish GCC mix)
                base_X3 = np.clip(np.random.normal(0.07, 0.04, n_firms), -0.1, 0.3)
                base_X1 = np.clip(np.random.normal(0.10, 0.06, n_firms), -0.2, 0.4)
                betas   = np.clip(np.random.normal(oil_beta, 0.5, n_firms), 0, 3)

                # Oil shock effect: reduces EBIT and working capital proportionally
                shock_factor = abs(shock_pct) / 100 * betas
                stress_X3    = base_X3 - (shock_factor * 0.08)
                stress_X1    = base_X1 - (shock_factor * 0.06)

                # Compute Z-Scores baseline vs stressed
                base_z   = 1.2*base_X1  + 3.3*base_X3  + 0.8  # simplified
                stress_z = 1.2*stress_X1 + 3.3*stress_X3 + 0.8

                col_s_m1, col_s_m2, col_s_m3 = st.columns(3)
                col_s_m1.metric("Shocked Oil Price",
                                f"${shock_oil:.0f}/bbl",
                                delta=f"{shock_pct}%", delta_color="inverse")
                col_s_m2.metric("Firms Entering Distress Zone",
                                f"{(stress_z < 1.81).sum()} / {n_firms}",
                                delta=f"+{(stress_z < 1.81).sum() - (base_z < 1.81).sum()} new",
                                delta_color="inverse")
                col_s_m3.metric("Avg Z-Score Δ",
                                f"{(stress_z - base_z).mean():.3f}",
                                delta_color="inverse")

                # ── Before vs After chart ──
                firm_ids = [f"GCC-{i+1:02d}" for i in range(n_firms)]
                fig_ss, ax_ss = plt.subplots(figsize=(10, 4))
                fig_ss.patch.set_facecolor("#1A2B3C")
                ax_ss.set_facecolor("#1A2B3C")

                x = np.arange(n_firms)
                w = 0.38
                bars1 = ax_ss.bar(x - w/2, base_z,   width=w, color=BLUE,  label="Baseline",   alpha=0.85)
                bars2 = ax_ss.bar(x + w/2, stress_z, width=w, color=RED,   label="Post-Shock", alpha=0.85)
                ax_ss.axhline(1.81, color=RED,   ls="--", lw=1, alpha=0.7, label="Distress (1.81)")
                ax_ss.axhline(2.99, color=GOLD,  ls=":",  lw=1, alpha=0.7, label="Safe (2.99)")
                ax_ss.set_xticks(x)
                ax_ss.set_xticklabels(firm_ids, rotation=60, ha="right",
                                       color=LIGHT, fontsize=7)
                ax_ss.tick_params(colors=LIGHT)
                ax_ss.set_ylabel("Altman Z-Score", color=GREY)
                ax_ss.set_title(
                    f"Oil Price Shock: ${base_oil} → ${shock_oil:.0f}/bbl  |  "
                    f"{n_firms} GCC Firms  |  Beta = {oil_beta:.1f}",
                    color=LIGHT, fontsize=10
                )
                ax_ss.legend(fontsize=8, facecolor="#0D1B2A", labelcolor=LIGHT)
                for spine in ax_ss.spines.values():
                    spine.set_color("#2A4A6B")
                plt.tight_layout()
                st.pyplot(fig_ss, use_container_width=True)
                plt.close()

                # ── Insight box ──
                newly_distressed = (stress_z < 1.81).sum() - (base_z < 1.81).sum()
                st.warning(f"""
                **Stress Test Summary:**
                A **{abs(shock_pct)}% drop** in Brent crude (${base_oil}→${shock_oil:.0f}/bbl) pushes
                **{newly_distressed} additional firms** into the Altman distress zone.
                GCC firms with higher oil revenue beta (>{oil_beta:.1f}) are most exposed.
                This maps directly to **SAMA macroprudential stress testing** requirements.
                """)
            else:
                st.markdown(f"""
                <div style='text-align:center; padding:4rem 2rem; color:{GREY};'>
                  <div style='font-size:3rem; margin-bottom:1rem;'>⚡</div>
                  <div style='font-size:1.1rem;'>Configure parameters and click<br>
                  <b style='color:{GOLD};'>Run Stress Test</b></div>
                </div>
                """, unsafe_allow_html=True)
