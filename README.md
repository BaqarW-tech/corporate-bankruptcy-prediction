# 📊 Corporate Bankruptcy & Sukuk Default Predictor

> **Altman Z-Score Baseline → XGBoost/LightGBM Ensemble → Islamic Finance Extension**  
> A data science portfolio project targeting KSA financial institutions, SAMA, IsDB, and Vision 2030-aligned roles.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange?logo=xgboost)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Project Overview

This project replicates and extends the landmark **Altman (1968) Z-Score** bankruptcy prediction model, then demonstrates how modern machine learning significantly outperforms it — and critically, how the standard model **breaks for Islamic finance instruments** and what to do about it.

### Three-Phase Methodology

| Phase | Approach | AUC-ROC |
|-------|----------|---------|
| **Baseline** | Altman Z-Score (1968) | ~0.72 |
| **ML Models** | XGBoost / LightGBM / Voting Ensemble | **~0.91** |
| **Islamic Finance** | Sukuk-adjusted ratio substitutions | — |

---

## 🏗️ Project Structure

```
bankruptcy-predictor/
├── app.py                      # Streamlit dashboard (3 screens)
├── bankruptcy_ml_pipeline.py   # Full ML pipeline (Colab-ready)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📱 Dashboard — 3 Screens

### Screen 1 — Company Bankruptcy Analyzer
- Input 20+ financial ratios from any company's annual report
- Get Altman Z-Score with zone classification (Safe / Grey / Distress)
- ML ensemble default probability with visual gauge
- Z-Score decomposition bar chart (contribution of each ratio)
- Qualitative interpretation and credit recommendation

### Screen 2 — GCC Sukuk Issuer Screener
- Islamic finance-adjusted ratio inputs (no interest expense)
- Shariah-compliant substitution rationale explained
- XGBoost default probability calibrated on sukuk-like data
- Feature importance chart (oil beta, asset coverage, leverage)
- Automatic warnings for under-collateralisation and leverage concentration

### Screen 3 — Portfolio Stress Tester
- **Upload any CSV portfolio** → instant risk heatmap across all firms
- **Oil price shock simulation** → stress-test GCC portfolios against Brent crude drops
- Identifies newly distressed firms post-shock
- Maps directly to SAMA macroprudential stress testing methodology

---

## 🌙 Why Standard Altman Fails for Sukuk

| Standard Ratio | Problem in Islamic Finance | Replacement Used |
|---|---|---|
| Interest Coverage | Riba — no interest in sukuk | Profit Sharing Ratio |
| Market Cap / Liabilities | SPV structure distorts values | Tangible Asset Coverage |
| Retained Earnings / TA | Reduced by Zakat obligation | Zakat-Adjusted Equity Ratio |
| Debt / Equity | Sukuk ≠ conventional debt | Sukuk to Total Assets |

---

## 📐 Altman Z-Score (1968)

```
Z = 1.2(X₁) + 1.4(X₂) + 3.3(X₃) + 0.6(X₄) + 1.0(X₅)

X₁ = Working Capital / Total Assets        (Liquidity)
X₂ = Retained Earnings / Total Assets      (Cumulative Profitability)
X₃ = EBIT / Total Assets                   (Operating Efficiency)
X₄ = Market Cap / Total Liabilities        (Solvency Buffer)
X₅ = Revenue / Total Assets                (Asset Utilization)

Z > 2.99  →  Safe Zone
1.81–2.99 →  Grey Zone
Z < 1.81  →  Distress Zone
```

---

## 🤖 Machine Learning Pipeline

```python
# Feature Engineering (28 features)
altman_ratios     = [X1, X2, X3, X4, X5]             # Altman core
extended_ratios   = [current_ratio, roa, roe, ...]    # 16 additional
derived_features  = [altman_z_raw, altman_z_sq,       # Meta-features
                     in_distress_zone, liquidity_x_leverage, ...]

# Class Imbalance → SMOTE resampling (~5% bankruptcy rate)
# Models trained with 5-fold stratified CV
# Evaluation: AUC-ROC, AUC-PR, Type I & Type II Error rates
```

### Model Comparison

| Model | AUC-ROC | Notes |
|---|---|---|
| Altman Z-Score | ~0.72 | Classic threshold classifier |
| Logistic Regression | ~0.79 | Interpretable baseline |
| Random Forest | ~0.84 | Non-linear, feature importance |
| XGBoost | ~0.89 | Best standalone model |
| LightGBM | ~0.88 | Fast alternative |
| **Voting Ensemble** | **~0.91** | **Production model** |

---

## 🚀 Quick Start

### Run Locally
```bash
git clone https://github.com/BaqarW-tech/bankruptcy-predictor
cd bankruptcy-predictor
pip install -r requirements.txt
streamlit run app.py
```

### Run in Google Colab
```python
!pip install xgboost lightgbm shap imbalanced-learn optuna -q
exec(open("bankruptcy_ml_pipeline.py").read())
main()
```

### Deploy to Streamlit Cloud
1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Deploy — no environment variables needed

---

## 📊 Portfolio CSV Template

Upload your own portfolio on Screen 3:

```csv
company_name,X1,X2,X3,X4,X5,sector,country
Saudi Aramco,0.18,0.35,0.15,3.20,0.80,Energy,Saudi Arabia
SABIC,0.12,0.22,0.09,1.80,0.75,Chemicals,Saudi Arabia
Al Rajhi Bank,0.05,0.28,0.08,1.50,0.30,Banking,Saudi Arabia
Distressed Co,-0.15,-0.08,-0.03,0.30,0.40,Manufacturing,Bahrain
```

---

## 📚 References

1. **Altman, E.I. (1968).** Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance, 23*(4), 589–609.
2. **Ohlson, J.A. (1980).** Financial ratios and the probabilistic prediction of bankruptcy. *Journal of Accounting Research, 18*(1), 109–131.
3. **Altman, E.I., Iwanicz-Drozdowska, M., et al. (2017).** Financial distress prediction in an international context: A review and empirical analysis of Altman's Z-score model. *Journal of International Financial Management & Accounting, 28*(2), 131–171.
4. **SAMA (2023).** Financial Stability Report. Saudi Central Bank.
5. **IIFM (2023).** Sukuk Report: A Comprehensive Study of the Global Sukuk Market. International Islamic Financial Market.

---

## 👤 Author

**Muhammad Baqar Wagan**  
MA Economics | Data Analytics & AI Portfolio  
Targeting: SAMA · IsDB · Vision 2030 institutions · KSA Financial Sector

[![GitHub](https://img.shields.io/badge/GitHub-BaqarW--tech-black?logo=github)](https://github.com/BaqarW-tech)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
