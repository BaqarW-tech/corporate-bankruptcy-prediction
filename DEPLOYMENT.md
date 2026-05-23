# 🚀 Deployment Guide — GitHub + Streamlit Cloud

## Step 1 — Prepare Local Files

Your repo should contain exactly these files:

```
bankruptcy-predictor/
├── app.py
├── bankruptcy_ml_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Step 2 — Initialize Git & Push to GitHub

```bash
# Navigate to project folder
cd bankruptcy-predictor

# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "feat: initial bankruptcy prediction dashboard

- Altman Z-Score baseline (1968)
- XGBoost/LightGBM ensemble (AUC ~0.91)
- GCC Sukuk Islamic finance extension
- 3-screen Streamlit dashboard
- Oil price stress tester"

# Create repo on GitHub (via GitHub CLI)
gh repo create bankruptcy-predictor --public --source=. --remote=origin --push

# OR if you created the repo manually on github.com:
git remote add origin https://github.com/BaqarW-tech/bankruptcy-predictor.git
git branch -M main
git push -u origin main
```

## Step 3 — Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click **"New app"**
3. Select:
   - Repository: `BaqarW-tech/bankruptcy-predictor`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **"Deploy!"**

First deployment takes ~3-5 minutes (installing dependencies).

## Step 4 — Update Your App

```bash
# After making changes locally:
git add .
git commit -m "feat: <describe your change>"
git push

# Streamlit Cloud auto-redeploys on push ✅
```

## Step 5 — Add Streamlit Badge to README

Replace this line in README.md:
```
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
```
With your actual URL after deployment:
```
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://baqarw-tech-bankruptcy-predictor-app-xxxxx.streamlit.app)
```

## Common Deployment Issues

| Error | Fix |
|---|---|
| `ModuleNotFoundError: lightgbm` | Check requirements.txt spelling |
| `DLL load failed` (Windows) | Add `packages.txt` with `libgomp1` |
| Memory limit exceeded | Reduce `n_samples` in `generate_training_data()` |
| Slow startup | Expected — model trains on first load, then caches |

## packages.txt (if needed for LightGBM on Streamlit Cloud)

Create a file called `packages.txt` with:
```
libgomp1
```
