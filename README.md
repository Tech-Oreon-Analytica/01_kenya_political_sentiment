# 01 — Kenya Political Discourse Sentiment Analysis
### Tech Oreon Analytica | Prototype Demo for Government Stakeholders

---

## Overview

A working sentiment analysis prototype covering Kenyan political and public discourse,
focused on the 2022 general election period (January–August 2022).

Built for rapid delivery to demonstrate analytical capability to a PS-office-level
government audience. Fully transparent about data provenance — uses a clearly labelled
synthetic sample dataset.

---

## What's in this project

| File / Folder | Purpose |
|---|---|
| `app.py` | Streamlit web application (main deliverable) |
| `requirements.txt` | Python dependencies |
| `src/sentiment_pipeline.py` | Run once to generate `data/processed/scored_data.csv` |
| `data/raw/generate_sample_data.py` | Generates the 500-record synthetic sample dataset |
| `data/raw/kenya_political_sample.csv` | Raw synthetic data (auto-generated) |
| `data/processed/scored_data.csv` | Cleaned + VADER-scored data (pipeline output) |
| `report/kenya_sentiment_report.md` | Stakeholder report (markdown → PDF) |

---

## Quick Start

### 1. Install dependencies
```bash
cd 01-kenya-political-sentiment
pip install -r requirements.txt
```

### 2. Generate data and run pipeline
```bash
# Step A: generate the synthetic sample dataset
python data/raw/generate_sample_data.py

# Step B: clean + score it
python src/sentiment_pipeline.py
```
> Note: The app has an auto-generation fallback — if you just run `streamlit run app.py`
> without running steps A/B first, it will trigger them automatically.

### 3. Launch the app
```bash
streamlit run app.py
```
App opens at: http://localhost:8501

---

## Deploy to Streamlit Cloud

1. Push this folder (as root of repo, or a subfolder) to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Set:
   - **Repository**: your GitHub repo URL
   - **Branch**: `main`
   - **Main file path**: `app.py` (or `01-kenya-political-sentiment/app.py` if in subfolder)
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.

> The app's auto-generation fallback means no pre-generated data files are needed
> in the repo — it will generate them on first cold start.

---

## Analytical Stack

| Component | Tool | Rationale |
|---|---|---|
| Sentiment scoring | VADER 3.3+ | Fast, no training required, social-media calibrated |
| Text preprocessing | Custom (regex) | Lightweight, transparent |
| Visualisation | Plotly Express | Interactive, clean, stakeholder-ready |
| App framework | Streamlit | Fastest path from code to deployed demo |

---

## Data Notice

> **This prototype uses a 100% synthetic dataset.** No real tweets, real persons,
> or real events are represented. The dataset mimics the structure and topic distribution
> of 2022 Kenyan political discourse for demonstration purposes only.
> This is clearly disclosed in the application UI and in the report.
