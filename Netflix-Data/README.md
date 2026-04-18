# 🎬 Netflix Intelligence System — v2

## 📁 Project Structure

```
Netflix-Data-main/
├── app.py                       ← Streamlit Dashboard (8 tabs)
├── Netflix Data.ipynb           ← Added Chatbot, CV & ML sections
├── netflix1.csv                 ← Dataset (8,790 titles × 10 columns)
├── README.md                    ← This file
│
├── chatbot/                     ← Netflix Q&A Chatbot
│   └── netflix_chatbot.py
│
├── cv_module/                   ← Computer Vision module
│   └── poster_analyser.py
│
├── models/                      ← ML Models
│   └── netflix_models.py        (Duration Predictor + Content Recommender)
│
├── exports/                     ← Saved charts & outputs
└── assets/                      ← Static assets (CSS, logos)
```

## 🚀 How to Run

```bash
pip install streamlit pandas numpy plotly scikit-learn opencv-python pillow matplotlib seaborn wordcloud
streamlit run app.py
```

## 📊 Dashboard Tabs (v2)

| # | Tab | Description |
|---|-----|-------------|
| 1 | 📊 Overview | KPI cards, type split donut, yearly trend, monthly heatmap |
| 2 | 🌍 Geo Analysis | Top countries bar chart, choropleth map, Movie vs TV by country |
| 3 | 🎭 Genre Explorer | Genre bar chart, trend over years, genre word cloud |
| 4 | 🎬 Content Deep-Dive | Top directors, rating pie, duration histogram, seasons dist. |
| 5 | 🔮 ML Predictions | Duration predictor (RF + Ridge), feature importance |
| 6 | 🎯 Recommender | Content-based similar title finder (TF-IDF cosine) |
| 7 | 👁️ CV Analysis | Poster analyser: mood, faces, dominant colours, filter gallery |
| 8 | 💬 Chatbot | Netflix Q&A assistant with quick-question buttons |

## 📓 Notebook Sections (Updated)

| # | Section |
|---|---------|
| 1–12 | Original EDA (cleaning, viz, ML) |
| 13 | 💬 **Chatbot Demo**  |
| 14 | 👁️ **CV Analysis — Poster Analyser**  |
| 15 | 🔮 **ML Models — Duration Predictor & Recommender**  |

## 🔧 Changes
- ✅ Full Streamlit dashboard (`app.py`) created — 8 tabs
- ✅ `chatbot/netflix_chatbot.py` — Rule-based NLP chatbot
- ✅ `cv_module/poster_analyser.py` — OpenCV poster/thumbnail analyser
- ✅ `models/netflix_models.py` — RF + Ridge Duration Predictor + TF-IDF Recommender
- ✅ `chatbot/`, `cv_module/`, `models/`, `exports/`, `assets/`

