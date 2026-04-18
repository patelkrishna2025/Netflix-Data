"""
=============================================================
 Netflix Intelligence System
 MODULE: Streamlit Dashboard  (v2)
 Run:  streamlit run app.py
 Tabs:
   1. 📊 Overview          – KPIs, type split, yearly trend
   2. 🌍 Geo Analysis      – Top countries, choropleth map
   3. 🎭 Genre Explorer    – Genre breakdown, heatmap, word cloud
   4. 🎬 Content Deep-Dive – Directors, ratings, duration dist.
   5. 🔮 ML Predictions    – Duration predictor + feature importance
   6. 🎯 Recommender       – Content-based similar titles
   7. 👁️ CV Analysis       – Poster / thumbnail computer vision
   8. 💬 Chatbot           – Netflix Q&A chatbot
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os, io, re
import matplotlib.pyplot as plt
from PIL import Image

# ── Path setup ────────────────────────────────────────────
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from chatbot.netflix_chatbot import NetflixChatbot
from cv_module.poster_analyser import NetflixPosterAnalyser
from models.netflix_models import DurationPredictor, ContentRecommender

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Netflix Intelligence System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── BRAND COLOURS ─────────────────────────────────────────
RED      = "#E50914"   # Netflix red
DARK_RED = "#B20710"
DARK_BG  = "#141414"   # Netflix black
CARD_BG  = "#1F1F1F"
LIGHT    = "#E5E5E5"

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown(f"""
<style>
  .header-bar {{
    background: linear-gradient(90deg, {RED}, {DARK_RED});
    padding: 22px 28px; border-radius: 14px;
    margin-bottom: 22px;
    box-shadow: 0 6px 28px rgba(229,9,20,0.35);
  }}
  .metric-card {{
    background: linear-gradient(135deg, {CARD_BG}, #2a2a2a);
    border: 1px solid #333; border-radius: 12px;
    padding: 18px; text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.5);
  }}
  .metric-value {{ font-size: 2.0rem; font-weight: 800; color: {RED}; }}
  .metric-label {{ font-size: 0.82rem; color: #999; margin-top: 4px; }}
  .cb-user {{
    background: linear-gradient(135deg,{RED},{DARK_RED});
    color: white; padding: 11px 15px;
    border-radius: 18px 18px 4px 18px;
    margin: 7px 0; max-width: 80%; float: right; clear: both;
  }}
  .cb-bot {{
    background: {CARD_BG}; border: 1px solid #333;
    color: #ddd; padding: 11px 15px;
    border-radius: 18px 18px 18px 4px;
    margin: 7px 0; max-width: 80%; float: left; clear: both;
  }}
  .cf {{ clear: both; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════
@st.cache_data(show_spinner="Loading Netflix data …")
def load_data() -> pd.DataFrame:
    candidates = [
        os.path.join(ROOT, "netflix1.csv"),
        os.path.join(ROOT, "data", "netflix1.csv"),
        os.path.join(ROOT, "netflix_titles.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        st.error("❌ netflix1.csv not found. Place it in the project root.")
        st.stop()

    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)

    # Clean whitespace
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

    # Dates
    if "date_added" in df.columns:
        df["date_added"]  = pd.to_datetime(df["date_added"], errors="coerce")
        df["year_added"]  = df["date_added"].dt.year
        df["month_added"] = df["date_added"].dt.month

    # Duration numeric
    if "duration" in df.columns:
        df["duration_minutes"] = df["duration"].str.extract(r"(\d+)").astype(float)

    # Genres list
    if "listed_in" in df.columns:
        df["genres_list"] = df["listed_in"].fillna("").apply(
            lambda x: [g.strip() for g in x.split(",")]
        )

    return df


@st.cache_resource(show_spinner="Training ML models …")
def train_models(df: pd.DataFrame):
    predictor   = DurationPredictor().fit(df)
    recommender = ContentRecommender().fit(df)
    return predictor, recommender


# ═══════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════
def main():
    # Header
    st.markdown("""
    <div class="header-bar">
      <h1 style="color:white;margin:0;font-size:1.9rem;">🎬 Netflix Intelligence System</h1>
      <p style="color:#ffaaaa;margin:6px 0 0 0;font-size:0.95rem;">
        EDA · Geo Analysis · Genre Explorer · ML Predictions · Recommender · CV Analysis · AI Chatbot
      </p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    predictor, recommender = train_models(df)

    # ── Sidebar ──────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=120)
        st.title("⚙️ Controls")
        content_type = st.selectbox("🎬 Content Type", ["All", "Movie", "TV Show"])
        year_min = int(df["release_year"].min())
        year_max = int(df["release_year"].max())
        year_range = st.slider("📅 Release Year", year_min, year_max, (2000, year_max))
        top_n = st.slider("📊 Top N Results", 5, 20, 10)
        st.divider()
        st.caption("🎬 Netflix Intelligence v2")

    # ── Filter ───────────────────────────────────────────
    df_f = df[
        (df["release_year"] >= year_range[0]) &
        (df["release_year"] <= year_range[1])
    ]
    if content_type != "All":
        df_f = df_f[df_f["type"] == content_type]

    # ── Tabs ─────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview",
        "🌍 Geo Analysis",
        "🎭 Genre Explorer",
        "🎬 Content Deep-Dive",
        "🔮 ML Predictions",
        "🎯 Recommender",
        "👁️ CV Analysis",
        "💬 Chatbot",
    ])
    (tab_overview, tab_geo, tab_genre, tab_content,
     tab_ml, tab_rec, tab_cv, tab_chat) = tabs

    # ═══════════════════════════
    #  TAB 1 – OVERVIEW
    # ═══════════════════════════
    with tab_overview:
        total   = len(df_f)
        movies  = (df_f["type"] == "Movie").sum()
        tvshows = (df_f["type"] == "TV Show").sum()
        countries = df_f["country"].dropna().nunique()

        k1, k2, k3, k4 = st.columns(4)
        for col, val, label in [
            (k1, f"{total:,}",    "📺 Total Titles"),
            (k2, f"{movies:,}",   "🎬 Movies"),
            (k3, f"{tvshows:,}",  "📺 TV Shows"),
            (k4, f"{countries}",  "🌍 Countries"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        # Donut – type split
        with c1:
            st.subheader("🎬 Content Type Split")
            type_counts = df_f["type"].value_counts()
            fig_donut = go.Figure(go.Pie(
                labels=type_counts.index, values=type_counts.values,
                hole=0.55,
                marker_colors=[RED, "#444"],
                textinfo="label+percent",
            ))
            fig_donut.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", height=320,
                showlegend=False,
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Line – yearly trend
        with c2:
            st.subheader("📈 Titles Added Per Year")
            if "year_added" in df_f.columns:
                yearly = df_f["year_added"].dropna().value_counts().sort_index()
                fig_line = px.area(
                    x=yearly.index, y=yearly.values,
                    color_discrete_sequence=[RED],
                    template="plotly_dark",
                    labels={"x": "Year", "y": "Titles Added"},
                )
                fig_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320, showlegend=False,
                    margin=dict(t=10, b=10),
                )
                st.plotly_chart(fig_line, use_container_width=True)

        # Monthly heatmap
        st.subheader("📅 Monthly Content Addition Heatmap")
        if "year_added" in df_f.columns and "month_added" in df_f.columns:
            heat = df_f.dropna(subset=["year_added","month_added"])
            heat = heat.groupby(["year_added","month_added"]).size().reset_index(name="count")
            heat_pivot = heat.pivot(index="year_added", columns="month_added", values="count").fillna(0)
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
            fig_heat = px.imshow(
                heat_pivot,
                color_continuous_scale="Reds",
                template="plotly_dark",
                labels={"color": "Titles"},
                aspect="auto",
            )
            fig_heat.update_xaxes(
                tickvals=list(range(1, 13)),
                ticktext=month_names,
            )
            fig_heat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", height=320,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    # ═══════════════════════════
    #  TAB 2 – GEO ANALYSIS
    # ═══════════════════════════
    with tab_geo:
        st.subheader("🌍 Country-wise Content Distribution")

        top_countries = df_f["country"].dropna().value_counts().head(top_n)
        fig_bar = px.bar(
            x=top_countries.values, y=top_countries.index,
            orientation="h",
            color=top_countries.values,
            color_continuous_scale="Reds",
            template="plotly_dark",
            labels={"x": "Titles", "y": "Country"},
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420, coloraxis_showscale=False,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Choropleth
        st.subheader("🗺️ Global Content Map")
        country_counts = df_f["country"].dropna().value_counts().reset_index()
        country_counts.columns = ["country", "count"]
        fig_map = px.choropleth(
            country_counts,
            locations="country",
            locationmode="country names",
            color="count",
            color_continuous_scale="Reds",
            template="plotly_dark",
            hover_name="country",
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False),
            height=480,
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # Type breakdown per country
        st.subheader("🎬 Movie vs TV Show — Top Countries")
        top_c_list = top_countries.index.tolist()[:10]
        type_country = df_f[df_f["country"].isin(top_c_list)].groupby(
            ["country", "type"]
        ).size().reset_index(name="count")
        fig_grouped = px.bar(
            type_country, x="country", y="count", color="type",
            color_discrete_map={"Movie": RED, "TV Show": "#555"},
            barmode="group", template="plotly_dark",
            labels={"count": "Titles", "country": ""},
        )
        fig_grouped.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

    # ═══════════════════════════
    #  TAB 3 – GENRE EXPLORER
    # ═══════════════════════════
    with tab_genre:
        st.subheader("🎭 Genre Analysis")

        all_genres = df_f["listed_in"].dropna().str.split(",").explode().str.strip()
        top_genres = all_genres.value_counts().head(top_n)

        fig_g = px.bar(
            x=top_genres.values, y=top_genres.index,
            orientation="h",
            color=top_genres.values,
            color_continuous_scale="Reds",
            template="plotly_dark",
            labels={"x": "Titles", "y": "Genre"},
        )
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=420, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_g, use_container_width=True)

        # Genre vs Year trend (top-5 genres)
        st.subheader("📈 Genre Trend Over Years")
        if "year_added" in df_f.columns:
            top5_genres = all_genres.value_counts().head(5).index.tolist()
            trend_rows  = []
            for genre in top5_genres:
                sub = df_f[df_f["listed_in"].str.contains(genre, na=False)]
                yr_cnt = sub["year_added"].dropna().value_counts().sort_index()
                for yr, cnt in yr_cnt.items():
                    trend_rows.append({"year": yr, "genre": genre, "count": cnt})
            if trend_rows:
                trend_df = pd.DataFrame(trend_rows)
                fig_trend = px.line(
                    trend_df, x="year", y="count", color="genre",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.sequential.Reds[::-1][:5],
                )
                fig_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_trend, use_container_width=True)

        # Word cloud of genres
        st.subheader("☁️ Genre Word Cloud")
        try:
            from wordcloud import WordCloud
            genre_text = " ".join(all_genres.tolist())
            wc = WordCloud(
                width=900, height=350,
                background_color="#141414",
                colormap="Reds",
                max_words=120,
            ).generate(genre_text)
            fig_wc, ax = plt.subplots(figsize=(12, 4))
            fig_wc.patch.set_facecolor("#141414")
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#141414")
            buf.seek(0)
            st.image(Image.open(buf), use_container_width=True)
            plt.close()
        except ImportError:
            st.info("Install wordcloud: `pip install wordcloud`")

    # ═══════════════════════════
    #  TAB 4 – CONTENT DEEP-DIVE
    # ═══════════════════════════
    with tab_content:
        st.subheader("🎬 Content Deep-Dive")

        c1, c2 = st.columns(2)

        # Top directors
        with c1:
            st.markdown("**🎥 Top Directors**")
            if "director" in df_f.columns:
                top_dir = df_f["director"].dropna().value_counts().head(top_n)
                fig_dir = px.bar(
                    x=top_dir.values, y=top_dir.index,
                    orientation="h",
                    color=top_dir.values,
                    color_continuous_scale="Reds",
                    template="plotly_dark",
                    labels={"x": "Titles", "y": ""},
                )
                fig_dir.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380, coloraxis_showscale=False,
                )
                st.plotly_chart(fig_dir, use_container_width=True)

        # Rating distribution
        with c2:
            st.markdown("**⭐ Rating Distribution**")
            if "rating" in df_f.columns:
                rating_counts = df_f["rating"].dropna().value_counts().head(12)
                fig_rat = px.pie(
                    values=rating_counts.values,
                    names=rating_counts.index,
                    color_discrete_sequence=px.colors.sequential.Reds[::-1],
                    hole=0.4,
                    template="plotly_dark",
                )
                fig_rat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", height=380,
                    margin=dict(t=10, b=10),
                )
                st.plotly_chart(fig_rat, use_container_width=True)

        # Duration histogram
        st.subheader("⏱️ Movie Duration Distribution")
        if "duration_minutes" in df_f.columns:
            movies_dur = df_f[df_f["type"] == "Movie"]["duration_minutes"].dropna()
            fig_dur = px.histogram(
                movies_dur, nbins=40,
                color_discrete_sequence=[RED],
                template="plotly_dark",
                labels={"value": "Duration (minutes)", "count": "Titles"},
            )
            fig_dur.add_vline(
                x=movies_dur.mean(), line_dash="dash",
                line_color="white",
                annotation_text=f"Mean: {movies_dur.mean():.0f} min",
                annotation_position="top right",
            )
            fig_dur.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320, showlegend=False,
            )
            st.plotly_chart(fig_dur, use_container_width=True)

        # TV Show seasons
        st.subheader("📺 TV Show Season Distribution")
        tv_df = df_f[df_f["type"] == "TV Show"].copy()
        tv_df["seasons"] = tv_df["duration"].str.extract(r"(\d+)").astype(float)
        fig_sea = px.histogram(
            tv_df["seasons"].dropna(), nbins=15,
            color_discrete_sequence=[DARK_RED],
            template="plotly_dark",
            labels={"value": "Seasons", "count": "TV Shows"},
        )
        fig_sea.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280, showlegend=False,
        )
        st.plotly_chart(fig_sea, use_container_width=True)

        # Release year trend
        st.subheader("📅 Titles Released Per Year")
        yr_cnt = df_f["release_year"].value_counts().sort_index()
        fig_yr = px.bar(
            x=yr_cnt.index, y=yr_cnt.values,
            color=yr_cnt.values,
            color_continuous_scale="Reds",
            template="plotly_dark",
            labels={"x": "Year", "y": "Titles"},
        )
        fig_yr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_yr, use_container_width=True)

    # ═══════════════════════════
    #  TAB 5 – ML PREDICTIONS
    # ═══════════════════════════
    with tab_ml:
        st.subheader("🔮 Movie Duration Predictor")
        st.info("Ensemble: Random Forest + Ridge Regression trained on Netflix metadata.")

        # Model metrics
        if predictor.metrics:
            m1, m2, m3 = st.columns(3)
            m1.metric("🌲 RF MAE",       f"{predictor.metrics['RF MAE']} min")
            m2.metric("📐 Ridge MAE",    f"{predictor.metrics['Ridge MAE']} min")
            m3.metric("🔗 Ensemble MAE", f"{predictor.metrics['Ensemble MAE']} min")

        st.divider()
        st.subheader("🧮 Predict Duration for a New Movie")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            p_year    = st.slider("Release Year",  1990, 2025, 2020)
            p_country = st.selectbox(
                "Country",
                sorted(df["country"].dropna().apply(lambda x: x.split(",")[0].strip()).unique())
            )
        with col_p2:
            p_rating     = st.selectbox("Rating", df["rating"].dropna().unique().tolist())
            p_genre_cnt  = st.slider("Number of Genres",  1, 5, 2)
            p_country_cnt = st.slider("Number of Countries", 1, 5, 1)

        if st.button("🔮 Predict Duration", type="primary"):
            pred = predictor.predict(p_year, p_country, p_rating, p_genre_cnt, p_country_cnt)
            st.success(f"🎬 Predicted Duration: **{pred} minutes** ({pred/60:.1f} hours)")
            st.balloons()

        # Feature importance
        st.subheader("📊 Feature Importance")
        fi = predictor.feature_importance()
        if not fi.empty:
            fig_fi = px.bar(
                fi, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Reds",
                template="plotly_dark",
            )
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280, coloraxis_showscale=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    # ═══════════════════════════
    #  TAB 6 – RECOMMENDER
    # ═══════════════════════════
    with tab_rec:
        st.subheader("🎯 Content-Based Recommender")
        st.markdown("Find similar Netflix titles using genre + country + rating similarity.")

        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            query_title = st.text_input(
                "🔍 Enter a Netflix title",
                placeholder="e.g. Stranger Things, Squid Game, The Crown …"
            )
        with col_r2:
            n_recs = st.slider("Results", 3, 10, 5)

        if st.button("🎯 Get Recommendations", type="primary") and query_title:
            with st.spinner("Finding similar titles …"):
                recs = recommender.recommend(query_title, n=n_recs)
            if "Error" in recs.columns:
                st.warning(recs.iloc[0]["Error"])
            elif len(recs) == 0:
                st.warning("No recommendations found.")
            else:
                st.success(f"✅ Top {len(recs)} titles similar to **{query_title}**:")
                st.dataframe(recs, use_container_width=True, hide_index=True)

        # Sample titles to try
        st.markdown("**💡 Try these popular titles:**")
        sample_titles = df["title"].dropna().sample(8, random_state=42).tolist()
        cols = st.columns(4)
        for i, t in enumerate(sample_titles):
            if cols[i % 4].button(t[:22], key=f"sample_{i}", use_container_width=True):
                recs = recommender.recommend(t, n=5)
                st.success(f"✅ Similar to **{t}**:")
                st.dataframe(recs, use_container_width=True, hide_index=True)

    # ═══════════════════════════
    #  TAB 7 – CV ANALYSIS
    # ═══════════════════════════
    with tab_cv:
        st.subheader("👁️ Computer Vision — Poster / Thumbnail Analyser")
        st.markdown(
            "Upload any Netflix poster or thumbnail to detect mood, dominant colours, "
            "faces, edge density, and genre hints using OpenCV."
        )

        cv_mode = st.radio(
            "Mode",
            ["🔍 Poster Analysis", "🎨 Filter Gallery"],
            horizontal=True
        )

        uploaded = st.file_uploader(
            "📤 Upload Image (poster / thumbnail / screenshot)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="cv_upload"
        )

        if uploaded:
            try:
                import cv2
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("Could not decode image.")
                else:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    analyser = NetflixPosterAnalyser()

                    if "Poster Analysis" in cv_mode:
                        col_img, col_res = st.columns(2)
                        with col_img:
                            st.markdown("**📸 Original**")
                            st.image(img_rgb, use_container_width=True)

                        result = analyser.analyse(frame)

                        with col_res:
                            st.markdown("**🔎 Analysis Results**")
                            st.metric("Width × Height", f"{result.width} × {result.height} px")
                            st.metric("🌟 Brightness",  f"{result.brightness}/255")
                            st.metric("🔲 Contrast",    f"{result.contrast:.1f}")
                            st.metric("🔍 Edge Density", f"{result.edge_density:.3f}")
                            st.metric("👤 Faces Detected", result.face_count)
                            st.markdown(f"**🎨 Color Mood:** {result.color_mood}")
                            st.markdown(f"**🎬 Genre Hint:** {result.genre_hint}")

                            # Dominant colours
                            st.markdown("**🎨 Dominant Colours:**")
                            hex_cols = result.dominant_colors
                            c1c, c2c, c3c = st.columns(3)
                            for col_widget, hex_val in zip([c1c, c2c, c3c], hex_cols):
                                col_widget.markdown(
                                    f'<div style="background:{hex_val};height:40px;'
                                    f'border-radius:6px;text-align:center;'
                                    f'line-height:40px;color:white;font-size:0.75rem;">'
                                    f'{hex_val}</div>',
                                    unsafe_allow_html=True
                                )

                        # Annotated frame (faces)
                        if result.face_count > 0 and result.annotated_frame is not None:
                            st.markdown("**🟢 Face Detection:**")
                            annotated_rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                            st.image(annotated_rgb, use_container_width=True)

                        # Color histogram
                        st.markdown("### 🌈 RGB Color Histogram")
                        fig_h, ax = plt.subplots(figsize=(10, 3))
                        fig_h.patch.set_facecolor("#1F1F1F")
                        ax.set_facecolor("#1F1F1F")
                        for i, color in enumerate(["red", "green", "blue"]):
                            hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
                            ax.plot(hist, color=color, linewidth=1.5, alpha=0.9)
                        ax.set_title("RGB Color Histogram", color="white")
                        ax.tick_params(colors="white")
                        for sp in ax.spines.values():
                            sp.set_color("#555")
                        buf = io.BytesIO()
                        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#1F1F1F")
                        buf.seek(0)
                        st.image(Image.open(buf), use_container_width=True)
                        plt.close()

                        # Pixel stats
                        st.markdown("### 📊 Pixel Statistics")
                        st.dataframe(
                            pd.DataFrame(analyser.pixel_stats(img_rgb)),
                            use_container_width=True, hide_index=True
                        )

                    else:
                        # Filter Gallery
                        filters_map = analyser.apply_filters(img_rgb)
                        n_cols = 3
                        filter_names = list(filters_map.keys())
                        for row_start in range(0, len(filter_names), n_cols):
                            cols = st.columns(n_cols)
                            for j, name in enumerate(filter_names[row_start:row_start + n_cols]):
                                with cols[j]:
                                    st.markdown(f"**{name}**")
                                    img_out = filters_map[name]
                                    if len(img_out.shape) == 2:
                                        st.image(img_out, use_container_width=True, clamp=True)
                                    else:
                                        st.image(img_out, use_container_width=True)

            except ImportError:
                st.error("OpenCV not installed. Run: `pip install opencv-python`")
        else:
            st.info("📤 Upload a Netflix poster or any image to start analysis.")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🔍 Poster Analysis Capabilities**")
                st.dataframe(pd.DataFrame([
                    {"Feature": "Brightness Analysis",    "Method": "Grayscale Mean"},
                    {"Feature": "Contrast Measurement",   "Method": "Grayscale Std Dev"},
                    {"Feature": "Dominant Colours",       "Method": "K-Means (k=3)"},
                    {"Feature": "Color Mood Detection",   "Method": "HSV Average"},
                    {"Feature": "Face Detection",         "Method": "Haar Cascade"},
                    {"Feature": "Edge / Detail Density",  "Method": "Canny Edge"},
                    {"Feature": "Genre Hint",             "Method": "Rule-based fusion"},
                ]), use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**🎨 Filter Gallery**")
                st.markdown("""
                - 🔳 Grayscale
                - ✏️ Edge Detection (Canny)
                - 🌫️ Gaussian Blur
                - 🔪 Sharpen
                - 🪨 Emboss
                - 🔄 Invert
                - 🟤 Sepia
                - 🔴 Netflix Red Tint
                - ⬆️ High Contrast
                """)

    # ═══════════════════════════
    #  TAB 8 – CHATBOT
    # ═══════════════════════════
    with tab_chat:
        st.subheader("💬 Netflix AI Assistant")
        st.markdown("Ask anything about the Netflix dataset!")

        if "nf_chatbot" not in st.session_state:
            st.session_state.nf_chatbot = NetflixChatbot(df)
        if "nf_chat_history" not in st.session_state:
            st.session_state.nf_chat_history = [
                ("bot", (
                    "👋 Hi! I'm your **Netflix Data Assistant**.\n\n"
                    "Try asking:\n"
                    "- *How many movies are on Netflix?*\n"
                    "- *Top 5 directors?*\n"
                    "- *Most popular genres?*\n"
                    "- *Average movie duration?*\n"
                    "- *Search title: Squid Game*"
                ))
            ]

        # Quick buttons
        st.markdown("**💡 Quick Questions:**")
        quick_qs = [
            "How many titles?",
            "Top 5 directors?",
            "Most popular genres?",
            "Top countries?",
            "Average movie duration?",
            "Titles added in 2020?",
            "Most common rating?",
            "Oldest title?",
        ]
        qcols = st.columns(4)
        for i, q in enumerate(quick_qs):
            if qcols[i % 4].button(q, key=f"nfqb_{i}", use_container_width=True):
                reply = st.session_state.nf_chatbot.answer(q)
                st.session_state.nf_chat_history.append(("user", q))
                st.session_state.nf_chat_history.append(("bot", reply))
                st.rerun()

        st.markdown("---")

        for role, msg in st.session_state.nf_chat_history:
            css = "cb-user" if role == "user" else "cb-bot"
            html_msg = msg.replace("\n", "<br>").replace("**", "<b>", 1)
            st.markdown(
                f'<div class="{css}">{html_msg}</div><div class="cf"></div>',
                unsafe_allow_html=True,
            )

        ci, cb = st.columns([5, 1])
        with ci:
            user_input = st.text_input(
                "Ask …", key="nf_chat_input",
                label_visibility="collapsed",
                placeholder="e.g. Top 10 directors? / Average duration? / Search title: Narcos",
            )
        with cb:
            send = st.button("Send 🚀", use_container_width=True)

        if send and user_input.strip():
            reply = st.session_state.nf_chatbot.answer(user_input)
            st.session_state.nf_chat_history.append(("user", user_input))
            st.session_state.nf_chat_history.append(("bot", reply))
            st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.nf_chat_history = [
                ("bot", "👋 Chat cleared! Ask me about Netflix data.")
            ]
            st.rerun()

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align:center;color:#555;font-size:0.8rem;">
      🎬 Netflix Intelligence System v2 &nbsp;|&nbsp;
      Data: Netflix Catalog &nbsp;|&nbsp;
      Models: RF · Ridge · TF-IDF &nbsp;|&nbsp;
      CV: OpenCV &nbsp;|&nbsp;
      Chatbot: Rule-based NLP
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
