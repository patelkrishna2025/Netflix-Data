"""
=============================================================
 Netflix Intelligence System
 MODULE: ML Models — Duration Predictor & Content Recommender
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  DURATION PREDICTOR (Movies)
# ─────────────────────────────────────────────
class DurationPredictor:
    """
    Predicts movie duration (minutes) from metadata.
    Uses Random Forest + Ridge ensemble.
    """

    def __init__(self):
        self.rf     = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        self.ridge  = Ridge(alpha=10.0)
        self.scaler = StandardScaler()
        self.le_rating  = LabelEncoder()
        self.le_country = LabelEncoder()
        self.trained     = False
        self.feature_cols = []
        self.metrics = {}

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["country_count"] = df["country"].fillna("").apply(
            lambda x: len([c for c in x.split(",") if c.strip()])
        )
        df["genre_count"] = df["listed_in"].fillna("").apply(
            lambda x: len([g for g in x.split(",") if g.strip()])
        )
        df["primary_country"] = df["country"].fillna("Unknown").apply(
            lambda x: x.split(",")[0].strip()
        )
        df["rating_enc"] = self.le_rating.fit_transform(df["rating"].fillna("Unknown"))
        df["country_enc"] = self.le_country.fit_transform(df["primary_country"])
        return df

    def fit(self, df: pd.DataFrame) -> "DurationPredictor":
        movies = df[df["type"] == "Movie"].copy()
        movies["duration_minutes"] = movies["duration"].str.extract(r"(\d+)").astype(float)
        movies = movies.dropna(subset=["duration_minutes"])

        movies = self._build_features(movies)

        self.feature_cols = [
            "release_year", "country_count", "genre_count",
            "rating_enc", "country_enc",
        ]
        self.feature_cols = [c for c in self.feature_cols if c in movies.columns]

        X = movies[self.feature_cols].fillna(0).values
        y = movies["duration_minutes"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        self.rf.fit(X_train, y_train)
        self.ridge.fit(X_train_s, y_train)

        rf_pred    = self.rf.predict(X_test)
        ridge_pred = self.ridge.predict(X_test_s)
        ensemble   = (rf_pred + ridge_pred) / 2

        self.metrics = {
            "RF MAE"       : round(mean_absolute_error(y_test, rf_pred), 2),
            "RF R²"        : round(r2_score(y_test, rf_pred), 4),
            "Ridge MAE"    : round(mean_absolute_error(y_test, ridge_pred), 2),
            "Ridge R²"     : round(r2_score(y_test, ridge_pred), 4),
            "Ensemble MAE" : round(mean_absolute_error(y_test, ensemble), 2),
            "Ensemble R²"  : round(r2_score(y_test, ensemble), 4),
            "Train size"   : len(X_train),
            "Test size"    : len(X_test),
        }
        self.trained = True
        print(f"[DurationPredictor] Ensemble MAE={self.metrics['Ensemble MAE']} "
              f"R²={self.metrics['Ensemble R²']}")
        return self

    def predict(self, release_year: int, country: str = "United States",
                rating: str = "TV-MA", genre_count: int = 2,
                country_count: int = 1) -> float:
        if not self.trained:
            return 90.0
        try:
            country_enc = self.le_country.transform([country])[0]
        except ValueError:
            country_enc = 0
        try:
            rating_enc = self.le_rating.transform([rating])[0]
        except ValueError:
            rating_enc = 0

        row = np.array([[release_year, country_count, genre_count,
                         rating_enc, country_enc]])
        rf_p    = self.rf.predict(row)[0]
        ridge_p = self.ridge.predict(self.scaler.transform(row))[0]
        return round((rf_p + ridge_p) / 2, 1)

    def feature_importance(self) -> pd.DataFrame:
        if not self.trained:
            return pd.DataFrame()
        return pd.DataFrame({
            "Feature"   : self.feature_cols,
            "Importance": self.rf.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
#  CONTENT RECOMMENDER (Cosine similarity on genres)
# ─────────────────────────────────────────────
class ContentRecommender:
    """
    Lightweight content-based recommender using genre + country TF-IDF.
    """

    def __init__(self):
        self.df    : pd.DataFrame | None = None
        self.matrix: np.ndarray  | None = None
        self.trained = False

    def fit(self, df: pd.DataFrame) -> "ContentRecommender":
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.df = df.copy().reset_index(drop=True)
        # Combine genre + country as a "document"
        self.df["doc"] = (
            self.df["listed_in"].fillna("") + " " +
            self.df["country"].fillna("") + " " +
            self.df["rating"].fillna("")
        )
        tfidf = TfidfVectorizer(max_features=300, stop_words="english")
        self.matrix = tfidf.fit_transform(self.df["doc"]).toarray()
        self.trained = True
        print(f"[Recommender] Fitted on {len(self.df)} titles.")
        return self

    def recommend(self, title: str, n: int = 5) -> pd.DataFrame:
        if not self.trained or self.df is None:
            return pd.DataFrame()

        # Find title index
        mask = self.df["title"].str.lower() == title.lower()
        if not mask.any():
            # Fuzzy match
            mask = self.df["title"].str.lower().str.contains(
                title.lower(), na=False
            )
        if not mask.any():
            return pd.DataFrame({"Error": [f"Title '{title}' not found."]})

        idx = mask.idxmax()
        vec = self.matrix[idx]

        # Cosine similarities
        norms    = np.linalg.norm(self.matrix, axis=1) + 1e-9
        sims     = self.matrix @ vec / (norms * np.linalg.norm(vec) + 1e-9)
        sims[idx] = -1  # exclude self

        top_idx = np.argsort(-sims)[:n]
        result  = self.df.iloc[top_idx][
            ["title", "type", "listed_in", "country", "rating", "release_year"]
        ].copy()
        result["similarity"] = sims[top_idx].round(3)
        return result.reset_index(drop=True)
