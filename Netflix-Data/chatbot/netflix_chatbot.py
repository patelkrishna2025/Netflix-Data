"""
=============================================================
 Netflix Intelligence System
 MODULE: Netflix Q&A Chatbot
 Rule-based NLP chatbot for Netflix dataset exploration
=============================================================
"""

import pandas as pd
import numpy as np
import re


class NetflixChatbot:
    """
    Rule-based chatbot that answers questions about the Netflix dataset.
    Supports queries about titles, genres, directors, ratings, countries,
    year trends, duration, and top-N rankings.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._preprocess()

    def _preprocess(self):
        """Prepare derived columns used in answering queries."""
        df = self.df

        # Date parsing
        if "date_added" in df.columns:
            df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
            df["year_added"]  = df["date_added"].dt.year
            df["month_added"] = df["date_added"].dt.month

        # Duration in minutes (movies)
        if "duration" in df.columns:
            df["duration_minutes"] = (
                df["duration"].str.extract(r"(\d+)").astype(float)
            )

        # Genre list
        if "listed_in" in df.columns:
            df["genres_list"] = df["listed_in"].fillna("").apply(
                lambda x: [g.strip() for g in x.split(",")]
            )

        self.df = df

    # ── Public entry point ──────────────────────────────────────────────────
    def answer(self, question: str) -> str:
        q = question.lower().strip()

        # Greeting
        if any(w in q for w in ["hello", "hi", "hey", "namaste", "hola"]):
            return (
                "👋 Hello! I'm your **Netflix Data Assistant**.\n\n"
                "Try asking:\n"
                "- *How many titles are on Netflix?*\n"
                "- *Top 5 directors?*\n"
                "- *Most popular genres?*\n"
                "- *Movies added in 2020?*\n"
                "- *Average movie duration?*\n"
                "- *Top countries?*\n"
                "- *Search title: Squid Game*"
            )

        # Total count
        if re.search(r"how many (titles|shows|movies|content)", q):
            total    = len(self.df)
            movies   = (self.df["type"] == "Movie").sum()
            tvshows  = (self.df["type"] == "TV Show").sum()
            return (
                f"📊 **Netflix Catalog Summary:**\n"
                f"- Total Titles : **{total:,}**\n"
                f"- Movies       : **{movies:,}**\n"
                f"- TV Shows     : **{tvshows:,}**"
            )

        # Movie vs TV Show split
        if re.search(r"(movie|tv show|split|breakdown|type)", q) and re.search(r"(how many|count|number|percent)", q):
            total   = len(self.df)
            movies  = (self.df["type"] == "Movie").sum()
            tv      = (self.df["type"] == "TV Show").sum()
            return (
                f"🎬 **Content Type Breakdown:**\n"
                f"- Movies  : {movies:,} ({movies/total*100:.1f}%)\n"
                f"- TV Shows: {tv:,} ({tv/total*100:.1f}%)"
            )

        # Top directors
        if re.search(r"(top|best|popular).*(director)", q) or re.search(r"director.*(top|best)", q):
            n   = self._extract_n(q, default=5)
            top = self.df["director"].dropna().value_counts().head(n)
            rows = "\n".join([f"  {i+1}. **{d}** — {c} titles" for i, (d, c) in enumerate(top.items())])
            return f"🎬 **Top {n} Directors:**\n{rows}"

        # Top genres
        if re.search(r"(top|popular|most|genre|category|listed)", q):
            n    = self._extract_n(q, default=5)
            all_genres = self.df["listed_in"].dropna().str.split(",").explode().str.strip()
            top  = all_genres.value_counts().head(n)
            rows = "\n".join([f"  {i+1}. **{g}** — {c} titles" for i, (g, c) in enumerate(top.items())])
            return f"🎭 **Top {n} Genres:**\n{rows}"

        # Top countries
        if re.search(r"(top|country|countries|nation|where)", q):
            n   = self._extract_n(q, default=5)
            top = self.df["country"].dropna().value_counts().head(n)
            rows = "\n".join([f"  {i+1}. **{c}** — {v} titles" for i, (c, v) in enumerate(top.items())])
            return f"🌍 **Top {n} Countries:**\n{rows}"

        # Average movie duration
        if re.search(r"(average|avg|mean|duration|long|length)", q):
            if "duration_minutes" in self.df.columns:
                avg = self.df[self.df["type"] == "Movie"]["duration_minutes"].mean()
                med = self.df[self.df["type"] == "Movie"]["duration_minutes"].median()
                return (
                    f"⏱️ **Movie Duration Stats:**\n"
                    f"- Average: **{avg:.0f} minutes**\n"
                    f"- Median : **{med:.0f} minutes**"
                )

        # Titles added in a specific year
        year_match = re.search(r"\b(19|20)\d{2}\b", q)
        if year_match and re.search(r"(added|released|year|in)", q):
            yr  = int(year_match.group())
            col = "year_added" if "added" in q else "release_year"
            if col in self.df.columns:
                count = (self.df[col] == yr).sum()
                return f"📅 **{yr}:** {count:,} titles were {'added to Netflix' if col == 'year_added' else 'released'}."

        # Ratings
        if re.search(r"(rating|rated|mature|pg|tv-ma|r-rated)", q):
            top = self.df["rating"].dropna().value_counts().head(8)
            rows = "\n".join([f"  {i+1}. **{r}** — {c}" for i, (r, c) in enumerate(top.items())])
            return f"⭐ **Rating Distribution:**\n{rows}"

        # Search for a specific title
        title_match = re.search(r"(search|find|look up|about|tell me about)\s+(?:title[:\s]+)?(.+)", q)
        if title_match:
            keyword = title_match.group(2).strip()
            results = self.df[self.df["title"].str.lower().str.contains(keyword, na=False)]
            if len(results) == 0:
                return f"❌ No titles found matching **'{keyword}'**."
            row = results.iloc[0]
            return (
                f"🎬 **{row['title']}**\n"
                f"- Type    : {row.get('type', 'N/A')}\n"
                f"- Country : {row.get('country', 'N/A')}\n"
                f"- Rating  : {row.get('rating', 'N/A')}\n"
                f"- Duration: {row.get('duration', 'N/A')}\n"
                f"- Release : {row.get('release_year', 'N/A')}\n"
                f"- Genre   : {row.get('listed_in', 'N/A')}\n"
                f"- Added   : {str(row.get('date_added', 'N/A'))[:10]}"
            )

        # Most added month
        if re.search(r"(month|when|seasonal)", q):
            if "month_added" in self.df.columns:
                months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                top_m  = self.df["month_added"].dropna().value_counts().idxmax()
                count  = self.df["month_added"].dropna().value_counts().max()
                return (
                    f"📅 **Peak Month for New Additions:** "
                    f"**{months[int(top_m)-1]}** with {count:,} titles added."
                )

        # Oldest / newest
        if re.search(r"(oldest|earliest|first)", q):
            row = self.df.loc[self.df["release_year"].idxmin()]
            return f"🏛️ **Oldest Title:** *{row['title']}* ({int(row['release_year'])})"

        if re.search(r"(newest|latest|recent)", q):
            row = self.df.loc[self.df["release_year"].idxmax()]
            return f"🆕 **Newest Title:** *{row['title']}* ({int(row['release_year'])})"

        # TV Show seasons
        if re.search(r"(season|tv show|series)", q):
            tv = self.df[self.df["type"] == "TV Show"].copy()
            tv["seasons"] = tv["duration"].str.extract(r"(\d+)").astype(float)
            avg = tv["seasons"].mean()
            most = tv["seasons"].value_counts().idxmax()
            return (
                f"📺 **TV Show Season Stats:**\n"
                f"- Average seasons : **{avg:.1f}**\n"
                f"- Most common     : **{int(most)} Season(s)**"
            )

        # Help / fallback
        return (
            "🤔 I didn't understand that. Try asking:\n"
            "- *How many movies are on Netflix?*\n"
            "- *Top 10 directors?*\n"
            "- *Most popular genres?*\n"
            "- *Top countries?*\n"
            "- *Average movie duration?*\n"
            "- *Titles added in 2020?*\n"
            "- *Search title: Breaking Bad*\n"
            "- *What is the most common rating?*"
        )

    def _extract_n(self, text: str, default: int = 5) -> int:
        """Extract a number from the query, e.g. 'top 10'."""
        m = re.search(r"\b(\d+)\b", text)
        return int(m.group(1)) if m else default
