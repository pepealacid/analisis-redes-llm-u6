"""Limpieza de texto, hashtags, palabras clave y gráficos asociados."""

import re
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from .base import ExtractorBase


class TextAnalysisMixin(ExtractorBase):
    """Preprocesado y análisis exploratorio de hashtags."""

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"[^\w\s#]", " ", text, flags=re.UNICODE)
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def prepare_clean_text_column(self) -> pd.DataFrame:
        self._require_data()

        if "text" not in self.data.columns:
            raise ValueError("No existe la columna 'text' en el dataset.")

        self.data["clean_text"] = self.data["text"].apply(self.clean_text)
        return self.data

    def extract_hashtags(self, text: str) -> list:
        if pd.isna(text):
            return []

        text = str(text).lower()
        return re.findall(r"#\w+", text, flags=re.UNICODE)

    def analytics_hashtags_extended(self) -> dict:
        self._require_data()

        df = self.data.copy()

        required_columns = ["text", "date", "user_name"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Faltan columnas requeridas: {missing_columns}. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        df["clean_text"] = df["text"].apply(self.clean_text)
        df["hashtags"] = df["clean_text"].apply(self.extract_hashtags)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].dt.date

        exploded = df.explode("hashtags").copy()
        exploded = exploded.dropna(subset=["hashtags"])
        exploded = exploded[exploded["hashtags"].astype(str).str.strip() != ""]

        overall = exploded["hashtags"].value_counts().reset_index()
        overall.columns = ["hashtag", "frequency"]

        by_user = (
            exploded.groupby(["user_name", "hashtags"])
            .size()
            .reset_index(name="frequency")
            .sort_values(["user_name", "frequency"], ascending=[True, False])
            .reset_index(drop=True)
            .rename(columns={"hashtags": "hashtag"})
        )

        by_date = (
            exploded.groupby(["date", "hashtags"])
            .size()
            .reset_index(name="frequency")
            .sort_values(["date", "frequency"], ascending=[True, False])
            .reset_index(drop=True)
            .rename(columns={"hashtags": "hashtag"})
        )

        return {
            "overall": overall,
            "by_user": by_user,
            "by_date": by_date,
        }

    def extract_keywords(
        self,
        text_series: Optional[pd.Series] = None,
        top_n: int = 30,
        min_length: int = 3,
    ) -> pd.DataFrame:
        if text_series is None:
            self._require_data()
            if "text" not in self.data.columns:
                raise ValueError("No existe la columna 'text'.")
            text_series = self.data["text"]

        stop_words = {
            "the", "and", "for", "that", "with", "this", "from", "you", "your",
            "are", "have", "has", "was", "were", "will", "about", "just",
            "they", "them", "what", "when", "where", "which", "while", "into",
            "than", "then", "there", "here", "been", "being", "also", "their",
            "would", "could", "should", "bitcoin", "btc", "https", "co", "amp",
        }

        tokens = []

        for text in text_series.fillna(""):
            clean = self.clean_text(text)
            words = clean.split()

            for word in words:
                if word.startswith("#"):
                    continue
                if word.isdigit():
                    continue
                if len(word) < min_length:
                    continue
                if word in stop_words:
                    continue
                tokens.append(word)

        counts = Counter(tokens)

        return pd.DataFrame(
            counts.most_common(top_n),
            columns=["keyword", "frequency"],
        )

    def generate_hashtag_wordcloud(
        self,
        overall_df: pd.DataFrame = None,
        max_words: int = 100,
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None,
    ) -> None:
        if overall_df is None:
            results = self.analytics_hashtags_extended()
            overall_df = results["overall"]

        if overall_df.empty:
            raise ValueError("No hay hashtags para generar la wordcloud.")

        freq_dict = dict(zip(overall_df["hashtag"], overall_df["frequency"]))

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            max_words=max_words,
            collocations=False,
        ).generate_from_frequencies(freq_dict)

        plt.figure(figsize=figsize)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Wordcloud de hashtags")

        self._save_current_figure(save_path)
        plt.show()

    def plot_top_hashtags(
        self,
        overall_df: pd.DataFrame = None,
        top_n: int = 10,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> None:
        if overall_df is None:
            results = self.analytics_hashtags_extended()
            overall_df = results["overall"]

        if overall_df.empty:
            raise ValueError("No hay datos para representar hashtags.")

        top_df = overall_df.head(top_n)

        plt.figure(figsize=figsize)
        plt.bar(top_df["hashtag"], top_df["frequency"])
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Top {top_n} hashtags")
        plt.xlabel("Hashtag")
        plt.ylabel("Frecuencia")
        plt.tight_layout()

        self._save_current_figure(save_path)
        plt.show()

    def plot_hashtag_trend(
        self,
        hashtag: str,
        by_date_df: pd.DataFrame = None,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> None:
        if not hashtag.startswith("#"):
            hashtag = f"#{hashtag}"

        hashtag = hashtag.lower()

        if by_date_df is None:
            results = self.analytics_hashtags_extended()
            by_date_df = results["by_date"]

        trend_df = by_date_df[by_date_df["hashtag"] == hashtag].copy()

        if trend_df.empty:
            raise ValueError(f"No hay datos para el hashtag {hashtag}")

        trend_df = trend_df.sort_values("date")

        plt.figure(figsize=figsize)
        plt.plot(trend_df["date"], trend_df["frequency"], marker="o")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Evolución temporal de {hashtag}")
        plt.xlabel("Fecha")
        plt.ylabel("Frecuencia")
        plt.tight_layout()

        self._save_current_figure(save_path)
        plt.show()
