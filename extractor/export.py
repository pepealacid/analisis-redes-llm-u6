"""Persistencia de resultados en CSV y ficheros de texto."""

import os

import pandas as pd

from .base import ExtractorBase


class ExportMixin(ExtractorBase):
    """Exportación de análisis básicos y avanzados."""

    def save_results(self, output_dir: str = "outputs") -> dict:
        self._require_data()
        os.makedirs(output_dir, exist_ok=True)

        results = self.analytics_hashtags_extended()

        cleaned_df = self.data.copy()

        if "text" in cleaned_df.columns:
            cleaned_df["clean_text"] = cleaned_df["text"].apply(self.clean_text)
            cleaned_df["hashtags"] = cleaned_df["clean_text"].apply(self.extract_hashtags)

        keywords_df = self.extract_keywords()

        paths = {
            "overall": os.path.join(output_dir, "hashtags_overall.csv"),
            "by_user": os.path.join(output_dir, "hashtags_by_user.csv"),
            "by_date": os.path.join(output_dir, "hashtags_by_date.csv"),
            "keywords": os.path.join(output_dir, "keywords_top.csv"),
            "cleaned_data": os.path.join(output_dir, "cleaned_data.csv"),
        }

        results["overall"].to_csv(paths["overall"], index=False, encoding="utf-8")
        results["by_user"].to_csv(paths["by_user"], index=False, encoding="utf-8")
        results["by_date"].to_csv(paths["by_date"], index=False, encoding="utf-8")
        keywords_df.to_csv(paths["keywords"], index=False, encoding="utf-8")
        cleaned_df.to_csv(paths["cleaned_data"], index=False, encoding="utf-8")

        return paths

    def save_advanced_results(
        self,
        topics: list,
        parsing_df: pd.DataFrame,
        summary: str,
        output_dir: str = "outputs",
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        self._require_data()

        if "sentiment_label" not in self.data.columns:
            self.analyze_sentiment()

        topics_df = pd.DataFrame(topics)
        sentiment_distribution = (
            self.data["sentiment_label"]
            .value_counts()
            .reset_index()
        )
        sentiment_distribution.columns = ["sentiment", "frequency"]

        paths = {
            "topics": os.path.join(output_dir, "topics.csv"),
            "sentiment_results": os.path.join(output_dir, "sentiment_results.csv"),
            "sentiment_distribution": os.path.join(
                output_dir, "sentiment_distribution.csv"
            ),
            "parsing_examples": os.path.join(output_dir, "parsing_examples.csv"),
            "summary": os.path.join(output_dir, "summary.txt"),
        }

        topics_df.to_csv(paths["topics"], index=False, encoding="utf-8")
        self.data.to_csv(paths["sentiment_results"], index=False, encoding="utf-8")
        sentiment_distribution.to_csv(
            paths["sentiment_distribution"],
            index=False,
            encoding="utf-8",
        )
        parsing_df.to_csv(paths["parsing_examples"], index=False, encoding="utf-8")

        with open(paths["summary"], "w", encoding="utf-8") as file:
            file.write(summary)

        return paths
