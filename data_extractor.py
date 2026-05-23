import os
import re
import math
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud

from rapidapi_client import RapidApiTwitterClient


class DataExtractor:
    def __init__(self, source_file: str = None, chunksize: int = 100000):
        self.source_file = source_file
        self.data = None
        self.chunksize = chunksize

    def load_data_api(
        self,
        query: str,
        max_results: int = 100,
        output_file: str = "datasets/tweets_from_api.csv",
        search_type: str = "Top",
    ) -> pd.DataFrame:
        """
        Obtiene tweets vía RapidAPI (twitter-api45) y los guarda en CSV.

        Usa el endpoint /search.php con paginación por next_cursor.
        """
        client = RapidApiTwitterClient()
        df = client.search_tweets(
            query=query,
            max_results=max_results,
            search_type=search_type,
        )

        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(output_file, index=False, encoding="utf-8")

        self.source_file = output_file
        self.data = df

        return self.data

    def load_data(self) -> pd.DataFrame:
        if not self.source_file:
            raise ValueError("No se ha indicado ningún archivo de origen.")

        if not os.path.exists(self.source_file):
            raise FileNotFoundError(f"No existe el archivo: {self.source_file}")

        file_ext = os.path.splitext(self.source_file)[1].lower()

        if file_ext == ".csv":
            self.data = self._load_csv()
        elif file_ext == ".json":
            self.data = pd.read_json(self.source_file)
        else:
            raise ValueError("Formato no soportado. Usa .csv o .json")

        return self.data

    def _load_csv(self) -> pd.DataFrame:
        read_kwargs = {
            "encoding": "utf-8",
            "low_memory": False,
        }

        try:
            return pd.read_csv(self.source_file, **read_kwargs)
        except Exception:
            pass

        try:
            read_kwargs_python = {
                k: v for k, v in read_kwargs.items() if k != "low_memory"
            }
            return pd.read_csv(
                self.source_file,
                sep=None,
                engine="python",
                **read_kwargs_python
            )
        except Exception as exc:
            raise ValueError(
                f"No se pudo leer el CSV correctamente: {self.source_file}"
            ) from exc

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
        if self.data is None:
            raise ValueError("Primero debes cargar los datos.")

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
        if self.data is None:
            raise ValueError("Primero debes cargar los datos.")

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
        min_length: int = 3
    ) -> pd.DataFrame:
        if text_series is None:
            if self.data is None:
                raise ValueError("Primero debes cargar los datos.")
            if "text" not in self.data.columns:
                raise ValueError("No existe la columna 'text'.")
            text_series = self.data["text"]

        stop_words = {
            "the", "and", "for", "that", "with", "this", "from", "you", "your",
            "are", "have", "has", "was", "were", "will", "about", "just",
            "they", "them", "what", "when", "where", "which", "while", "into",
            "than", "then", "there", "here", "been", "being", "also", "their",
            "would", "could", "should", "bitcoin", "btc", "https", "co", "amp"
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
            columns=["keyword", "frequency"]
        )

    def generate_hashtag_wordcloud(
        self,
        overall_df: pd.DataFrame = None,
        max_words: int = 100,
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None
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
            collocations=False
        ).generate_from_frequencies(freq_dict)

        plt.figure(figsize=figsize)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Wordcloud de hashtags")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        plt.show()

    def plot_top_hashtags(
        self,
        overall_df: pd.DataFrame = None,
        top_n: int = 10,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
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

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        plt.show()

    def plot_hashtag_trend(
        self,
        hashtag: str,
        by_date_df: pd.DataFrame = None,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
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

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        plt.show()

    def model_topics(self, num_topics: int = 5, passes: int = 10) -> list:
        if self.data is None:
            raise ValueError("Primero debes cargar los datos.")

        self.prepare_clean_text_column()

        nltk.download("stopwords", quiet=True)

        stop_words = set(stopwords.words("english"))
        extra_stopwords = {
            "bitcoin", "btc", "crypto", "https", "amp", "co", "rt", "the",
            "and", "for", "with", "from", "this", "that", "will", "just"
        }
        stop_words.update(extra_stopwords)

        documents = []

        df_sample = self.data.sample(n=min(2000, len(self.data)), random_state=42)
        
        for text in df_sample["clean_text"].fillna(""):
            tokens = [
                token
                for token in text.split()
                if token.isalpha()
                and token not in stop_words
                and len(token) > 2
                and not token.startswith("#")
            ]

            if tokens:
                documents.append(tokens)

        if not documents:
            raise ValueError("No hay documentos suficientes para entrenar LDA.")

        dictionary = corpora.Dictionary(documents)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus_bow = [dictionary.doc2bow(doc) for doc in documents]

        lda_model = LdaModel(
            corpus=corpus_bow,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=passes,
            update_every=1,
            chunksize=100,
            alpha="auto",
            per_word_topics=True,
        )

        topics = []

        for topic_id, topic_words in lda_model.show_topics(
            formatted=False,
            num_words=8
        ):
            words = [word for word, _ in topic_words]
            topics.append({
                "topic_id": topic_id,
                "words": words,
                "words_joined": ", ".join(words),
            })

        return topics

    def analyze_sentiment(self, method: str = "textblob") -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Primero debes cargar los datos.")

        if method != "textblob":
            raise ValueError("En esta implementación se usa method='textblob'.")

        self.prepare_clean_text_column()

        def get_polarity(text: str) -> float:
            return TextBlob(text).sentiment.polarity

        def get_subjectivity(text: str) -> float:
            return TextBlob(text).sentiment.subjectivity

        def get_label(polarity: float) -> str:
            if polarity > 0.05:
                return "positivo"
            if polarity < -0.05:
                return "negativo"
            return "neutro"

        df_sample = self.data.sample(n=5000, random_state=42) if len(self.data) > 5000 else self.data.copy()

        df_sample["sentiment_polarity"] = df_sample["clean_text"].apply(get_polarity)
        df_sample["sentiment_subjectivity"] = df_sample["clean_text"].apply(get_subjectivity)
        df_sample["sentiment_label"] = df_sample["sentiment_polarity"].apply(get_label)

        self.data = df_sample.reset_index(drop=True)

        return self.data[["clean_text", "sentiment_polarity", "sentiment_subjectivity", "sentiment_label"]]

    def plot_sentiment_distribution(
        self,
        figsize: tuple = (8, 5),
        save_path: Optional[str] = None
    ) -> None:
        if self.data is None:
            raise ValueError("Primero debes cargar los datos.")

        if "sentiment_label" not in self.data.columns:
            self.analyze_sentiment()

        sentiment_counts = self.data["sentiment_label"].value_counts()

        plt.figure(figsize=figsize)
        plt.bar(sentiment_counts.index, sentiment_counts.values)
        plt.title("Distribución de sentimientos")
        plt.xlabel("Sentimiento")
        plt.ylabel("Número de tweets")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=150)

        plt.show()

    def parse_sample_texts(
        self,
        sample_size: int = 3,
        model_name: str = "en_core_web_sm"
    ) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("Primero debes cargar los datos.")

        self.prepare_clean_text_column()

        try:
            nlp = spacy.load(model_name)
        except OSError as exc:
            raise OSError(
                f"No se ha encontrado el modelo {model_name}. "
                f"Ejecuta: python -m spacy download {model_name}"
            ) from exc

        sample_texts = (
            self.data["clean_text"]
            .dropna()
            .astype(str)
            .loc[lambda s: s.str.len() > 20]
            .head(sample_size)
            .tolist()
        )

        rows = []

        for text_id, text in enumerate(sample_texts):
            doc = nlp(text)

            for token in doc:
                rows.append({
                    "text_id": text_id,
                    "original_text": text,
                    "token": token.text,
                    "pos": token.pos_,
                    "dep": token.dep_,
                    "head": token.head.text,
                })

        return pd.DataFrame(rows)

    def parse_and_summarize(self, summary_ratio: float = 0.3) -> str:
        if self.data is None:
            raise ValueError("Primero debes cargar los datos.")

        self.prepare_clean_text_column()

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)

        full_text = ". ".join(self.data["clean_text"].dropna().astype(str).tolist())

        if not full_text.strip():
            raise ValueError("No hay texto suficiente para resumir.")

        original_sentences = nltk.sent_tokenize(full_text)

        if not original_sentences:
            raise ValueError("No se han podido separar oraciones.")

        stop_words = set(stopwords.words("english"))
        extra_stopwords = {
            "bitcoin", "btc", "crypto", "https", "amp", "co", "rt"
        }
        stop_words.update(extra_stopwords)

        word_counts = {}

        for sentence in original_sentences:
            cleaned = re.sub(r"[^\w\s]", "", sentence.lower())
            tokens = nltk.word_tokenize(cleaned)

            for token in tokens:
                if token not in stop_words and token.isalpha() and len(token) > 2:
                    word_counts[token] = word_counts.get(token, 0) + 1

        if not word_counts:
            raise ValueError("No hay palabras suficientes para calcular frecuencias.")

        max_freq = max(word_counts.values())

        sentence_scores = {}

        for idx, sentence in enumerate(original_sentences):
            cleaned = re.sub(r"[^\w\s]", "", sentence.lower())
            tokens = nltk.word_tokenize(cleaned)

            score = 0

            for token in tokens:
                if token in word_counts:
                    score += word_counts[token] / max_freq

            sentence_scores[idx] = score

        sorted_sentences = sorted(
            sentence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        total_sentences = len(original_sentences)
        top_n = max(1, math.ceil(total_sentences * summary_ratio))

        selected_idx = [idx for idx, _ in sorted_sentences[:top_n]]
        selected_idx.sort()

        summary_sentences = [original_sentences[i] for i in selected_idx]
        summary = " ".join(summary_sentences)

        return summary

    def save_results(self, output_dir: str = "outputs") -> dict:
        if self.data is None:
            raise ValueError("Primero debes cargar los datos.")

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
        output_dir: str = "outputs"
    ) -> dict:
        os.makedirs(output_dir, exist_ok=True)

        if self.data is None:
            raise ValueError("Primero debes cargar los datos.")

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
            "sentiment_distribution": os.path.join(output_dir, "sentiment_distribution.csv"),
            "parsing_examples": os.path.join(output_dir, "parsing_examples.csv"),
            "summary": os.path.join(output_dir, "summary.txt"),
        }

        topics_df.to_csv(paths["topics"], index=False, encoding="utf-8")
        self.data.to_csv(paths["sentiment_results"], index=False, encoding="utf-8")
        sentiment_distribution.to_csv(
            paths["sentiment_distribution"],
            index=False,
            encoding="utf-8"
        )
        parsing_df.to_csv(paths["parsing_examples"], index=False, encoding="utf-8")

        with open(paths["summary"], "w", encoding="utf-8") as file:
            file.write(summary)

        return paths