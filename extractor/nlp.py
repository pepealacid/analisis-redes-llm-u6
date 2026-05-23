"""Modelado LDA, sentimiento, parsing spaCy y resumen extractivo."""

import math
import re
from typing import Optional

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from textblob import TextBlob

from .base import ExtractorBase


class NlpAnalysisMixin(ExtractorBase):
    """Técnicas de minería de texto sobre la columna clean_text."""

    def model_topics(self, num_topics: int = 5, passes: int = 10) -> list:
        self._require_data()
        self.prepare_clean_text_column()

        nltk.download("stopwords", quiet=True)

        stop_words = set(stopwords.words("english"))
        extra_stopwords = {
            "bitcoin", "btc", "crypto", "https", "amp", "co", "rt", "the",
            "and", "for", "with", "from", "this", "that", "will", "just",
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
            num_words=8,
        ):
            words = [word for word, _ in topic_words]
            topics.append({
                "topic_id": topic_id,
                "words": words,
                "words_joined": ", ".join(words),
            })

        return topics

    def analyze_sentiment(self, method: str = "textblob") -> pd.DataFrame:
        self._require_data()

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

        if len(self.data) > 5000:
            df_sample = self.data.sample(n=5000, random_state=42)
        else:
            df_sample = self.data.copy()

        df_sample["sentiment_polarity"] = df_sample["clean_text"].apply(get_polarity)
        df_sample["sentiment_subjectivity"] = df_sample["clean_text"].apply(
            get_subjectivity
        )
        df_sample["sentiment_label"] = df_sample["sentiment_polarity"].apply(get_label)

        self.data = df_sample.reset_index(drop=True)

        return self.data[
            [
                "clean_text",
                "sentiment_polarity",
                "sentiment_subjectivity",
                "sentiment_label",
            ]
        ]

    def plot_sentiment_distribution(
        self,
        figsize: tuple = (8, 5),
        save_path: Optional[str] = None,
    ) -> None:
        self._require_data()

        if "sentiment_label" not in self.data.columns:
            self.analyze_sentiment()

        sentiment_counts = self.data["sentiment_label"].value_counts()

        plt.figure(figsize=figsize)
        plt.bar(sentiment_counts.index, sentiment_counts.values)
        plt.title("Distribución de sentimientos")
        plt.xlabel("Sentimiento")
        plt.ylabel("Número de tweets")
        plt.tight_layout()

        self._save_current_figure(save_path)
        plt.show()

    def parse_sample_texts(
        self,
        sample_size: int = 3,
        model_name: str = "en_core_web_sm",
    ) -> pd.DataFrame:
        self._require_data()
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
            .loc[lambda series: series.str.len() > 20]
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
        self._require_data()
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
            "bitcoin", "btc", "crypto", "https", "amp", "co", "rt",
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
            key=lambda item: item[1],
            reverse=True,
        )

        total_sentences = len(original_sentences)
        top_n = max(1, math.ceil(total_sentences * summary_ratio))

        selected_idx = [idx for idx, _ in sorted_sentences[:top_n]]
        selected_idx.sort()

        summary_sentences = [original_sentences[i] for i in selected_idx]
        return " ".join(summary_sentences)
