"""
Clase principal del pipeline: combina carga, análisis de texto y NLP.

La Unidad 6 ampliará esta clase con análisis de redes (ver network.py)
e interpretación con LLM (ver llm.py).
"""

from typing import Optional

import pandas as pd

from .export import ExportMixin
from .io import DataLoadingMixin
from .llm import LlmMixin
from .network import NetworkAnalysisMixin
from .nlp import NlpAnalysisMixin
from .text_analysis import TextAnalysisMixin


class DataExtractor(
    DataLoadingMixin,
    TextAnalysisMixin,
    NlpAnalysisMixin,
    ExportMixin,
    NetworkAnalysisMixin,
    LlmMixin,
):
    """
    Orquestador del análisis de tweets.

    Responsabilidades por capa:
      - DataLoadingMixin: CSV, JSON y RapidAPI
      - TextAnalysisMixin: limpieza, hashtags y visualizaciones
      - NlpAnalysisMixin: LDA, sentimiento, spaCy y resumen
      - ExportMixin: guardado de resultados en outputs/
      - NetworkAnalysisMixin: grafos y métricas (Unidad 6)
      - LlmMixin: interpretación con modelos de lenguaje (Unidad 6)
    """

    def __init__(self, source_file: Optional[str] = None, chunksize: int = 100000):
        self.source_file = source_file
        self.data: Optional[pd.DataFrame] = None
        self.chunksize = chunksize
