"""
Clase principal del pipeline: combina carga, análisis de texto y NLP.

La Unidad 6 ampliará esta clase con análisis de redes (ver network.py)
e interpretación con LLM (ver llm.py).
"""

import os
from typing import Any, Callable, Optional

import pandas as pd

from .export import ExportMixin
from .io import DataLoadingMixin
from .llm import DEFAULT_CHAT_LOG, LlmMixin
from .network import DEFAULT_NETWORK_FIGURE, DEFAULT_NETWORK_PROMPT, NetworkAnalysisMixin
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
        self._llm_tokenizer = None
        self._llm_model = None
        self._chat_history: list[dict[str, str]] = []

    def execute_pipeline(
        self,
        output_dir: str = "outputs",
        *,
        run_llm: bool = True,
    ) -> dict[str, Any]:
        """
        Ejecuta el flujo completo de la Unidad 6 sin reprocesar pasos.

        Secuencia:
            load_data -> clean_data -> build_interaction_graph ->
            analyze_network -> generate_prompt_from_network -> chat_local_llm

        Args:
            output_dir: Carpeta donde se guardan figuras, prompt y chat.
            run_llm: Si es True, abre el chat interactivo con Gemma.

        Returns:
            Diccionario con grafo, métricas, prompt y historial de chat.
        """
        os.makedirs(output_dir, exist_ok=True)
        results: dict[str, Any] = {}

        steps: list[tuple[str, Callable[[], Any]]] = [
            ("1. Carga de datos", self._pipeline_load_data),
            ("2. Limpieza de textos", self.clean_data),
            ("3. Grafo de interacciones", self.build_interaction_graph),
            (
                "4. Análisis de red",
                lambda: self.analyze_network(
                    results["graph"],
                    save_path=os.path.join(output_dir, os.path.basename(DEFAULT_NETWORK_FIGURE)),
                ),
            ),
            (
                "5. Prompt para el LLM",
                lambda: self.generate_prompt_from_network(
                    results["graph"],
                    save_path=os.path.join(output_dir, os.path.basename(DEFAULT_NETWORK_PROMPT)),
                ),
            ),
        ]

        for label, step in steps:
            print(f"\n=== {label} ===")
            try:
                output = step()
            except Exception as exc:
                raise RuntimeError(f"Error en '{label}': {exc}") from exc

            if label.startswith("3."):
                results["graph"] = output
            elif label.startswith("4."):
                results["insights"] = output
            elif label.startswith("5."):
                results["prompt"] = output

        if run_llm:
            print("\n=== 6. Chat con LLM local ===")
            try:
                chat_path = os.path.join(output_dir, os.path.basename(DEFAULT_CHAT_LOG))
                results["chat_history"] = self.chat_local_llm(
                    prompt=results["prompt"],
                    save_path=chat_path,
                )
            except Exception as exc:
                raise RuntimeError(f"Error en '6. Chat con LLM local': {exc}") from exc
        else:
            results["chat_history"] = []

        print("\nPipeline de la Unidad 6 completado.")
        print(f"Salidas en: {output_dir}/")

        return results

    def _pipeline_load_data(self) -> pd.DataFrame:
        if self.data is not None:
            print(f"Datos ya cargados ({len(self.data)} filas).")
            return self.data

        if not self.source_file:
            raise ValueError("No hay datos cargados ni archivo de origen definido.")

        df = self.load_data()
        print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
        return df
