"""
Análisis de redes con NetworkX (Unidad 6).

Métodos previstos:
  - build_interaction_graph(): grafo dirigido de menciones @usuario
  - analyze_network(): métricas, comunidades Louvain y visualización
  - generate_prompt_from_network(): texto estructurado para el LLM
"""

import re

import networkx as nx
import pandas as pd

from .base import ExtractorBase

# Captura el nombre de usuario tras @ (letras, números y guion bajo).
_MENTION_REGEX = re.compile(r"@([\w]+)", flags=re.UNICODE)


class NetworkAnalysisMixin(ExtractorBase):
    """Capacidades de análisis de grafos sobre self.data."""

    def _find_user_mentions(self, text: object) -> list[str]:
        """Devuelve los usuarios mencionados en un tweet, en minúsculas."""
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return []

        return [match.lower() for match in _MENTION_REGEX.findall(str(text))]

    def build_interaction_graph(self) -> nx.DiGraph:
        """
        Construye un grafo dirigido de menciones entre usuarios.

        Cada tweet genera aristas autor → mencionado a partir de la columna
        ``text``. Las menciones duplicadas (mismo par origen-destino) se
        registran una sola vez.

        Returns:
            nx.DiGraph con nodos = usuarios y aristas = interacciones por @.
        """
        self._require_data()

        required = {"user_name", "text"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(
                f"Faltan columnas para el grafo: {sorted(missing)}. "
                f"Disponibles: {list(self.data.columns)}"
            )

        graph = nx.DiGraph()

        for _, row in self.data.iterrows():
            author = str(row["user_name"]).strip().lower()
            if not author:
                continue

            graph.add_node(author)

            for mentioned in self._find_user_mentions(row["text"]):
                if not mentioned or mentioned == author:
                    continue

                graph.add_node(mentioned)

                if not graph.has_edge(author, mentioned):
                    graph.add_edge(author, mentioned)

        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        print(f"Grafo de interacciones: {node_count} nodos, {edge_count} aristas")

        return graph
