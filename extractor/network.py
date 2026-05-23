"""
Análisis de redes con NetworkX (Unidad 6).

Métodos previstos:
  - build_interaction_graph(): grafo dirigido de menciones @usuario
  - analyze_network(): métricas, comunidades Louvain y visualización
  - generate_prompt_from_network(): texto estructurado para el LLM
"""

from .base import ExtractorBase


class NetworkAnalysisMixin(ExtractorBase):
    """Capacidades de análisis de grafos sobre self.data."""
