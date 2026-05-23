"""
Análisis de redes con NetworkX (Unidad 6).

Métodos previstos:
  - build_interaction_graph(): grafo dirigido de menciones @usuario
  - analyze_network(): métricas, comunidades Louvain y visualización
  - generate_prompt_from_network(): texto estructurado para el LLM
"""

import os
import re
from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms import community as nx_community

from .base import ExtractorBase

# Captura el nombre de usuario tras @ (letras, números y guion bajo).
_MENTION_REGEX = re.compile(r"@([\w]+)", flags=re.UNICODE)

DEFAULT_NETWORK_FIGURE = "outputs/network_graph.png"
DEFAULT_NETWORK_PROMPT = "outputs/network_prompt.txt"


def _ascii_safe(text: str) -> str:
    """Evita fallos de consola en Windows con caracteres no ASCII."""
    return text.encode("ascii", errors="replace").decode("ascii")


class NetworkAnalysisMixin(ExtractorBase):
    """Capacidades de análisis de grafos sobre self.data."""

    _network_insights: Optional[dict[str, Any]]

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

    def _plot_network_by_community(
        self,
        graph: nx.DiGraph,
        degree_scores: dict[str, float],
        node_community: dict[str, int],
        save_path: str,
    ) -> None:
        """Dibuja el grafo coloreando comunidades y escalando nodos por centralidad."""
        layout = nx.spring_layout(graph, seed=42)
        cmap = plt.get_cmap("tab10")

        node_colors = [
            cmap(node_community.get(node, 0) % 10) for node in graph.nodes()
        ]
        node_sizes = [
            200 + degree_scores.get(node, 0.0) * 4000 for node in graph.nodes()
        ]

        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(
            graph,
            layout,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
        )
        nx.draw_networkx_labels(graph, layout, font_size=7)
        nx.draw_networkx_edges(
            graph,
            layout,
            alpha=0.35,
            arrows=True,
            arrowsize=10,
            width=0.8,
        )
        plt.title("Red de menciones (@usuario)")
        plt.axis("off")
        plt.tight_layout()
        self._save_current_figure(save_path)
        plt.close()

    def analyze_network(
        self,
        graph: nx.DiGraph,
        save_path: str = DEFAULT_NETWORK_FIGURE,
    ) -> dict[str, Any]:
        """
        Calcula métricas de centralidad, densidad y comunidades (Louvain).

        Genera una visualización donde el tamaño del nodo refleja la centralidad
        de grado y el color identifica la comunidad Louvain.

        Args:
            graph: Grafo dirigido devuelto por build_interaction_graph().
            save_path: Ruta donde guardar la figura matplotlib.

        Returns:
            Diccionario con métricas e insights de la red.
        """
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        if node_count == 0:
            insights = {
                "nodes": 0,
                "edges": 0,
                "density": 0.0,
                "components": 0,
                "component_sizes": [],
                "degree_centrality": {},
                "betweenness_centrality": {},
                "top_nodes": [],
                "communities": [],
                "community_sizes": [],
                "figure_path": save_path,
            }
            self._network_insights = insights
            print("Grafo vacío: no hay métricas que calcular.")
            return insights

        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        density = float(nx.density(graph))

        weak_components = list(nx.weakly_connected_components(graph))
        component_sizes = sorted(
            (len(component) for component in weak_components),
            reverse=True,
        )

        undirected = graph.to_undirected()
        if edge_count > 0:
            communities = list(
                nx_community.louvain_communities(undirected, seed=42)
            )
        else:
            communities = [{node} for node in graph.nodes()]

        community_sizes = sorted(
            (len(community) for community in communities),
            reverse=True,
        )

        top_nodes = sorted(
            degree_centrality.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]

        node_community = {}
        for community_id, members in enumerate(communities):
            for member in members:
                node_community[member] = community_id

        self._plot_network_by_community(
            graph,
            degree_centrality,
            node_community,
            save_path,
        )

        insights = {
            "nodes": node_count,
            "edges": edge_count,
            "density": density,
            "components": len(weak_components),
            "component_sizes": component_sizes,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "top_nodes": top_nodes,
            "communities": communities,
            "community_sizes": community_sizes,
            "figure_path": save_path,
        }
        self._network_insights = insights

        print("\n=== Análisis de red ===")
        print(f"Nodos: {node_count} | Aristas: {edge_count}")
        print(f"Densidad: {density:.4f}")
        print(f"Componentes débilmente conexas: {len(weak_components)}")

        print("\nTop 3 usuarios por centralidad de grado:")
        for rank, (user, score) in enumerate(top_nodes, start=1):
            print(f"  {rank}. @{_ascii_safe(user)} - {score:.4f}")

        print(f"\nComunidades Louvain: {len(communities)}")
        preview = ", ".join(str(size) for size in community_sizes[:8])
        if len(community_sizes) > 8:
            preview += ", ..."
        print(f"  Tamaños (mayor a menor): [{preview}]")

        print(f"\nVisualización guardada en: {save_path}")

        return insights

    def _dominant_hashtag(self) -> tuple[str, int]:
        """Obtiene el hashtag más repetido del corpus cargado."""
        self._require_data()
        hashtag_stats = self.analytics_hashtags_extended()["overall"]

        if hashtag_stats.empty:
            return "ninguno detectado", 0

        top_row = hashtag_stats.iloc[0]
        return str(top_row["hashtag"]), int(top_row["frequency"])

    def _format_top_users_for_prompt(
        self,
        top_nodes: list[tuple[str, float]],
    ) -> str:
        if not top_nodes:
            return "  (no hay usuarios con menciones registradas)"

        lines = []
        for index, (user, score) in enumerate(top_nodes, start=1):
            lines.append(
                f"  {index}. @{_ascii_safe(user)} "
                f"(centralidad de grado = {score:.4f})"
            )
        return "\n".join(lines)

    def generate_prompt_from_network(
        self,
        graph: nx.DiGraph,
        save_path: str = DEFAULT_NETWORK_PROMPT,
    ) -> str:
        """
        Construye un prompt reutilizable para interpretar la red con un LLM.

        Reutiliza las métricas de ``analyze_network()`` cuando están disponibles.
        Guarda una copia del texto en ``outputs/network_prompt.txt``.

        Args:
            graph: Grafo analizado previamente o listo para analizar.
            save_path: Ruta del fichero donde persistir el prompt.

        Returns:
            Texto del prompt listo para enviar al modelo.
        """
        if self._network_insights is None:
            self.analyze_network(graph)

        insights = self._network_insights
        top_hashtag, top_hashtag_freq = self._dominant_hashtag()

        top_users_block = self._format_top_users_for_prompt(
            insights.get("top_nodes", [])
        )
        community_count = len(insights.get("communities", []))
        density = insights.get("density", 0.0)

        prompt = f"""Contexto del análisis
---------------------
Se ha modelado una red social a partir de tweets. Cada arista del grafo
representa que un usuario mencionó (@) a otro en el texto original.

Indicadores calculados:
- Nodos (usuarios): {insights.get("nodes", 0)}
- Aristas (menciones): {insights.get("edges", 0)}
- Densidad de la red: {density:.4f}
- Comunidades detectadas (Louvain): {community_count}
- Hashtag más frecuente: {top_hashtag} ({top_hashtag_freq} apariciones)

Actores con mayor centralidad de grado (top 3):
{top_users_block}

Tareas de interpretación
------------------------
Con la información anterior, elabora un informe breve que cubra:

1. Comportamiento social observable en la red (quién menciona a quién,
   presencia de hubs, conversaciones aisladas o interconectadas).
2. Tendencias temáticas que puedan relacionarse con el hashtag dominante
   y con los usuarios más centrales.
3. Posibles causas o factores que expliquen esos patrones (eventos,
   campañas, cuentas influyentes, sesgo del corpus).
4. Actores relevantes a monitorizar y por qué destacan en la red.

Responde en español, con secciones numeradas y lenguaje claro. No inventes
cifras adicionales: apóyate únicamente en los datos proporcionados.
"""

        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as file:
            file.write(prompt)

        print(f"Prompt de red guardado en: {save_path}")

        return prompt
