"""
Cliente para Twitter Search API en RapidAPI (twitter-api45).

Documentación del endpoint: GET /search.php con paginación por next_cursor.
"""

import os
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

RAPIDAPI_HOST_DEFAULT = "twitter-api45.p.rapidapi.com"
SEARCH_ENDPOINT = "/search.php"
PAGE_LIMIT = 50


class RapidApiTwitterClient:
    """Descarga tweets de búsqueda y los devuelve en formato tabular."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        timeout: int = 30,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("RAPIDAPI_KEY")
        self.host = host or os.getenv("RAPIDAPI_HOST", RAPIDAPI_HOST_DEFAULT)
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("No se ha encontrado RAPIDAPI_KEY. Configúralo en .env.")

    def search_tweets(
        self,
        query: str,
        max_results: int = 100,
        search_type: str = "Top",
    ) -> pd.DataFrame:
        """Obtiene tweets paginando hasta alcanzar max_results o agotar cursor."""
        if not query.strip():
            raise ValueError("La consulta de búsqueda no puede estar vacía.")

        collected: list[dict[str, str]] = []
        cursor: Optional[str] = None
        pages = 0

        while len(collected) < max_results and pages < PAGE_LIMIT:
            pages += 1
            payload = self._get_search_page(query, search_type, cursor)
            page_items = self._items_from_timeline(payload)

            for item in page_items:
                row = self._map_tweet_row(item)
                if row:
                    collected.append(row)
                    if len(collected) >= max_results:
                        break

            if len(collected) >= max_results or not page_items:
                break

            cursor = (payload.get("next_cursor") or "").strip() or None
            if not cursor:
                break

        if not collected:
            raise ValueError(
                f"No se obtuvieron tweets para la consulta {query!r}. "
                "Revisa la cuota de RapidAPI o la consulta."
            )

        return pd.DataFrame(collected[:max_results])

    def _get_search_page(
        self,
        query: str,
        search_type: str,
        cursor: Optional[str],
    ) -> dict[str, Any]:
        params: dict[str, str] = {"query": query, "search_type": search_type}
        if cursor:
            params["cursor"] = cursor

        url = f"https://{self.host}{SEARCH_ENDPOINT}"
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host,
        }

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"RapidAPI respondió {response.status_code}: {response.text}"
            )

        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("La respuesta de RapidAPI no es un objeto JSON válido.")

        return data

    @staticmethod
    def _items_from_timeline(payload: dict[str, Any]) -> list[dict[str, Any]]:
        timeline = payload.get("timeline") or []
        if not isinstance(timeline, list):
            return []
        return [entry for entry in timeline if isinstance(entry, dict)]

    @staticmethod
    def _map_tweet_row(tweet: dict[str, Any]) -> Optional[dict[str, str]]:
        text = (tweet.get("text") or "").strip()
        if not text:
            return None

        return {
            "text": text,
            "date": tweet.get("created_at") or "",
            "user_name": tweet.get("screen_name") or "unknown",
        }
