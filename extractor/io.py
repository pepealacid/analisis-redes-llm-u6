"""Carga de datos desde fichero local o RapidAPI."""

import os

import pandas as pd

from rapidapi_client import RapidApiTwitterClient

from .base import ExtractorBase


class DataLoadingMixin(ExtractorBase):
    """Métodos de ingesta: CSV/JSON y API twitter-api45."""

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
                key: value for key, value in read_kwargs.items() if key != "low_memory"
            }
            return pd.read_csv(
                self.source_file,
                sep=None,
                engine="python",
                **read_kwargs_python,
            )
        except Exception as exc:
            raise ValueError(
                f"No se pudo leer el CSV correctamente: {self.source_file}"
            ) from exc
