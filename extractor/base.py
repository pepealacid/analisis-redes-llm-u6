"""Utilidades compartidas por los mixins de DataExtractor."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


class ExtractorBase:
    """Atributos y helpers comunes (usado como mixin, no instanciar solo)."""

    source_file: Optional[str]
    data: Optional[pd.DataFrame]
    chunksize: int

    def _require_data(self) -> None:
        if self.data is None:
            raise ValueError("Primero debes cargar los datos.")

    def _save_current_figure(self, save_path: Optional[str]) -> None:
        if not save_path:
            return
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
