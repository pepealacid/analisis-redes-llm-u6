"""
Integración con modelos de lenguaje (Unidad 6).

Método previsto:
  - chat_local_llm(): inferencia local con transformers y chat interactivo
"""

from .base import ExtractorBase


class LlmMixin(ExtractorBase):
    """Interacción con LLM local (o Gemini como respaldo)."""
