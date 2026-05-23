"""
Integración con modelos de lenguaje (Unidad 6).

Método principal:
  - chat_local_llm(): inferencia local con transformers y chat interactivo
"""

import os
from typing import Any, Optional

from .base import ExtractorBase

DEFAULT_LLM_MODEL = "google/gemma-4-E2B-it"
DEFAULT_CHAT_LOG = "outputs/llm_chat.txt"


class LlmMixin(ExtractorBase):
    """Interacción con LLM local mediante transformers."""

    _llm_tokenizer: Any
    _llm_model: Any
    _chat_history: list[dict[str, str]]

    def _load_local_llm(self, model_id: str = DEFAULT_LLM_MODEL) -> None:
        """Carga perezosa del tokenizer y del modelo causal en local."""
        if self._llm_model is not None and self._llm_tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"Cargando modelo local {model_id!r} en {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to(device)

        model.eval()
        self._llm_tokenizer = tokenizer
        self._llm_model = model

    def _generate_llm_reply(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 512,
        model_id: str = DEFAULT_LLM_MODEL,
    ) -> str:
        import torch

        self._load_local_llm(model_id)
        assert self._llm_tokenizer is not None
        assert self._llm_model is not None

        tokenizer = self._llm_tokenizer
        model = self._llm_model

        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            input_ids = tokenizer(text, return_tensors="pt").input_ids

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        input_len = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _save_chat_log(self, save_path: str = DEFAULT_CHAT_LOG) -> None:
        """Persiste la conversación en disco (requisito del enunciado)."""
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as file:
            for message in self._chat_history:
                role = "Usuario" if message["role"] == "user" else "Asistente"
                file.write(f"{role}:\n{message['content']}\n\n")

        print(f"Conversación guardada en: {save_path}")

    def chat_local_llm(
        self,
        prompt: Optional[str] = None,
        *,
        model_id: str = DEFAULT_LLM_MODEL,
        max_new_tokens: int = 512,
    ) -> list[dict[str, str]]:
        """
        Levanta Gemma en local y permite chat interactivo.

        Si se pasa un prompt (p. ej. el generado desde la red), el modelo
        responde automáticamente y esa respuesta queda en el contexto.
        """
        self._chat_history = []

        if prompt:
            print("\n--- Prompt inicial (insights de la red) ---\n")
            print(prompt)
            self._chat_history.append({"role": "user", "content": prompt})
            reply = self._generate_llm_reply(
                self._chat_history,
                max_new_tokens=max_new_tokens,
                model_id=model_id,
            )
            print("\n--- Respuesta del modelo ---\n")
            print(reply)
            self._chat_history.append({"role": "assistant", "content": reply})

        print(
            "\nModo chat (escribe 'salir' para terminar). "
            "El contexto incluye el análisis previo si se proporcionó un prompt inicial."
        )

        while True:
            try:
                user_text = input("\nTú: ").strip()
            except KeyboardInterrupt:
                print("\nFin del chat.")
                break

            if user_text.lower() in {"salir", "exit", "quit"}:
                print("Fin del chat.")
                break
            if not user_text:
                continue

            self._chat_history.append({"role": "user", "content": user_text})
            reply = self._generate_llm_reply(
                self._chat_history,
                max_new_tokens=max_new_tokens,
                model_id=model_id,
            )
            print(f"\nAsistente: {reply}")
            self._chat_history.append({"role": "assistant", "content": reply})

        if self._chat_history:
            self._save_chat_log()

        return self._chat_history
