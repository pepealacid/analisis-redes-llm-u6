"""
Integración con modelos de lenguaje (Unidad 6).

Método principal:
  - chat_local_llm(): inferencia local con transformers y chat interactivo
"""

import os
import sys
import time
from threading import Thread
from typing import Any, Optional

from tqdm import tqdm
from transformers.generation.streamers import BaseStreamer

from .base import ExtractorBase

DEFAULT_LLM_MODEL = "google/gemma-4-E2B-it"
DEFAULT_CHAT_LOG = "outputs/llm_chat.txt"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_CHAR_DELAY = 0.006


class _GenerationDisplayStreamer(BaseStreamer):
    """Barra de progreso + texto en consola carácter a carácter."""

    def __init__(
        self,
        tokenizer: Any,
        progress_bar: tqdm,
        *,
        char_delay: float = DEFAULT_CHAR_DELAY,
    ) -> None:
        self._tokenizer = tokenizer
        self._progress_bar = progress_bar
        self._char_delay = char_delay
        self._parts: list[str] = []
        self._text_started = False

    def put(self, value: Any) -> None:
        if len(value.shape) > 1:
            value = value[0]
        self._progress_bar.update(int(value.shape[0]))

        text = self._tokenizer.decode(value.tolist(), skip_special_tokens=True)
        if not text:
            return

        if not self._text_started:
            self._progress_bar.close()
            self._text_started = True

        self._parts.append(text)
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            if self._char_delay > 0:
                time.sleep(self._char_delay)

    def end(self) -> None:
        if not self._text_started:
            self._progress_bar.close()
        if self._parts:
            sys.stdout.write("\n")
            sys.stdout.flush()

    @property
    def text(self) -> str:
        return "".join(self._parts).strip()


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
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        model_id: str = DEFAULT_LLM_MODEL,
        char_delay: float = DEFAULT_CHAR_DELAY,
    ) -> str:
        import torch

        self._load_local_llm(model_id)
        assert self._llm_tokenizer is not None
        assert self._llm_model is not None

        tokenizer = self._llm_tokenizer
        model = self._llm_model

        if hasattr(tokenizer, "apply_chat_template"):
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            encoded = tokenizer(text, return_tensors="pt")

        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
            attention_mask = getattr(encoded, "attention_mask", None)
        elif isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        else:
            input_ids = encoded
            attention_mask = None

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        progress_bar = tqdm(
            total=max_new_tokens,
            unit="tok",
            desc="Generando",
            dynamic_ncols=True,
        )
        display_streamer = _GenerationDisplayStreamer(
            tokenizer,
            progress_bar,
            char_delay=char_delay,
        )
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.pad_token_id,
            "streamer": display_streamer,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask.to(device)

        output_ids: list[Any] = []

        def _run_generation() -> None:
            with torch.no_grad():
                output_ids.append(model.generate(**generate_kwargs))

        thread = Thread(target=_run_generation, daemon=True)
        thread.start()
        thread.join()

        if not output_ids:
            progress_bar.close()
            return ""

        return display_streamer.text

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
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        save_path: str = DEFAULT_CHAT_LOG,
        char_delay: float = DEFAULT_CHAR_DELAY,
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
            print("\n--- Respuesta del modelo ---\n")
            reply = self._generate_llm_reply(
                self._chat_history,
                max_new_tokens=max_new_tokens,
                model_id=model_id,
                char_delay=char_delay,
            )
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
            print("\nAsistente:")
            reply = self._generate_llm_reply(
                self._chat_history,
                max_new_tokens=max_new_tokens,
                model_id=model_id,
                char_delay=char_delay,
            )
            self._chat_history.append({"role": "assistant", "content": reply})

        if self._chat_history:
            self._save_chat_log(save_path)

        return self._chat_history
