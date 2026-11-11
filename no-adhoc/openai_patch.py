"""
Module Patch untuk OpenAI LLM dan Chat Model di LangChain.
Mengatasi keterbatasan langchain==0.0.300 yang belum mendukung model OpenAI generasi baru (seperti gpt-5.6-terra, gpt-5.6-sol).

Fitur:
1. PatchedOpenAI:
   - Menghubungkan pemanggilan interface OpenAI (llm(...), LLMChain) ke endpoint Chat Completions secara transparan.
   - Menyesuaikan parameter untuk model next-gen (gpt-5, terra, sol, o1, o3, o4):
     * Mengalihkan 'max_tokens' ke 'max_completion_tokens'.
     * Menghapus 'temperature' jika != 1 (karena model reasoning / next-gen menolak temperature selain default 1.0).
2. PatchedChatOpenAI:
   - Menyesuaikan pemanggilan ChatOpenAI LangChain untuk model next-gen:
     * Menghapus 'max_tokens: None' atau mengalihkannya ke 'max_completion_tokens' jika diatur.
     * Menghapus 'temperature' jika != 1.0 agar tidak ditolak oleh API.
"""

import os
from typing import Any, Dict, List, Optional
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI as BaseChatOpenAI
from langchain.schema.output import LLMResult, Generation
from langchain.callbacks.manager import CallbackManagerForLLMRun


NEXT_GEN_TAGS = ["gpt-5", "terra", "sol", "o1", "o3", "o4"]


class PatchedOpenAI(OpenAI):
    """
    Subclass OpenAI yang mendukung model chat dan model generasi baru (seperti gpt-5.6-terra, gpt-5.6-sol)
    melalui Chat Completions API secara transparan tanpa UserWarning atau error parameter.
    """

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        is_next_gen = any(tag in (self.model_name or "").lower() for tag in NEXT_GEN_TAGS)

        api_key = getattr(self, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")
        for p in prompts:
            params: Dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": p}],
                "api_key": api_key,
                **kwargs,
            }

            if not is_next_gen:
                if self.temperature is not None:
                    params["temperature"] = self.temperature
                if self.max_tokens:
                    params["max_tokens"] = self.max_tokens
            else:
                # Model generasi baru (gpt-5.6-terra, gpt-5.6-sol) menolak max_tokens dan temperature != 1
                if self.max_tokens:
                    params["max_completion_tokens"] = self.max_tokens

            if stop:
                params["stop"] = stop

            resp = openai.ChatCompletion.create(**params)
            text = resp.choices[0].message.content or ""
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)


class PatchedChatOpenAI(BaseChatOpenAI):
    """
    Subclass ChatOpenAI yang mendukung model generasi baru (seperti gpt-5.6-sol, gpt-5.6-terra, o1, o3, o4)
    dengan menangani parameter yang ditolak oleh API model generasi baru:
    - Menghapus 'max_tokens' jika None, atau mengalihkannya ke 'max_completion_tokens' jika bernilai integer.
    - Menghapus 'temperature' jika != 1 (karena model-model reasoning / next-gen OpenAI hanya mendukung default 1).
    """

    def _sanitize_next_gen_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        is_next_gen = any(tag in (self.model_name or "").lower() for tag in NEXT_GEN_TAGS)
        if is_next_gen:
            if "max_tokens" in params:
                max_toks = params.pop("max_tokens")
                if max_toks is not None and "max_completion_tokens" not in params:
                    params["max_completion_tokens"] = max_toks
            if "temperature" in params:
                temp = params.get("temperature")
                if temp is not None and temp != 1:
                    params.pop("temperature", None)
        return params

    @property
    def _default_params(self) -> Dict[str, Any]:
        params = super()._default_params
        return self._sanitize_next_gen_params(params)

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        kwargs = self._sanitize_next_gen_params(kwargs)
        return super().completion_with_retry(run_manager=run_manager, **kwargs)


# Alias untuk kemudahan import
OpenAI = PatchedOpenAI
ChatOpenAI = PatchedChatOpenAI

__all__ = ["PatchedOpenAI", "PatchedChatOpenAI", "OpenAI", "ChatOpenAI"]
