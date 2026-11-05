"""
Module Patch untuk ChatAnthropic.
Mengatasi keterbatasan langchain==0.0.300 yang masih menembak endpoint legacy (/v1/complete).
Dengan PatchedChatAnthropic, model Claude modern (seperti claude-3, claude-3-5, claude-sonnet-4-5)
dapat berjalan di perangkat mana pun via Messages API (/v1/messages) tanpa perlu memodifikasi library.
"""

from typing import Any, Dict, List, Optional
import requests
from langchain.chat_models import ChatAnthropic
from langchain.chat_models.base import _generate_from_stream
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.messages import AIMessage, BaseMessage
from langchain.schema.output import ChatGeneration, ChatResult


class PatchedChatAnthropic(ChatAnthropic):
    """Subclass ChatAnthropic yang mendukung Anthropic Messages API (/v1/messages)

    secara otomatis jika endpoint completions gagal atau tidak didukung model.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return _generate_from_stream(stream_iter)

        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {
            "prompt": prompt,
            **self._default_params,
            **kwargs,
        }
        if stop:
            params["stop_sequences"] = stop

        try:
            response = self.client.completions.create(**params)
            completion = response.completion
        except Exception:
            # Fallback otomatis ke Anthropic Messages API (/v1/messages) untuk model Claude modern
            headers = {
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens_to_sample or 1024,
                "temperature": self.temperature if self.temperature is not None else 0,
                "messages": [{"role": "user", "content": prompt}],
            }
            if stop:
                body["stop_sequences"] = stop

            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
            )
            if resp.status_code != 200:
                raise ValueError(
                    f"Anthropic API Error [{resp.status_code}]: {resp.text}"
                )
            completion = resp.json()["content"][0]["text"]

        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])
