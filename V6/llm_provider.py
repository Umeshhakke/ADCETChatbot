from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import requests

from settings import RuntimeConfig


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GPT4AllProvider(LLMProvider):
    def __init__(self, config: RuntimeConfig) -> None:
        from gpt4all import GPT4All

        if not config.gpt4all_model_path and not config.llm_model:
            raise ValueError(
                "GPT4All requires ADCET_GPT4ALL_MODEL_PATH or ADCET_LLM_MODEL."
            )

        model_name = config.gpt4all_model_path or config.llm_model
        self._config = config
        self._client = GPT4All(
            model_name=model_name,
            allow_download=config.gpt4all_allow_download,
            device=config.gpt4all_device,
        )

    def generate(self, prompt: str) -> str:
        return self._client.generate(
            prompt,
            max_tokens=self._config.llm_max_tokens,
            temp=self._config.llm_temperature,
            top_k=self._config.llm_top_k,
            top_p=self._config.llm_top_p,
            repeat_penalty=self._config.llm_repeat_penalty,
        ).strip()


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, config: RuntimeConfig) -> None:
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai provider.")

        if not config.llm_model:
            raise ValueError("ADCET_LLM_MODEL is required for the openai provider.")

        self._config = config
        self._session = requests.Session()

    def generate(self, prompt: str) -> str:
        response = self._session.post(
            f"{self._config.openai_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._config.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._config.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._config.llm_temperature,
                "max_tokens": self._config.llm_max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload["choices"][0]["message"]["content"].strip()


class OllamaProvider(LLMProvider):
    def __init__(self, config: RuntimeConfig) -> None:
        if not config.llm_model:
            raise ValueError("ADCET_LLM_MODEL is required for the ollama provider.")

        self._config = config
        self._session = requests.Session()

    def generate(self, prompt: str) -> str:
        response = self._session.post(
            f"{self._config.ollama_base_url}/api/generate",
            json={
                "model": self._config.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self._config.llm_temperature,
                    "top_k": self._config.llm_top_k,
                    "top_p": self._config.llm_top_p,
                    "repeat_penalty": self._config.llm_repeat_penalty,
                    "num_predict": self._config.llm_max_tokens,
                    "num_ctx": self._config.llm_context_window,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload["response"].strip()


def create_llm_provider(config: RuntimeConfig) -> LLMProvider:
    providers = {
        "gpt4all": GPT4AllProvider,
        "openai": OpenAICompatibleProvider,
        "ollama": OllamaProvider,
    }

    provider_class = providers.get(config.llm_provider)

    if provider_class is None:
        supported = ", ".join(sorted(providers))
        raise ValueError(
            f"Unsupported ADCET_LLM_PROVIDER '{config.llm_provider}'. "
            f"Supported values: {supported}."
        )

    return provider_class(config)
