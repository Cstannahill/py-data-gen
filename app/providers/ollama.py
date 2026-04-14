"""
Ollama provider — local, self-hosted, and cloud LLM access via the Ollama REST API.
"""

import json
import requests
from typing import Any, Callable, Dict, Optional

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama provider for local, self-hosted, and cloud LLM access"""

    LOCAL_BASE_URL = "http://localhost:11434"
    CLOUD_BASE_URL = "https://ollama.com"

    def __init__(
        self,
        model: str,
        base_url: str = LOCAL_BASE_URL,
        api_key: Optional[str] = None,
        use_thinking: bool = True,
        use_structured_outputs: bool = True,
        temperature: float = 0.7,
        timeout: int = 300,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = self._resolve_base_url(model=model, base_url=base_url)
        self.request_model = self._resolve_request_model(model=model, base_url=self.base_url)
        self.use_thinking = use_thinking
        self.use_structured_outputs = use_structured_outputs
        self.temperature = temperature
        self.timeout = timeout

    @classmethod
    def is_cloud_model(cls, model: str) -> bool:
        return model.endswith("-cloud")

    @classmethod
    def _is_cloud_base_url(cls, base_url: str) -> bool:
        return "ollama.com" in base_url.lower()

    @classmethod
    def _resolve_base_url(cls, model: str, base_url: str) -> str:
        normalized_base_url = base_url.rstrip("/")
        if cls.is_cloud_model(model) and normalized_base_url == cls.LOCAL_BASE_URL:
            return cls.CLOUD_BASE_URL
        return normalized_base_url

    @classmethod
    def _resolve_request_model(cls, model: str, base_url: str) -> str:
        if cls._is_cloud_base_url(base_url) and cls.is_cloud_model(model):
            return model[: -len("-cloud")]
        return model

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._is_cloud_base_url(self.base_url):
            if not self.api_key:
                raise ValueError(
                    "OLLAMA_API_KEY is required for Ollama cloud models. "
                    "Set it in .env, pass --api-key, or provide api_key in config."
                )
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        *,
        structured_outputs: Optional[bool] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        url = f"{self.base_url}/api/generate"

        use_structured = (
            self.use_structured_outputs
            if structured_outputs is None
            else structured_outputs
        )

        payload: Dict[str, Any] = {
            "model": self.request_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 131072  # Support large context windows
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if use_structured:
            payload["format"] = "json"

        if context:
            payload["context"] = context

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self._headers(),
                stream=True,
                timeout=(10, 1800),
            )
            resp.raise_for_status()

            out: list[str] = []

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                    if "error" in msg:
                        raise Exception(f"Ollama error: {msg['error']}")
                except json.JSONDecodeError:
                    continue

                piece = msg.get("response") or ""
                if piece:
                    out.append(piece)
                    if on_chunk:
                        on_chunk(piece)

                if msg.get("done") is True:
                    break

            return "".join(out)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {e}")

    def chat(self, messages: list, context: Optional[Dict] = None) -> str:
        """Generate a response using the Ollama chat API (non-streaming)."""
        url = f"{self.base_url}/api/chat"

        use_structured = self.use_structured_outputs

        payload: Dict[str, Any] = {
            "model": self.request_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 131072
            },
        }

        if use_structured:
            payload["format"] = "json"

        if context:
            payload["context"] = context

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=(10, 1800),
            )
            response.raise_for_status()

            result = response.json()
            return result.get("message", {}).get("content", "")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {e}")

    def test_connection(self) -> bool:
        """Test if Ollama is reachable."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                headers=self._headers(),
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False
