"""
OpenRouter provider — OpenAI-compatible API giving access to hundreds of models.

Docs: https://openrouter.ai/docs
Base URL: https://openrouter.ai/api/v1
"""

import json
import requests
from typing import Any, Callable, Dict, Optional

from .base import LLMProvider


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider — routes requests to many upstream model providers."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str,
        api_key: str,
        use_structured_outputs: bool = True,
        temperature: float = 0.7,
        timeout: int = 300,
        site_url: str = "",
        site_name: str = "py-data-gen",
    ):
        self.model = model
        self.api_key = api_key
        self.use_structured_outputs = use_structured_outputs
        self.temperature = temperature
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: list,
        *,
        structured_outputs: Optional[bool] = None,
        stream: bool = True,
    ) -> Dict[str, Any]:
        use_structured = (
            self.use_structured_outputs
            if structured_outputs is None
            else structured_outputs
        )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }

        if use_structured:
            payload["response_format"] = {"type": "json_object"}

        return payload

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        *,
        structured_outputs: Optional[bool] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate a response from a plain prompt string (streaming) with retries."""
        
        import time

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if context and "system" in context:
            messages.insert(0, {"role": "system", "content": context["system"]})

        url = f"{self.BASE_URL}/chat/completions"
        payload = self._build_payload(
            messages, structured_outputs=structured_outputs, stream=True
        )

        max_retries = 5
        base_delay = 5

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers,
                    json=payload,
                    stream=True,
                    timeout=(10, self.timeout),
                )
                resp.raise_for_status()

                out: list[str] = []

                for raw_line in resp.iter_lines(decode_unicode=True):
                    line = raw_line.strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break

                    try:
                        msg = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    delta = msg.get("choices", [{}])[0].get("delta", {})
                    piece = delta.get("content") or ""
                    if piece:
                        out.append(piece)
                        if on_chunk:
                            on_chunk(piece)

                return "".join(out)

            except requests.exceptions.RequestException as e:
                status_code = getattr(e.response, 'status_code', None)
                if status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"\n[OpenRouter] Retrying in {wait_time}s due to error: {e}")
                    time.sleep(wait_time)
                    continue
                raise Exception(f"OpenRouter API error: {e}")
        
        return ""

    def chat(self, messages: list, context: Optional[Dict] = None) -> str:
        """Generate a response from a list of chat messages (non-streaming) with retries."""
        import time

        if context and "system" in context:
            messages = [{"role": "system", "content": context["system"]}] + messages

        url = f"{self.BASE_URL}/chat/completions"
        payload = self._build_payload(messages, stream=False)

        max_retries = 5
        base_delay = 5

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers,
                    json=payload,
                    timeout=(10, self.timeout),
                )
                resp.raise_for_status()

                result = resp.json()
                return result["choices"][0]["message"]["content"] or ""

            except requests.exceptions.RequestException as e:
                status_code = getattr(e.response, 'status_code', None)
                if status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"\n[OpenRouter] Retrying in {wait_time}s due to error: {e}")
                    time.sleep(wait_time)
                    continue
                raise Exception(f"OpenRouter API error: {e}")
        
        return ""

    def test_connection(self) -> bool:
        """Test connectivity by fetching the available model list."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/models",
                headers=self._headers,
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False
