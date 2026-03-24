"""
LLM Provider implementations
"""

import json
import requests
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable


class LLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response from LLM"""
        pass


class OllamaProvider(LLMProvider):
    """Ollama provider for local and cloud LLM access"""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        use_thinking: bool = True,
        use_structured_outputs: bool = True,
        temperature: float = 0.7,
        timeout: int = 300,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.use_thinking = use_thinking
        self.use_structured_outputs = use_structured_outputs
        self.temperature = temperature
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
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
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": self.temperature},
        }

        if self.use_thinking:
            payload["options"]["enable_thinking"] = True

        if use_structured:
            payload["format"] = "json"

        if context:
            payload["context"] = context

        num_predict = 10000

        try:
            resp = requests.post(url, json=payload, stream=True, timeout=(10, 1800))
            resp.raise_for_status()

            out: list[str] = []

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    msg = json.loads(line)
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
        """Generate response using Ollama chat API"""
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        if self.use_thinking:
            payload["options"]["enable_thinking"] = True

        if self.use_structured_outputs:
            payload["format"] = "json"

        if context:
            payload["context"] = context

        try:
            response = requests.post(
                url, json=payload, stream=True, timeout=(10, 1800)
            )  # 10s connect, 30min read

            response.raise_for_status()

            result = response.json()
            return result.get("message", {}).get("content", "")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {e}")

    def test_connection(self) -> bool:
        """Test if Ollama is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# Placeholder for additional providers
class OpenAIProvider(LLMProvider):
    """Placeholder for OpenAI provider"""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        raise NotImplementedError("OpenAI provider not yet implemented")

    def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        raise NotImplementedError()


class AnthropicProvider(LLMProvider):
    """Placeholder for Anthropic provider"""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        raise NotImplementedError("Anthropic provider not yet implemented")

    def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        raise NotImplementedError()
