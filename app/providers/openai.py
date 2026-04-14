"""
OpenAI provider — placeholder, not yet implemented.
"""

from typing import Dict, Optional

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """Placeholder for OpenAI provider"""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        raise NotImplementedError("OpenAI provider not yet implemented")

    def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        raise NotImplementedError()

    def test_connection(self) -> bool:
        raise NotImplementedError()
