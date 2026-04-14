"""
Anthropic provider — placeholder, not yet implemented.
"""

from typing import Dict, Optional

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Placeholder for Anthropic provider"""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        raise NotImplementedError("Anthropic provider not yet implemented")

    def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        raise NotImplementedError()

    def test_connection(self) -> bool:
        raise NotImplementedError()
