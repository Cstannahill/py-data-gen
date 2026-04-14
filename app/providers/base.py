"""
Abstract base class for all LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class LLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate response from LLM"""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the provider is reachable"""
        pass
