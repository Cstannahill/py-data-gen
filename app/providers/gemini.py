"""
Google AI Studio (Gemini) provider — modular integration for Gemini models.
"""

import json
import requests
from typing import Any, Callable, Dict, List, Optional

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """Provider for Google AI Studio (Gemini) API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        use_structured_outputs: bool = True,
        temperature: float = 0.7,
        timeout: int = 300,
    ):
        self.model = model
        self.api_key = api_key
        self.use_structured_outputs = use_structured_outputs
        self.temperature = temperature
        self.timeout = timeout

    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        structured_outputs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        use_structured = (
            self.use_structured_outputs
            if structured_outputs is None
            else structured_outputs
        )

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }

        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        if use_structured:
            payload["generationConfig"]["response_mime_type"] = "application/json"

        return payload

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        *,
        structured_outputs: Optional[bool] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate response from Gemini API with streaming support and retries."""
        
        import time

        # If context has system prompt, prioritize it or merge it
        effective_system = system_prompt
        if context and "system" in context:
            effective_system = f"{effective_system}\n\n{context['system']}" if effective_system else context["system"]

        method = "streamGenerateContent" if on_chunk else "generateContent"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:{method}?key={self.api_key}"
        
        payload = self._build_payload(
            prompt=prompt,
            system_prompt=effective_system,
            structured_outputs=structured_outputs
        )

        max_retries = 5
        base_delay = 5
        
        for attempt in range(max_retries):
            try:
                if on_chunk:
                    return self._generate_stream(url, payload, on_chunk)
                else:
                    return self._generate_non_stream(url, payload)
            except requests.exceptions.RequestException as e:
                # Check for rate limits (429) or server errors (5xx)
                status_code = getattr(e.response, 'status_code', None)
                if status_code in [429, 500, 503] and attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"\n[Gemini] Retrying in {wait_time}s due to error: {e}")
                    time.sleep(wait_time)
                    continue
                raise Exception(f"Gemini API error: {e}")
        
        return "" # Should not reach here

    def _generate_non_stream(self, url: str, payload: Dict[str, Any]) -> str:
        resp = requests.post(url, json=payload, timeout=(10, self.timeout))
        resp.raise_for_status()
        
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            if "error" in data:
                raise Exception(f"Gemini API error: {data['error']['message']}")
            raise Exception(f"Unexpected Gemini API response format: {json.dumps(data)}")

    def _generate_stream(self, url: str, payload: Dict[str, Any], on_chunk: Callable[[str], None]) -> str:
        resp = requests.post(url, json=payload, stream=True, timeout=(10, self.timeout))
        resp.raise_for_status()
        
        full_text = []
        
        # Gemini streaming returns a JSON array of objects over time.
        # We need to handle the streaming response which might come in chunks that aren't full JSON lines.
        # Note: requests.iter_lines might not be perfect for this if the chunks are large or split differently.
        # However, for Google's implementation, it often sends full objects or identifiable chunks.
        
        # Simpler approach: gather content and look for 'text' fields in candidates.
        # Google's stream format is actually a sequence of JSON objects, usually one per line or similar.
        
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            
            # Remove leading/trailing markers if it's sent as a single large array (it often is in streamGenerateContent)
            # Actually, Gemini stream returns a JSON array that grows. 
            # This is tricky to parse with simple json.loads while streaming.
            # A common trick is to remove the leading '[' and trailing ']' or ', '
            
            clean_line = line.strip()
            if clean_line.startswith("["): clean_line = clean_line[1:]
            if clean_line.endswith("]"): clean_line = clean_line[:-1]
            if clean_line.startswith(","): clean_line = clean_line[1:]
            
            if not clean_line:
                continue
                
            try:
                chunk_data = json.loads(clean_line)
                piece = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                if piece:
                    full_text.append(piece)
                    on_chunk(piece)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
                
        return "".join(full_text)

    def chat(self, messages: list, context: Optional[Dict] = None) -> str:
        """Generate a response from a list of chat messages (non-streaming) with retries."""
        
        import time

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        # Convert messages to Gemini format
        # Gemini messages format: {"role": "user"|"model", "parts": [{"text": "..."}]}
        contents = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                # System prompt is usually handled separately in system_instruction
                continue

            gemini_role = "user" if role == "user" else "model"
            contents.append({
                "role": gemini_role,
                "parts": [{"text": msg["content"]}]
            })

        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
        if context and "system" in context:
            system_prompt = f"{system_prompt}\n\n{context['system']}" if system_prompt else context["system"]

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
            },
        }

        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        if self.use_structured_outputs:
            payload["generationConfig"]["response_mime_type"] = "application/json"

        max_retries = 5
        base_delay = 5

        for attempt in range(max_retries):
            try:
                return self._generate_non_stream(url, payload)
            except requests.exceptions.RequestException as e:
                status_code = getattr(e.response, 'status_code', None)
                if status_code in [429, 500, 503] and attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"\n[Gemini] Retrying in {wait_time}s due to error: {e}")
                    time.sleep(wait_time)
                    continue
                raise Exception(f"Gemini API error: {e}")
        
        return ""

    def test_connection(self) -> bool:
        """Test if the Gemini API is reachable and the key is valid."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
        try:
            resp = requests.get(url, timeout=10)
            return resp.status_code == 200
        except Exception:
            return False
