
import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import json

# Add the project root to sys.path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.providers.ollama import OllamaProvider
from app.providers.openrouter import OpenRouterProvider
from app.dataset_generator import DatasetGenerator, GenerationConfig

class TestTask6SystemPrompts(unittest.TestCase):
    def test_ollama_provider_system_prompt(self):
        """Step 2: OllamaProvider.generate should include system_prompt in payload."""
        provider = OllamaProvider(model="test-model")
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = [json.dumps({"response": "test", "done": True}).encode('utf-8')]
            mock_post.return_value = mock_response
            
            provider.generate(prompt="User prompt", system_prompt="System instructions")
            
            # Verify payload
            args, kwargs = mock_post.call_args
            payload = kwargs['json']
            self.assertEqual(payload.get("system"), "System instructions")
            self.assertEqual(payload.get("prompt"), "User prompt")

    def test_openrouter_provider_system_prompt(self):
        """Step 3: OpenRouterProvider.generate should include system_prompt as a system message."""
        provider = OpenRouterProvider(model="test-model", api_key="test-key")
        
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            # Mock streaming response
            mock_response.iter_lines.return_value = [
                'data: {"choices": [{"delta": {"content": "test"}}]}',
                'data: [DONE]'
            ]
            mock_post.return_value = mock_response
            
            provider.generate(prompt="User prompt", system_prompt="System instructions")
            
            # Verify messages in payload
            args, kwargs = mock_post.call_args
            payload = kwargs['json']
            messages = payload.get("messages")
            self.assertEqual(len(messages), 2)
            self.assertEqual(messages[0], {"role": "system", "content": "System instructions"})
            self.assertEqual(messages[1], {"role": "user", "content": "User prompt"})

    def test_ollama_cloud_uses_auth_header_and_cloud_base_url(self):
        provider = OllamaProvider(model="gpt-oss:20b-cloud", api_key="ollama-test-key")

        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = [
                json.dumps({"response": "test", "done": True}).encode('utf-8')
            ]
            mock_post.return_value = mock_response

            provider.generate(prompt="User prompt")

            args, kwargs = mock_post.call_args
            self.assertEqual(args[0], "https://ollama.com/api/generate")
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer ollama-test-key")
            self.assertEqual(kwargs["json"]["model"], "gpt-oss:20b")

    def test_dataset_generator_passes_system_prompt(self):
        """Step 4: DatasetGenerator._generate_batch should pass generation_prompt as system_prompt."""
        config = GenerationConfig(
            provider="ollama",
            model="test-model",
            output_dir="./test_output"
        )
        
        with patch('app.dataset_generator.OllamaProvider'), \
             patch('app.dataset_generator.PromptBuilder'), \
             patch('app.dataset_generator.DatasetEvaluator'), \
             patch('app.dataset_generator.ProgressTracker'):
            generator = DatasetGenerator(config)
            generator.generation_prompt = "Master System Prompt"
            generator.provider.generate.return_value = '[{"id": 1}]'
            
            generator._generate_batch(num_entries=1)
            
            # Verify generate call
            args, kwargs = generator.provider.generate.call_args
            # Depending on if we use positional or keyword args in the implementation
            # The task says: response = self.provider.generate(prompt=batch_prompt, system_prompt=self.generation_prompt)
            self.assertEqual(kwargs.get("system_prompt"), "Master System Prompt")
            
            # The prompt should be just the batch request, not including the base prompt anymore
            batch_prompt = kwargs.get("prompt")
            self.assertIn("## BATCH REQUEST", batch_prompt)
            self.assertNotIn("Master System Prompt", batch_prompt)

    @patch.dict(os.environ, {"OLLAMA_API_KEY": "env-ollama-key"}, clear=False)
    def test_dataset_generator_uses_ollama_api_key_for_cloud_models(self):
        config = GenerationConfig(
            provider="ollama",
            model="gpt-oss:20b-cloud",
            output_dir="./test_output"
        )

        with patch('app.dataset_generator.OllamaProvider') as mock_ollama_provider, \
             patch('app.dataset_generator.PromptBuilder'), \
             patch('app.dataset_generator.DatasetEvaluator'), \
             patch('app.dataset_generator.ProgressTracker'):
            DatasetGenerator(config)

            _, kwargs = mock_ollama_provider.call_args
            self.assertEqual(kwargs["api_key"], "env-ollama-key")

if __name__ == "__main__":
    unittest.main()
