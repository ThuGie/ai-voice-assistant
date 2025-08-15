import unittest
from unittest.mock import patch, MagicMock
from src.ai import AIEngine, OllamaConnectionError

class TestAIEngine(unittest.TestCase):

    @patch('requests.get')
    def test_init_ollama_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3"}]}
        ai = AIEngine(model="llama3", api_type="ollama")
        self.assertEqual(ai.api_type, "ollama")

    def test_init_webui_success(self):
        # WebUI doesn't have an initial check, so this should just work
        ai = AIEngine(model="test_model", api_type="text-generation-webui")
        self.assertEqual(ai.api_type, "text-generation-webui")

    @patch('src.ai.AIEngine._generate_complete')
    def test_generate_response_ollama(self, mock_generate):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": [{"name": "llama3"}]}
            ai = AIEngine(api_type="ollama")
            ai.generate_response("test prompt")
            mock_generate.assert_called_once()

    @patch('src.ai.AIEngine._generate_complete_webui')
    def test_generate_response_webui(self, mock_generate_webui):
        ai = AIEngine(api_type="text-generation-webui")
        ai.generate_response("test prompt")
        mock_generate_webui.assert_called_once()

    @patch('requests.get')
    def test_list_models_ollama(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3"}]}
        ai = AIEngine(api_type="ollama")
        models = ai.list_available_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['name'], 'llama3')
        mock_get.assert_called_with(f"{ai.api_base}/tags", timeout=5)

    @patch('requests.get')
    def test_list_models_webui(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"data": [{"id": "test_model"}]}
        ai = AIEngine(api_type="text-generation-webui")
        models = ai.list_available_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['name'], 'test_model')
        mock_get.assert_called_with(f"{ai.api_base}/models", timeout=5)

if __name__ == '__main__':
    unittest.main()
