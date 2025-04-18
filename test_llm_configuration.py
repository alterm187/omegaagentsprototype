import pytest
from unittest.mock import patch, MagicMock
from LLMConfiguration import LLMConfiguration, VERTEX_AI, AZURE, ANTHROPIC


class TestLLMConfiguration:

    @patch("google.cloud.aiplatform.init")
    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    def test_get_vertex_ai_config_success(self, mock_from_service_account_info, mock_aiplatform_init):
        mock_creds = MagicMock()
        mock_from_service_account_info.return_value = mock_creds
        config_data = {
            "provider": VERTEX_AI,
            "model": "test-model",
            "project_id": "test-project",
            "location": "test-location",
            "vertex_credentials": {"test": "credentials"},
            "cache_seed": 123,
            "temperature": 0.5,
            "max_tokens": 1000,
        }
        llm_config = LLMConfiguration(**config_data)
        config = llm_config.get_config()

        assert config["config_list"][0]["model"] == "test-model"
        assert config["config_list"][0]["api_type"] == "google"
        assert config["config_list"][0]["location"] == "test-location"
        assert config["config_list"][0]["project_id"] == "test-project"
        assert config["cache_seed"] == 123
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 1000
        mock_from_service_account_info.assert_called_once_with({"test": "credentials"})
        mock_aiplatform_init.assert_called_once_with(
            project="test-project", location="test-location", credentials=mock_creds
        )

    def test_get_vertex_ai_config_missing_params(self):
        config_data = {
            "provider": VERTEX_AI,
            "model": "test-model",
            # Missing project_id, location, vertex_credentials
        }
        llm_config = LLMConfiguration(**config_data)
        with pytest.raises(ValueError, match="Missing required parameters for Vertex AI"):
            llm_config.get_config()

    def test_get_azure_config_success(self):
        config_data = {
            "provider": AZURE,
            "model": "test-model",
            "api_key": "test-key",
            "base_url": "test-url",
            "api_version": "v1",
            "temperature": 0.7,
            "max_tokens": 2000,
        }
        llm_config = LLMConfiguration(**config_data)
        config = llm_config.get_config()

        assert config["config_list"][0]["model"] == "test-model"
        assert config["config_list"][0]["api_key"] == "test-key"
        assert config["config_list"][0]["base_url"] == "test-url"
        assert config["config_list"][0]["api_type"] == "azure"
        assert config["config_list"][0]["api_version"] == "v1"
        assert config["config_list"][0]["max_tokens"] == 2000
        assert config["temperature"] == 0.7

    def test_get_azure_config_missing_params(self):
        config_data = {
            "provider": AZURE,
            "model": "test-model",
            # Missing api_key, base_url, api_version
        }
        llm_config = LLMConfiguration(**config_data)
        with pytest.raises(ValueError, match="Missing required parameters for Azure"):
            llm_config.get_config()

    def test_get_anthropic_config_success(self):
        config_data = {
            "provider": ANTHROPIC,
            "model": "test-model",
            "api_key": "test-key",
            "base_url": "test-url",
            "request_timeout": 60,
            "temperature": 0.9,
            "max_tokens": 1500,
        }
        llm_config = LLMConfiguration(**config_data)
        config = llm_config.get_config()

        assert config["config_list"][0]["model"] == "test-model"
        assert config["config_list"][0]["api_key"] == "test-key"
        assert config["config_list"][0]["api_type"] == "anthropic"
        assert config["config_list"][0]["max_tokens"] == 1500
        assert config["request_timeout"] == 60
        assert config["temperature"] == 0.9

    def test_get_anthropic_config_missing_params(self):
        config_data = {
            "provider": ANTHROPIC,
            "model": "test-model",
            # Missing api_key
        }
        llm_config = LLMConfiguration(**config_data)
        with pytest.raises(ValueError, match="Missing required parameters for Anthropic"):
            llm_config.get_config()

    def test_unsupported_provider(self):
        config_data = {"provider": "UNSUPPORTED", "model": "test-model"}
        llm_config = LLMConfiguration(**config_data)
        with pytest.raises(ValueError, match="Unsupported provider: UNSUPPORTED"):
            llm_config.get_config()