import logging
from typing import Dict, List, Optional
from google.oauth2 import service_account
from google.cloud import aiplatform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for model providers (USING UPPERCASE)
VERTEX_AI = "VERTEX_AI"
AZURE = "AZURE"
ANTHROPIC = "ANTHROPIC"


class LLMConfiguration:
    """Generic LLM configuration class."""

    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = provider
        self.model = model
        self.config = kwargs

    def get_config(self) -> Dict:
        """Returns the LLM configuration dictionary."""

        if self.provider == VERTEX_AI:
            return self._get_vertex_ai_config()
        elif self.provider == AZURE:
            return self._get_azure_config()
        elif self.provider == ANTHROPIC:  # Add Anthropic case
            return self._get_anthropic_config()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_vertex_ai_config(self) -> Dict:
        """Constructs and returns the Vertex AI configuration."""

        project_id = self.config.get("project_id")
        location = self.config.get("location")
        credentials_dict = self.config.get("vertex_credentials")

        if not all([project_id, location, credentials_dict]):
            raise ValueError("Missing required parameters for Vertex AI: project_id, location, vertex_credentials")

        # Convert credentials dictionary to proper credentials object
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)

        # Initialize aiplatform with proper credentials
        aiplatform.init(
            project=project_id,
            location=location,
            credentials=credentials  # Use the credentials object instead of dict
        )

        # Note: Removed unsupported fields: retry_timeout, max_retries, api_rate_limit from config_list entry
        # Note: Removed unsupported field: request_timeout from the top level
        config = {
            "config_list": [
                {
                    "model": self.model,
                    "api_type": "google", # Changed from 'vertex_ai' to 'google' as per potential autogen requirements
                    "location": location,
                    "project_id": project_id,
                    # Pass credentials directly if needed by the specific autogen google model wrapper,
                    # though often it relies on the environment or global aiplatform init.
                    # Check autogen documentation for the exact way to pass credentials if required here.
                    # "credentials": credentials, # This might be needed depending on autogen version/usage
                }
            ],
            "cache_seed": self.config.get("cache_seed", 42),
            "temperature": self.config.get("temperature", 0),
            "max_tokens": self.config.get("max_tokens", 4096),
        }
        # Add timeout/retry config at the top level if supported by autogen's OpenAIWrapper or similar
        # Check autogen documentation for supported top-level parameters
        # e.g., config["request_timeout"] = self.config.get("request_timeout", 600)
        return config

    def _get_azure_config(self) -> Dict:
        """Constructs and returns the Azure configuration."""

        required_params = ["api_key", "base_url", "api_version"]
        if not all(param in self.config for param in required_params):
            raise ValueError(f"Missing required parameters for Azure: {required_params}")

        config = {
            "config_list": [
                {
                    "model": self.model,
                    "api_key": self.config["api_key"],
                    "base_url": self.config["base_url"],
                    "api_type": "azure",
                    "api_version": self.config["api_version"],
                    "max_tokens": self.config.get("max_tokens", 4096),
                }
            ],
            "temperature": self.config.get("temperature", 0),
        }
        return config

    def _get_anthropic_config(self) -> Dict:
        """Constructs and returns the Anthropic configuration."""

        required_params = ["api_key", "base_url"] # Assuming base_url might be needed, adjust if not
        if not all(param in self.config for param in required_params):
            raise ValueError(f"Missing required parameters for Anthropic: {required_params}")

        config = {
            "config_list": [
                {
                    "model": self.model,
                    "api_key": self.config["api_key"],
                    "api_type": "anthropic", # Use 'anthropic' type
                    # "base_url": self.config["base_url"], # Include if needed by autogen
                    "max_tokens": self.config.get("max_tokens", 4096),
                }
            ],
            # Anthropic timeouts/retries might be configured differently or have defaults.
            # Adjust based on autogen's anthropic wrapper specifics.
            "request_timeout": self.config.get("request_timeout", 120),
            "temperature": self.config.get("temperature", 0),
            # "api_rate_limit": self.config.get("api_rate_limit", 0.1), # Check if supported
        }
        return config
