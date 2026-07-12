#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class LLMClient(ABC):
    """Abstract base class for LLM clients (OpenAI, Azure, etc)."""
    
    @abstractmethod
    def text_completion(
        self, 
        prompt: str, 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text completion from text prompt."""
        pass
    
    @abstractmethod
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion from message list."""
        pass
    
    @abstractmethod
    def vision_completion(
        self, 
        messages: List[Dict[str, Any]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion from text+image input."""
        pass

def get_client(provider: str, api_key: str = None, **kwargs) -> Any:
    """
    Get an LLM client instance for the given provider.
    
    Args:
        provider: The LLM provider to use (e.g., 'openai', 'azure', 'vertex')
        api_key: Optional API key to use for the client
        **kwargs: Additional arguments for specific providers
    """
    provider = provider.lower()
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(api_key=api_key)
    elif provider == "azure":
        from .azure_client import AzureClient
        return AzureClient(api_key=api_key)
    elif provider in ["vertex", "vertexai", "google"]:
        from .vertex_client import VertexAIClient
        return VertexAIClient(
            project_id=kwargs.get('project_id'),
            location=kwargs.get('location'),
            credentials_path=kwargs.get('credentials_path')
        )
    elif provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(api_key=api_key)
    elif provider in ["bedrock", "aws", "aws-bedrock"]:
        from .bedrock_client import BedrockClient
        return BedrockClient(
            region=kwargs.get('region'),
            model_id=kwargs.get('model_id'),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# Bedrock model-ID prefixes that identify AWS-hosted models
# Includes cross-region inference profile prefixes: eu.*, us.*, ap.*, global.*
_BEDROCK_PREFIXES = (
    "anthropic.", "amazon.", "meta.", "mistral.", "cohere.",
    "ai21.", "stability.", "writer.", "nvidia.", "qwen.",
    "minimax.", "twelvelabs.", "openai.gpt-oss", "zai.",
    "eu.anthropic.", "us.anthropic.", "ap.anthropic.", "global.anthropic.",
    "eu.amazon.", "us.amazon.", "ap.amazon.", "global.amazon.",
    "eu.meta.", "us.meta.", "ap.meta.", "global.meta.",
    "eu.mistral.", "us.mistral.", "ap.mistral.", "global.mistral.",
    "google.",
)


def get_client_for_model(model_id: str, **kwargs) -> Any:
    """
    Auto-select the right LLM client backend based on the model ID.

    Routing rules:
      - Starts with a Bedrock provider prefix  → BedrockClient
      - Starts with 'gemini'                   → VertexAIClient
      - Starts with 'gpt' or 'o1'              → OpenAIClient
      - Falls back to the DEFAULT_PROVIDER env var / config value

    Extra kwargs are forwarded to the chosen client constructor.
    """
    import os
    from .config import get_config

    if model_id and any(model_id.startswith(p) for p in _BEDROCK_PREFIXES):
        from .bedrock_client import BedrockClient
        return BedrockClient(model_id=model_id, **kwargs)

    if model_id and model_id.startswith("gemini"):
        from .vertex_client import VertexAIClient
        return VertexAIClient(
            project_id=kwargs.get('project_id') or os.getenv('GOOGLE_CLOUD_PROJECT'),
            location=kwargs.get('location') or os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1'),
            credentials_path=kwargs.get('credentials_path') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        )

    if model_id and (model_id.startswith("gpt") or model_id.startswith("o1")):
        from .openai_client import OpenAIClient
        return OpenAIClient(api_key=kwargs.get('api_key'))

    # Fallback: use configured default provider
    cfg = get_config()
    default_provider = os.getenv('DEFAULT_PROVIDER', cfg.get('default_provider', 'vertex'))
    return get_client(default_provider, **kwargs)
