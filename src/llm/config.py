#!/usr/bin/env python3

import os
import json
from pathlib import Path
from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    # OpenAI models
    "openai_text_model": "gpt-4o",
    "openai_mini_model": "gpt-4o-mini", 
    "openai_vision_model": "gpt-4-turbo",
    
    # Azure deployments (model names on Azure)
    "azure_text_deployment": "gpt-4o",
    "azure_vision_deployment": "gpt-4o",
    
    # Vertex AI models
    "vertex_text_model": "gemini-2.5-flash",
    "vertex_vision_model": "gemini-2.5-flash",
    "vertex_embedding_model": "gemini-embedding-001",
    "vertex_flash_model": "gemini-2.5-flash",
    
    # AWS Bedrock models
    "bedrock_default_model": "eu.anthropic.claude-sonnet-4-6",
    "bedrock_default_max_tokens": 8192,
    "bedrock_region": "eu-west-1",

    # Other settings
    "default_provider": "vertex",
    "default_temperature": 0.1,
    "default_max_tokens": 1000,
    "vertex_default_max_tokens": 12000,  # Increased from 8000 to handle complex medical reasoning with thoughts mechanism
}

# Config file paths to check
CONFIG_PATHS = [
    Path("config.json"),
    Path("backend/llm/config.json"),
    Path(os.path.expanduser("~/.cursor/config.json")),
]

_config_cache = None

def get_config() -> Dict[str, Any]:
    """
    Load configuration from config file or use defaults.
    
    Checks for config file in several locations, with environment variables
    taking precedence over file settings.
    
    Returns:
        Dict containing configuration settings
    """
    global _config_cache
    
    # Return cached config if available
    if _config_cache is not None:
        return _config_cache
    
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from config file
    for config_path in CONFIG_PATHS:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
                break
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config from {config_path}: {e}")
    
    # Override with environment variables if set
    env_overrides = {
        "OPENAI_TEXT_MODEL": "openai_text_model",
        "OPENAI_MINI_MODEL": "openai_mini_model",
        "OPENAI_VISION_MODEL": "openai_vision_model",
        "AZURE_TEXT_DEPLOYMENT": "azure_text_deployment",
        "AZURE_VISION_DEPLOYMENT": "azure_vision_deployment",
        "VERTEX_TEXT_MODEL": "vertex_text_model",
        "VERTEX_VISION_MODEL": "vertex_vision_model", 
        "VERTEX_EMBEDDING_MODEL": "vertex_embedding_model",
        "VERTEX_FLASH_MODEL": "vertex_flash_model",
        "VERTEX_DEFAULT_MAX_TOKENS": "vertex_default_max_tokens",
        "DEFAULT_PROVIDER": "default_provider",
        "DEFAULT_TEMPERATURE": "default_temperature",
        "DEFAULT_MAX_TOKENS": "default_max_tokens",
        "AWS_BEDROCK_MODEL": "bedrock_default_model",
        "AWS_REGION": "bedrock_region",
    }
    
    for env_var, config_key in env_overrides.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            # Convert to appropriate type
            if config_key in ("default_temperature",):
                config[config_key] = float(env_value)
            elif config_key in ("default_max_tokens", "vertex_default_max_tokens"):
                config[config_key] = int(env_value)
            else:
                config[config_key] = env_value
    
    # Cache the config
    _config_cache = config
    
    return config