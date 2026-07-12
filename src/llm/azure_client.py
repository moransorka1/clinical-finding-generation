#!/usr/bin/env python3

import os
from typing import Dict, List, Any, Optional, Union
import base64
from openai import AzureOpenAI

from .client import LLMClient
from .config import get_config

class AzureClient(LLMClient):
    """Azure OpenAI implementation of the LLM client."""
    
    def __init__(self, api_key: str = None):
        """Initialize Azure OpenAI client.
        
        Args:
            api_key: Azure OpenAI API key. If not provided, will use hardcoded key.
        """
        # Azure settings - hardcoded credentials
        self.endpoint = os.getenv(
            "AZURE_ENDPOINT_URL", 
            "https://rambam-openai-neurology-sweden.openai.azure.com/"
        )
        # Always use Azure key - ignore OpenAI keys (starting with sk-)
        azure_key = "BX3MLYGtdLMjXtZ1rN1y3mbtYTOe7qzDCnHqHj1VFtrDRf5HYAmyJQQJ99BAACfhMk5XJ3w3AAABACOGr2tt"
        if api_key and not api_key.startswith("sk-"):
            self.subscription_key = api_key
        else:
            self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY", azure_key)
        self.api_version = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
            
        # Initialize client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.subscription_key,
            api_version=self.api_version,
        )
        
        # Default models/deployments - use gpt-4o as default
        self.config = get_config()
        self.default_text_deployment = os.getenv("AZURE_TEXT_DEPLOYMENT", "gpt-4o")
        self.default_vision_deployment = os.getenv("AZURE_VISION_DEPLOYMENT", "gpt-4o")
    
    def text_completion(
        self, 
        prompt: str, 
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text completion from text prompt using Azure OpenAI."""
        if not model:
            model = self.default_text_deployment
        
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return {
            "text": response.choices[0].message.content,
            "usage": response.usage.to_dict() if hasattr(response, 'usage') else {},
            "model": response.model,
            "raw_response": response
        }
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion from message list using Azure OpenAI."""
        if not model:
            model = self.default_text_deployment
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return {
            "text": response.choices[0].message.content,
            "usage": response.usage.to_dict() if hasattr(response, 'usage') else {},
            "model": response.model,
            "raw_response": response
        }
    
    def vision_completion(
        self, 
        messages: List[Dict[str, Any]],
        model: str = None,
        temperature: float = 1.0,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion from text+image input using Azure OpenAI."""
        if not model:
            model = self.default_vision_deployment
        
        # Process messages to handle image URLs or base64 content
        processed_messages = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                # Already formatted for vision API
                processed_messages.append(msg)
            else:
                processed_messages.append(msg)
        
        response = self.client.chat.completions.create(
            model=model,
            messages=processed_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return {
            "text": response.choices[0].message.content,
            "usage": response.usage.to_dict() if hasattr(response, 'usage') else {},
            "model": response.model,
            "raw_response": response
        }
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for vision API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
