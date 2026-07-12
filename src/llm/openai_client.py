#!/usr/bin/env python3

import os
from typing import Dict, List, Any, Optional, Union
import base64
from openai import OpenAI

from .client import LLMClient
from .config import get_config

class OpenAIClient(LLMClient):
    """OpenAI implementation of the LLM client."""
    
    def __init__(self, api_key: str = None):
        """Initialize OpenAI client with API key.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY in environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided and OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Default models
        self.config = get_config()
        self.default_text_model = self.config.get("openai_text_model", "gpt-4o")
        self.default_vision_model = self.config.get("openai_vision_model", "gpt-4-turbo")
    
    def text_completion(
        self, 
        prompt: str, 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text completion from text prompt using OpenAI API."""
        if not model:
            model = self.default_text_model
        
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
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion from message list using OpenAI API."""
        if not model:
            model = self.default_text_model
        
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
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion from text+image input using OpenAI API."""
        if not model:
            model = self.default_vision_model
        
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
