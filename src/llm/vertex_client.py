#!/usr/bin/env python3

import os
import json
import base64
from typing import Dict, List, Any, Optional, Union
from vertexai import init
from vertexai.generative_models import GenerativeModel, Part, Content
from vertexai.language_models import TextEmbeddingModel
import vertexai.preview.generative_models as generative_models

from .client import LLMClient
from .config import get_config

class VertexAIClient(LLMClient):
    """Google Vertex AI implementation of the LLM client using Gemini models."""
    
    def __init__(self, project_id: str = None, location: str = None, credentials_path: str = None):
        """Initialize Vertex AI client.
        
        Args:
            project_id: Google Cloud project ID. If not provided, will look for GOOGLE_CLOUD_PROJECT in environment.
            location: Google Cloud location. If not provided, will look for GOOGLE_CLOUD_LOCATION in environment.
            credentials_path: Path to service account JSON. If not provided, will look for GOOGLE_APPLICATION_CREDENTIALS.
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if not self.project_id:
            raise ValueError("No project ID provided and GOOGLE_CLOUD_PROJECT not found in environment variables")
        
        # Set credentials if provided
        credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # Initialize Vertex AI
        try:
            init(project=self.project_id, location=self.location)
        except Exception as e:
            raise ValueError(f"Failed to initialize Vertex AI: {e}")
        
        # Load configuration
        self.config = get_config()
        self.default_text_model = self.config.get("vertex_text_model", "gemini-1.5-pro")
        self.default_vision_model = self.config.get("vertex_vision_model", "gemini-1.5-pro")
        self.default_embedding_model = self.config.get("vertex_embedding_model", "text-embedding-004")
        
        # Initialize models
        self._text_model = None
        self._vision_model = None
        self._embedding_model = None
    
    def _get_text_model(self, model_name: str = None) -> GenerativeModel:
        """Get or create text generation model."""
        model_name = model_name or self.default_text_model
        if self._text_model is None or self._text_model._model_name != model_name:
            self._text_model = GenerativeModel(model_name)
        return self._text_model
    
    def _get_vision_model(self, model_name: str = None) -> GenerativeModel:
        """Get or create vision model."""
        model_name = model_name or self.default_vision_model
        if self._vision_model is None or self._vision_model._model_name != model_name:
            self._vision_model = GenerativeModel(model_name)
        return self._vision_model
    
    def _get_embedding_model(self, model_name: str = None) -> TextEmbeddingModel:
        """Get or create embedding model."""
        model_name = model_name or self.default_embedding_model
        
        if self._embedding_model is None:
            self._embedding_model = TextEmbeddingModel.from_pretrained(model_name)
        
        return self._embedding_model
    
    def _create_generation_config(self, temperature: float = 0.1, max_tokens: int = 1000, **kwargs) -> Dict[str, Any]:
        """Create generation configuration for Vertex AI."""
        import logging
        logger = logging.getLogger(__name__)
        
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        logger.info(f"🔧 Vertex AI generation config: temperature={temperature}, max_output_tokens={max_tokens}")
        
        # Add other supported parameters
        if "top_p" in kwargs:
            config["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            config["top_k"] = kwargs["top_k"]
        if "candidate_count" in kwargs:
            config["candidate_count"] = kwargs["candidate_count"]
        if "stop_sequences" in kwargs:
            config["stop_sequences"] = kwargs["stop_sequences"]
            
        return config
    
    def text_completion(
        self, 
        prompt: str, 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text completion from text prompt using Vertex AI Gemini."""
        try:
            # Use config default if max_tokens not specified
            if max_tokens is None:
                max_tokens = self.config.get("vertex_default_max_tokens", 4000)
                
            model_instance = self._get_text_model(model)
            generation_config = self._create_generation_config(temperature, max_tokens, **kwargs)
            
            response = model_instance.generate_content(
                prompt,
                generation_config=generative_models.GenerationConfig(**generation_config)
            )
            
            # Extract text from response
            text = response.text if hasattr(response, 'text') else str(response)
            
            return {
                "text": text,
                "usage": self._extract_usage_info(response),
                "model": model or self.default_text_model,
                "raw_response": response
            }
            
        except Exception as e:
            raise RuntimeError(f"Vertex AI text completion failed: {e}")
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion from message list using Vertex AI Gemini."""
        try:
            # Use config default if max_tokens not specified
            if max_tokens is None:
                max_tokens = self.config.get("vertex_default_max_tokens", 4000)
                
            model_instance = self._get_text_model(model)
            generation_config = self._create_generation_config(temperature, max_tokens, **kwargs)
            
            # Convert messages to Vertex AI format
            chat_content = self._convert_messages_to_vertex_format(messages)
            
            # Disable safety filters that might truncate responses
            safety_settings = {
                generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
                generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            }
            
            response = model_instance.generate_content(
                chat_content,
                generation_config=generative_models.GenerationConfig(**generation_config),
                safety_settings=safety_settings
            )
            
            # Extract text from response with better error handling
            import logging
            logger = logging.getLogger(__name__)
            
            try:
                text = response.text if hasattr(response, 'text') else str(response)
                
                # Check if text is empty or just contains role info
                if not text or text.strip() == "" or text.strip() == '{"role": "model"}':
                    text = "I apologize, but I cannot provide a complete response at this time. Please try rephrasing your question or breaking it into smaller parts."
            except (ValueError, AttributeError) as e:
                # Handle safety filter or token limit issues
                logger.error(f"🔍 Gemini response extraction error: {e}")
                text = "I apologize, but I cannot provide a complete response at this time. Please try rephrasing your question or breaking it into smaller parts."
            
            return {
                "text": text,
                "usage": self._extract_usage_info(response),
                "model": model or self.default_text_model,
                "raw_response": response
            }
            
        except Exception as e:
            raise RuntimeError(f"Vertex AI chat completion failed: {e}")
    
    def vision_completion(
        self, 
        messages: List[Dict[str, Any]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate completion from text+image input using Vertex AI Gemini Vision."""
        try:
            # Use config default if max_tokens not specified
            if max_tokens is None:
                max_tokens = self.config.get("vertex_default_max_tokens", 4000)
                
            model_instance = self._get_vision_model(model)
            generation_config = self._create_generation_config(temperature, max_tokens, **kwargs)
            
            # Convert messages to Vertex AI format with image support
            content_parts = self._convert_vision_messages_to_vertex_format(messages)
            
            response = model_instance.generate_content(
                content_parts,
                generation_config=generative_models.GenerationConfig(**generation_config)
            )
            
            # Extract text from response
            text = response.text if hasattr(response, 'text') else str(response)
            
            return {
                "text": text,
                "usage": self._extract_usage_info(response),
                "model": model or self.default_vision_model,
                "raw_response": response
            }
            
        except Exception as e:
            raise RuntimeError(f"Vertex AI vision completion failed: {e}")
    
    def get_embeddings(
        self,
        texts: List[str],
        model: str = None,
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embedding_model = self._get_embedding_model(model)
            embeddings = embedding_model.get_embeddings(texts)
            return [embedding.values for embedding in embeddings]
        except Exception as e:
            raise RuntimeError(f"Vertex AI embedding generation failed: {e}")
    
    def _convert_messages_to_vertex_format(self, messages: List[Dict[str, str]]) -> List[Content]:
        """Convert OpenAI-style messages to Vertex AI Content format."""
        content_list = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Map OpenAI roles to Vertex AI roles
            vertex_role = "user" if role in ["user", "system"] else "model"
            
            content_list.append(Content(role=vertex_role, parts=[Part.from_text(content)]))
        
        return content_list
    
    def _convert_vision_messages_to_vertex_format(self, messages: List[Dict[str, Any]]) -> List[Part]:
        """Convert vision messages to Vertex AI Part format."""
        parts = []
        
        for message in messages:
            content = message.get("content")
            
            if isinstance(content, str):
                # Simple text content
                parts.append(Part.from_text(content))
            elif isinstance(content, list):
                # Multi-modal content (text + images)
                for item in content:
                    if item.get("type") == "text":
                        parts.append(Part.from_text(item.get("text", "")))
                    elif item.get("type") == "image_url":
                        # Handle image URL or base64
                        image_data = item.get("image_url", {})
                        url = image_data.get("url", "")
                        
                        if url.startswith("data:image"):
                            # Base64 encoded image
                            header, data = url.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            image_bytes = base64.b64decode(data)
                            parts.append(Part.from_data(image_bytes, mime_type))
                        else:
                            # URL or file path
                            parts.append(Part.from_uri(url, mime_type="image/jpeg"))
            else:
                # Fallback to text
                parts.append(Part.from_text(str(content)))
        
        return parts
    
    def _extract_usage_info(self, response) -> Dict[str, Any]:
        """Extract usage information from Vertex AI response."""
        usage_info = {}
        
        try:
            if hasattr(response, 'usage_metadata'):
                metadata = response.usage_metadata
                usage_info = {
                    "prompt_tokens": getattr(metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(metadata, 'total_token_count', 0)
                }
        except Exception:
            # If usage info is not available, return empty dict
            pass
        
        return usage_info
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for vision API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8') 