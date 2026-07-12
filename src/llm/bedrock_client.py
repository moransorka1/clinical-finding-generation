#!/usr/bin/env python3
"""
AWS Bedrock LLM client.

Supports any model available on Amazon Bedrock via the Converse API:
  - Anthropic Claude  (anthropic.claude-*)
  - Amazon Nova       (amazon.nova-*)
  - Meta Llama        (meta.llama*)
  - Mistral           (mistral.*)
  - Cohere            (cohere.*)
  - ...and all other Converse-API-compatible models

The client uses the unified Converse / ConverseStream API so the same code
works across all providers hosted on Bedrock.
"""

import os
import json
import base64
import logging
from typing import Dict, List, Any, Optional

import boto3
from botocore.exceptions import ClientError

from .client import LLMClient
from .config import get_config

logger = logging.getLogger(__name__)


# Bedrock model families that need special inference parameters
_THINKING_MODELS = {"anthropic.claude-3-7-sonnet", "anthropic.claude-sonnet-4"}

# Cross-region inference profile region prefixes
_PROFILE_PREFIXES = ("eu.", "us.", "ap.", "global.")


def _strip_region_prefix(model_id: str) -> str:
    """Strip cross-region inference profile prefix (e.g. 'eu.') to get the base model ID."""
    for prefix in _PROFILE_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def _model_family(model_id: str) -> str:
    """Return the base model family string (prefix before version suffix)."""
    # Strip inference profile region prefix first
    base = _strip_region_prefix(model_id)
    # e.g. "anthropic.claude-sonnet-4-6" -> "anthropic.claude-sonnet-4"
    parts = base.split("-")
    return "-".join(parts[:-1]) if len(parts) > 1 else base


class BedrockClient(LLMClient):
    """
    Amazon Bedrock implementation of the LLM client.

    Uses the Converse API which provides a unified interface for all
    Bedrock-hosted models.

    Environment variables (all optional if passed explicitly):
        AWS_REGION          – AWS region, default eu-west-1
        AWS_BEDROCK_MODEL   – Default model ID
        AWS_ACCESS_KEY_ID   – AWS credentials (uses IAM role if absent)
        AWS_SECRET_ACCESS_KEY
        AWS_SESSION_TOKEN
    """

    def __init__(
        self,
        region: str = None,
        model_id: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_session_token: str = None,
    ):
        self.region = (
            region
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
        )
        self.config = get_config()
        self.default_model = (
            model_id
            or os.getenv("AWS_BEDROCK_MODEL", "eu.anthropic.claude-sonnet-4-6")
        )
        self.default_max_tokens = self.config.get("bedrock_default_max_tokens", 8192)

        session_kwargs: Dict[str, Any] = {"region_name": self.region}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        session = boto3.Session(**session_kwargs)
        from botocore.config import Config as BotocoreConfig
        self.client = session.client(
            "bedrock-runtime",
            config=BotocoreConfig(
                read_timeout=120,      # 2 min max per API call
                connect_timeout=10,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
        logger.info(
            f"BedrockClient initialised | region={self.region} | default_model={self.default_model}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_converse_request(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        model_id: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Build the request dict for bedrock-runtime.converse()."""
        bedrock_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                # System messages are handled separately in Bedrock Converse API
                if system is None:
                    system = msg["content"]
                continue
            content_raw = msg["content"]
            if isinstance(content_raw, str):
                content = [{"text": content_raw}]
            elif isinstance(content_raw, list):
                # Already in Bedrock content-block format or OpenAI multipart format
                content = []
                for block in content_raw:
                    if isinstance(block, dict):
                        if "text" in block:
                            content.append({"text": block["text"]})
                        elif block.get("type") == "text":
                            content.append({"text": block["text"]})
                        elif block.get("type") == "image_url":
                            # OpenAI-style image block
                            url = block["image_url"].get("url", "")
                            if url.startswith("data:"):
                                header, b64data = url.split(",", 1)
                                media_type = header.split(";")[0].replace("data:", "")
                                content.append({
                                    "image": {
                                        "format": media_type.split("/")[-1],
                                        "source": {"bytes": base64.b64decode(b64data)},
                                    }
                                })
                            else:
                                content.append({"text": f"[Image URL: {url}]"})
                        else:
                            content.append(block)
                    else:
                        content.append({"text": str(block)})
            else:
                content = [{"text": str(content_raw)}]

            bedrock_messages.append({"role": role, "content": content})

        request: Dict[str, Any] = {
            "modelId": model_id,
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            request["system"] = [{"text": system}]

        return request

    def _call_converse(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call bedrock-runtime.converse() and return the raw response."""
        try:
            return self.client.converse(**request)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            msg = e.response["Error"]["Message"]
            raise RuntimeError(f"Bedrock Converse API error [{code}]: {msg}") from e

    def _extract_text(self, response: Dict[str, Any]) -> str:
        """Extract the assistant text from a Converse response."""
        try:
            content_blocks = response["output"]["message"]["content"]
            texts = [b["text"] for b in content_blocks if "text" in b]
            return "\n".join(texts)
        except (KeyError, TypeError):
            return str(response)

    def _extract_usage(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract token-usage metadata from a Converse response."""
        usage = response.get("usage", {})
        return {
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        }

    # ------------------------------------------------------------------
    # Public LLMClient interface
    # ------------------------------------------------------------------

    def text_completion(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Single-turn text completion (prompt → response)."""
        model_id = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens
        system = kwargs.pop("system", None)

        messages = [{"role": "user", "content": prompt}]
        request = self._build_converse_request(
            messages, system, model_id, temperature, max_tokens
        )
        response = self._call_converse(request)
        text = self._extract_text(response)

        logger.info(
            f"Bedrock text_completion | model={model_id} | tokens={self._extract_usage(response)}"
        )
        return {
            "text": text,
            "usage": self._extract_usage(response),
            "model": model_id,
            "raw_response": response,
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Multi-turn chat completion."""
        model_id = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens
        system = kwargs.pop("system", None)

        request = self._build_converse_request(
            messages, system, model_id, temperature, max_tokens
        )
        response = self._call_converse(request)
        text = self._extract_text(response)

        logger.info(
            f"Bedrock chat_completion | model={model_id} | tokens={self._extract_usage(response)}"
        )
        return {
            "text": text,
            "usage": self._extract_usage(response),
            "model": model_id,
            "raw_response": response,
        }

    def vision_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Multimodal (text + image) completion via Bedrock Converse API."""
        model_id = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens
        system = kwargs.pop("system", None)

        request = self._build_converse_request(
            messages, system, model_id, temperature, max_tokens
        )
        response = self._call_converse(request)
        text = self._extract_text(response)

        logger.info(
            f"Bedrock vision_completion | model={model_id} | tokens={self._extract_usage(response)}"
        )
        return {
            "text": text,
            "usage": self._extract_usage(response),
            "model": model_id,
            "raw_response": response,
        }
