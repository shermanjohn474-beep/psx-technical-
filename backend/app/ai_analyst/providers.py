"""Pluggable multimodal LLM providers (server-side keys only).

Every provider implements ``complete_json``: given a system prompt, user text,
optional images and a JSON schema, return a parsed JSON object plus metadata
(provider, model, latency, token usage, refusal info) for the audit trail.
Providers never see raw API keys outside this module and never echo them.
"""
from __future__ import annotations

import base64
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.config.settings import Settings, get_settings


class AIProviderError(RuntimeError):
    pass


class AIProviderNotConfigured(AIProviderError):
    pass


class AIRefusal(AIProviderError):
    pass


@dataclass
class ImageInput:
    data: bytes
    media_type: str  # image/png | image/jpeg | image/webp


@dataclass
class AIResult:
    data: dict
    provider: str
    model: str
    latency_s: float
    usage: dict = field(default_factory=dict)
    raw_text: str | None = None


class LLMProvider(ABC):
    name = "abstract"
    model = "n/a"

    @abstractmethod
    def complete_json(self, system: str, text: str, schema: dict, images: list[ImageInput] | None = None,
                      max_tokens: int = 16000) -> AIResult:
        ...


class NullProvider(LLMProvider):
    name = "none"

    def complete_json(self, system, text, schema, images=None, max_tokens=16000) -> AIResult:
        raise AIProviderNotConfigured("No AI provider configured (set AI_PROVIDER and the matching API key).")


class AnthropicProvider(LLMProvider):
    """Claude via the official ``anthropic`` SDK, with structured JSON output.

    Uses server-side refusal fallbacks (``fallbacks: "default"``) and checks
    ``stop_reason == "refusal"`` before reading content.
    """

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-opus-5-5", timeout: float = 90.0, effort: str = "medium"):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise AIProviderNotConfigured("The 'anthropic' package is not installed (pip install anthropic).") from exc
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.effort = effort

    def complete_json(self, system, text, schema, images=None, max_tokens=16000) -> AIResult:
        content: list[dict] = []
        for im in images or []:
            content.append({"type": "image", "source": {"type": "base64", "media_type": im.media_type,
                                                        "data": base64.standard_b64encode(im.data).decode()}})
        content.append({"type": "text", "text": text})
        t0 = time.time()
        a = self._anthropic
        try:
            resp = self._client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.RateLimitError as exc:
            raise AIProviderError("AI provider rate limit reached; retry later.") from exc
        except a.APIStatusError as exc:
            raise AIProviderError(f"AI provider returned HTTP {exc.status_code}.") from exc
        except a.APIConnectionError as exc:
            raise AIProviderError("Could not reach the AI provider.") from exc
        if resp.stop_reason == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            raise AIRefusal(f"Model declined the request (category: {cat}).")
        if resp.stop_reason == "max_tokens":
            raise AIProviderError("AI response truncated (max_tokens).")
        txt = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            data = json.loads(txt)
        except json.JSONDecodeError as exc:
            raise AIProviderError("AI provider returned invalid JSON.") from exc
        usage = {"input_tokens": getattr(resp.usage, "input_tokens", None),
                 "output_tokens": getattr(resp.usage, "output_tokens", None)}
        return AIResult(data=data, provider=self.name, model=getattr(resp, "model", self.model),
                        latency_s=round(time.time() - t0, 2), usage=usage, raw_text=txt)


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions with ``json_schema`` response format (raw HTTPS)."""

    name = "openai"

    def __init__(self, api_key: str, model: str, timeout: float = 90.0):
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def complete_json(self, system, text, schema, images=None, max_tokens=16000) -> AIResult:
        import httpx

        parts: list[dict] = [{"type": "text", "text": text}]
        for im in images or []:
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:{im.media_type};base64,{base64.standard_b64encode(im.data).decode()}"}})
        body = {"model": self.model, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": parts}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": True}}}
        t0 = time.time()
        try:
            r = httpx.post("https://api.openai.com/v1/chat/completions", json=body, timeout=self.timeout,
                           headers={"Authorization": f"Bearer {self.api_key}"})
        except httpx.HTTPError as exc:
            raise AIProviderError("Could not reach the AI provider.") from exc
        if r.status_code >= 400:
            raise AIProviderError(f"AI provider returned HTTP {r.status_code}.")
        msg = r.json()["choices"][0]["message"]
        if msg.get("refusal"):
            raise AIRefusal("Model declined the request.")
        try:
            data = json.loads(msg["content"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise AIProviderError("AI provider returned invalid JSON.") from exc
        return AIResult(data=data, provider=self.name, model=self.model, latency_s=round(time.time() - t0, 2),
                        usage=r.json().get("usage", {}), raw_text=msg["content"])


def get_ai_provider(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    if s.ai_provider == "anthropic" and s.anthropic_api_key:
        return AnthropicProvider(s.anthropic_api_key, s.anthropic_model, s.ai_timeout_seconds)
    if s.ai_provider == "openai" and s.openai_api_key:
        return OpenAIProvider(s.openai_api_key, s.openai_model, s.ai_timeout_seconds)
    return NullProvider()
