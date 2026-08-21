"""Provider-neutral chat client and OpenAI-compatible implementation."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Mapping, Protocol, Sequence

import httpx

from ._version import __version__
from .config import ProviderConfig
from .errors import ConfigurationError, ProviderError, ResponseError
from .types import Completion, Usage


class ChatClient(Protocol):
    """Minimal extension point for custom model providers."""

    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> Completion: ...


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _usage(data: Mapping[str, Any]) -> Usage:
    raw = data.get("usage")
    if not isinstance(raw, Mapping):
        return Usage()
    prompt = raw.get("prompt_tokens", raw.get("input_tokens", 0))
    completion = raw.get("completion_tokens", raw.get("output_tokens", 0))
    total = raw.get("total_tokens")
    cost = raw.get("cost_usd", raw.get("cost", 0.0))
    try:
        prompt_count = int(prompt or 0)
        completion_count = int(completion or 0)
        total_count = int(total) if total is not None else prompt_count + completion_count
        return Usage(prompt_count, completion_count, total_count, float(cost or 0.0))
    except (TypeError, ValueError):
        return Usage()


class OpenAICompatibleClient:
    """Async client for OpenRouter, ZenMux, and compatible chat endpoints."""

    def __init__(
        self,
        provider: ProviderConfig,
        *,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.getenv(provider.api_key_env, "")
        if not key:
            raise ConfigurationError(
                f"missing provider API key; set {provider.api_key_env} or pass api_key="
            )
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": f"panelwise-python/{__version__}",
            **provider.headers,
        }
        self._provider = provider
        self._client = httpx.AsyncClient(
            base_url=provider.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=provider.timeout_seconds,
            transport=transport,
        )

    async def __aenter__(self) -> "OpenAICompatibleClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "stream": False,
        }
        payload.update(self._provider.request_options)
        if extra:
            payload.update(extra)

        attempts = self._provider.retries + 1
        last_error: ProviderError | None = None
        for attempt in range(attempts):
            try:
                response = await self._client.post("chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = ProviderError(f"provider transport failure: {exc}", retryable=True)
            else:
                if response.is_success:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise ResponseError("provider returned non-JSON success response") from exc
                    return self._parse_completion(model, data)
                retryable = (
                    response.status_code in {408, 409, 425, 429} or response.status_code >= 500
                )
                detail = response.text.strip().replace("\n", " ")[:500]
                last_error = ProviderError(
                    f"provider returned HTTP {response.status_code}: {detail}",
                    status_code=response.status_code,
                    retryable=retryable,
                )
                if not retryable:
                    raise last_error
            if attempt + 1 < attempts:
                await asyncio.sleep(min(4.0, 0.5 * (2**attempt)))

        assert last_error is not None
        raise last_error

    @staticmethod
    def _parse_completion(requested_model: str, data: Any) -> Completion:
        if not isinstance(data, Mapping):
            raise ResponseError("provider response root must be an object")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            error = data.get("error")
            if error:
                raise ResponseError(f"provider response contains no choices: {error}")
            raise ResponseError("provider response contains no choices")
        first = choices[0]
        if not isinstance(first, Mapping) or not isinstance(first.get("message"), Mapping):
            raise ResponseError("provider response choice has no message")
        content = _message_text(first["message"].get("content"))
        if not content.strip():
            raise ResponseError("provider returned an empty message")
        model = str(data.get("model") or requested_model)
        return Completion(model=model, content=content, usage=_usage(data), raw=dict(data))


def client_from_config(
    provider: ProviderConfig,
    *,
    api_key: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(provider, api_key=api_key, transport=transport)
