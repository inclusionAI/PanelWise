"""ZenMux OpenAI-compatible chat client.

This mirrors the public surface of OpenRouterClient while keeping provider-specific
payload normalization isolated here.
"""
from __future__ import annotations

import asyncio
import copy
import json
import ssl
from typing import Any

import httpx

from . import config
from .clients import record_response
from .openrouter import CostTracker

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class ZenMuxClient:
    """Thin async wrapper around ZenMux /chat/completions."""

    provider_name = "zenmux"
    supports_openrouter_server_tools = False

    def __init__(self, max_inflight: int, cost: CostTracker):
        self._sem = asyncio.Semaphore(max_inflight)
        self.cost = cost
        self._model_map = self._parse_model_map(config.ZENMUX_MODEL_MAP)
        timeout = httpx.Timeout(
            config.HTTP_READ_TIMEOUT,
            connect=config.HTTP_CONNECT_TIMEOUT,
            read=config.HTTP_READ_TIMEOUT,
            write=60.0,
            pool=config.HTTP_READ_TIMEOUT,
        )
        limits = httpx.Limits(max_connections=max_inflight + 10, max_keepalive_connections=max_inflight)
        self._client = httpx.AsyncClient(
            base_url=config.ZENMUX_BASE_URL,
            timeout=timeout,
            limits=limits,
            headers={
                "Authorization": f"Bearer {config.ZENMUX_API_KEY}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ZenMuxClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def chat(self, stage: str, payload: dict) -> dict:
        """Send one ZenMux chat completion request with OpenRouter-like retries."""
        body = self._normalize_payload(payload)
        # 对齐 OpenRouterClient：请求带回 usage，便于 CostTracker 统计成本/用量。
        body.setdefault("usage", {"include": True})
        original_model = payload.get("model")
        remapped_model = body.get("model")

        last_exc: Exception | None = None
        for attempt in range(config.HTTP_MAX_RETRIES + 1):
            try:
                async with self._sem:
                    resp = await self._client.post("/chat/completions", json=body)
                if resp.status_code in _RETRYABLE_STATUSES:
                    raise httpx.HTTPStatusError(
                        f"retryable {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                try:
                    data = resp.json()
                except (json.JSONDecodeError, ValueError) as je:
                    raise RuntimeError(f"ZenMux non-JSON body: {je}") from je
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(f"ZenMux error: {data['error']}")
                await self.cost.add(stage, data.get("usage"))
                self._capture_response(stage, data, original_model, remapped_model)
                return data
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                if status not in _RETRYABLE_STATUSES:
                    raise
                last_exc = e
                if attempt >= config.HTTP_MAX_RETRIES:
                    break
                await asyncio.sleep(min(2 ** attempt + 0.5 * attempt, 30))
            except (httpx.TransportError, ssl.SSLError) as e:
                last_exc = e
                if attempt >= config.HTTP_MAX_RETRIES:
                    break
                await asyncio.sleep(min(2 ** attempt + 0.5 * attempt, 30))
        raise RuntimeError(f"[{stage}] ZenMux 重试 {config.HTTP_MAX_RETRIES} 次后仍失败: {last_exc}")

    def _normalize_payload(self, payload: dict) -> dict:
        body = copy.deepcopy(payload)
        response_format = body.get("response_format")
        if response_format is not None and config.ZENMUX_JSON_SCHEMA_FALLBACK:
            body.pop("response_format", None)
            self._inject_response_format_instruction(body, response_format)
        if "model" in body:
            body["model"] = self._remap_model(str(body["model"]))
        return body

    @staticmethod
    def _inject_response_format_instruction(body: dict, response_format: Any) -> None:
        schema = response_format
        if (
            isinstance(response_format, dict)
            and isinstance(response_format.get("json_schema"), dict)
            and isinstance(response_format["json_schema"].get("schema"), dict)
        ):
            schema = response_format["json_schema"]["schema"]
        instruction = (
            "\n\nReturn only a valid JSON object matching this schema. "
            "Do not wrap it in Markdown fences or include prose. "
            "Do not include any fields that are not allowed by the schema.\n"
            f"Schema:\n{json.dumps(schema, ensure_ascii=False)}"
        )
        messages = body.get("messages")
        if not isinstance(messages, list):
            body["messages"] = [{"role": "user", "content": instruction.strip()}]
            return
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = content + instruction
                return
            if isinstance(content, list):
                content.append({"type": "text", "text": instruction.strip()})
                return
        messages.append({"role": "user", "content": instruction.strip()})

    def _remap_model(self, model: str) -> str:
        return self._model_map.get(model, model)

    @staticmethod
    def _parse_model_map(raw: str) -> dict[str, str]:
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"ZENMUX_MODEL_MAP 必须是 JSON 对象: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit("ZENMUX_MODEL_MAP 必须是 JSON 对象，例如 {'old/model':'new/model'}。")
        out: dict[str, str] = {}
        for key, value in parsed.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SystemExit("ZENMUX_MODEL_MAP 的 key/value 都必须是字符串模型名。")
            out[key] = value
        return out

    @staticmethod
    def _response_id(data: dict[str, Any]) -> str:
        for key in ("id", "generation_id", "generationId"):
            value = data.get(key)
            if value:
                return str(value)
        return ""

    def _capture_response(
        self,
        stage: str,
        data: dict[str, Any],
        original_model: Any,
        remapped_model: Any,
    ) -> None:
        response_id = self._response_id(data)
        record_response({
            "provider": self.provider_name,
            "stage": stage,
            "response_id": response_id,
            "reconcilable": bool(response_id),
            "model": remapped_model,
            "original_model": original_model,
            "usage": data.get("usage") or {},
        })
