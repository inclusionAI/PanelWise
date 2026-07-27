"""Provider-neutral chat client helpers."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Protocol

from . import config
from .openrouter import CostTracker, OpenRouterClient


class ChatClient(Protocol):
    async def chat(self, stage: str, payload: dict) -> dict:
        ...

    async def aclose(self) -> None:
        ...

    async def __aenter__(self) -> "ChatClient":
        ...

    async def __aexit__(self, *exc: object) -> None:
        ...


_response_capture: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "response_capture", default=None
)


@contextmanager
def capture_responses():
    """Capture provider response IDs for the current async task context."""
    captured: list[dict[str, Any]] = []
    token = _response_capture.set(captured)
    try:
        yield captured
    finally:
        _response_capture.reset(token)


def record_response(entry: dict[str, Any]) -> None:
    captured = _response_capture.get()
    if captured is not None:
        captured.append(entry)


def make_client(max_inflight: int, cost: CostTracker) -> ChatClient:
    provider = config.MODEL_PROVIDER.strip().lower()
    if provider == "openrouter":
        return OpenRouterClient(max_inflight, cost)
    if provider == "zenmux":
        from .zenmux import ZenMuxClient

        return ZenMuxClient(max_inflight, cost)
    raise SystemExit(
        f"不支持的 MODEL_PROVIDER={config.MODEL_PROVIDER!r}；请使用 openrouter 或 zenmux。"
    )
