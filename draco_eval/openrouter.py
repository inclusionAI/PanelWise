"""共享的 OpenRouter 异步客户端：限速信号量 + 退避重试 + 精确成本累计。

成本直接用 OpenRouter 返回的 usage.cost（请求时带 usage:{include:true}），
不靠本地硬编码价格，所以换模型也准。
"""
from __future__ import annotations

import asyncio
import json
import re
import ssl
from dataclasses import dataclass, field

import httpx

from . import config


@dataclass
class CostTracker:
    """asyncio 安全的成本/用量累计。"""

    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    by_stage: dict[str, dict[str, float]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, stage: str, usage: dict | None) -> None:
        async with self._lock:
            self.calls += 1
            s = self.by_stage.setdefault(
                stage, {"cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
            )
            s["calls"] += 1
            if not usage:
                return
            cost = float(usage.get("cost", 0.0) or 0.0)
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ct = int(usage.get("completion_tokens", 0) or 0)
            self.cost_usd += cost
            self.prompt_tokens += pt
            self.completion_tokens += ct
            s["cost_usd"] += cost
            s["prompt_tokens"] += pt
            s["completion_tokens"] += ct

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(self.cost_usd, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls": self.calls,
            "by_stage": {
                k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in self.by_stage.items()
            },
        }


class OpenRouterClient:
    """对 /chat/completions 的薄封装，带全局 HTTP 并发上限。"""

    def __init__(self, max_inflight: int, cost: CostTracker):
        self._sem = asyncio.Semaphore(max_inflight)
        self.cost = cost
        timeout = httpx.Timeout(
            config.HTTP_READ_TIMEOUT,
            connect=config.HTTP_CONNECT_TIMEOUT,
            read=config.HTTP_READ_TIMEOUT,
            write=60.0,
            pool=config.HTTP_READ_TIMEOUT,
        )
        limits = httpx.Limits(max_connections=max_inflight + 10, max_keepalive_connections=max_inflight)
        self._client = httpx.AsyncClient(
            base_url=config.OPENROUTER_BASE_URL,
            timeout=timeout,
            limits=limits,
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": config.OR_REFERER,
                "X-Title": config.OR_TITLE,
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def chat(self, stage: str, payload: dict) -> dict:
        """发一次 chat completion，返回完整 JSON。失败按退避重试。"""
        body = dict(payload)
        body.setdefault("usage", {"include": True})  # 让 OpenRouter 回传精确成本

        last_exc: Exception | None = None
        for attempt in range(config.HTTP_MAX_RETRIES + 1):
            try:
                async with self._sem:
                    resp = await self._client.post("/chat/completions", json=body)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"retryable {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                try:
                    data = resp.json()
                except (json.JSONDecodeError, ValueError) as je:
                    # 偶发返回非 JSON 体（截断/SSE/错误页）——当可重试
                    raise RuntimeError(f"non-JSON body: {je}") from je
                # OpenRouter 也可能在 200 体里夹带 error
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(f"OpenRouter error: {data['error']}")
                await self.cost.add(stage, data.get("usage"))
                return data
            except (httpx.HTTPError, ssl.SSLError, RuntimeError) as e:
                last_exc = e
                if attempt >= config.HTTP_MAX_RETRIES:
                    break
                await asyncio.sleep(min(2 ** attempt + 0.5 * attempt, 30))
        raise RuntimeError(f"[{stage}] 重试 {config.HTTP_MAX_RETRIES} 次后仍失败: {last_exc}")


def message_text(data: dict) -> str:
    """从 chat 响应里取出文本内容。"""
    try:
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # 某些模型返回 content 块数组
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content or "")
    except (KeyError, IndexError, TypeError):
        return ""


def extract_json(text: str) -> dict | None:
    """从文本里尽力解析出一个 JSON 对象（容忍 ```json 围栏和前后噪音）。"""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\{.*\})", text, re.DOTALL)  # 第一个 { 到最后一个 }
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None
