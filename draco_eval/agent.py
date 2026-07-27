"""被测系统（system under test）：默认 DeepSeek V4 Pro，经 OpenRouter，对一个 DRACO
任务产出一份深度研究报告。

联网默认用 OpenRouter 的 **server tools**（`openrouter:web_search` + `openrouter:web_fetch`）：
服务端 agentic 多步循环——模型自己决定搜什么、搜几次、要不要抓网页，OpenRouter 在服务端
跑完整个工具循环后把最终报告返回（一次 API 调用即可）。这正是 Fusion 给 panel 模型挂的
同一套工具（对齐论文的 max_tool_calls）。

防数据污染：把托管 DRACO rubric/答案的域名放进 web_search 的 excluded_domains，
模型就搜不到评分标准。

回退：AGENT_WEB_MODE=plugin 时改用旧的单轮 web 插件（已被 OpenRouter 废弃）。
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config
from .dataset import Task
from .openrouter import OpenRouterClient, message_text

AGENT_SYSTEM_PROMPT = (
    "You are an expert deep-research agent with web_search and web_fetch tools. "
    "Research thoroughly, then produce a comprehensive, accurate, well-structured "
    "report that fully answers the user's task.\n"
    "- Search iteratively: issue multiple targeted queries, fetch primary sources, "
    "and verify key facts before writing.\n"
    "- Ground every factual claim in current web sources; cite them inline with URLs.\n"
    "- Be precise with numbers, dates, named entities, and units.\n"
    "- Match the breadth and depth the task implies; address every sub-question, "
    "persona, comparison, and constraint stated.\n"
    "- Do NOT fabricate facts or citations. If something is uncertain, say so.\n"
    "- Do NOT pad with vague filler; verbose-but-empty answers are penalized.\n"
    "- Do NOT consult or cite benchmark/rubric pages; answer the question on its merits."
)


@dataclass
class AgentResult:
    task_id: str
    report: str
    ok: bool
    error: str = ""


def _server_tools(excluded_domains: list[str]) -> list[dict]:
    search_params: dict = {
        "engine": config.WEB_ENGINE,
        "max_results": config.WEB_MAX_RESULTS,
        "search_context_size": config.WEB_CONTEXT_SIZE,
    }
    if config.WEB_MAX_TOTAL_RESULTS > 0:  # 0 = 不设累计上限（推荐，难题才能写完）
        search_params["max_total_results"] = config.WEB_MAX_TOTAL_RESULTS
    if excluded_domains:
        search_params["excluded_domains"] = list(excluded_domains)
    return [
        {"type": "openrouter:web_search", "parameters": search_params},
        {"type": "openrouter:web_fetch"},
    ]


def _web_plugin(excluded_domains: list[str]) -> list[dict]:
    plugin: dict = {"id": "web", "max_results": config.WEB_MAX_RESULTS}
    if excluded_domains:  # 旧插件对排除域名支持不稳，带上以便支持时生效
        plugin["excluded_domains"] = list(excluded_domains)
    return [plugin]


async def run_agent(
    client: OpenRouterClient, task: Task, model: str, excluded_domains: list[str]
) -> AgentResult:
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": task.problem},
        ],
    }
    if config.AGENT_MAX_TOKENS > 0:  # 0 = 不传，用 provider 满额上限（推荐）
        payload["max_tokens"] = config.AGENT_MAX_TOKENS
    if config.AGENT_WEB_MODE == "plugin":
        payload["plugins"] = _web_plugin(excluded_domains)
    else:  # server_tool（默认）
        payload["tools"] = _server_tools(excluded_domains)

    # 多次尝试：高并发下模型偶发返回稀薄/退化输出（如原样回吐问题），
    # 报告太短就重试，取首个达标的；全不达标则返回最长那篇并标注。
    best = ""
    last_err = ""
    for _ in range(config.AGENT_MAX_ATTEMPTS):
        try:
            data = await client.chat("agent", payload)
            report = message_text(data).strip()
            if len(report) > len(best):
                best = report
            if len(report) >= config.AGENT_MIN_REPORT_CHARS:
                return AgentResult(task.id, report, ok=True)
            last_err = f"thin report ({len(report)} chars)"
        except Exception as e:  # noqa: BLE001 — 单题失败不应拖垮整批
            last_err = str(e)
    if best:
        # 有内容但始终偏短：仍返回（标 ok）但记下警告，由上层/人工判断
        return AgentResult(task.id, best, ok=True, error=f"warn: {last_err}")
    return AgentResult(task.id, "", ok=False, error=last_err or "no report")
