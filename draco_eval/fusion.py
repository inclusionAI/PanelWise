"""复现 Fusion 管线（自建版，去 OpenRouter 黑盒）：

  N 个自建研究 agent 并行研究同一题
      → fusion judge 出结构化分析（consensus / contradictions / partial_coverage /
        unique_insights / blind_spots，对比而非合并）
      → synthesizer 基于分析合成终稿。

每个 panel 成员就是我们验证过的 research_agent（自管 ReAct 循环 + Exa + 修好的 fetch）。
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from dataclasses import dataclass, field

from . import config
from .clients import ChatClient
from .dataset import Task
from .messages import ModelInput, normalize_messages
from .openrouter import extract_json, message_text
from .research_agent import run_research_agent, run_research_messages

# ---------- fusion judge（结构化对比分析）----------
FUSION_JUDGE_PROMPT = (
    "You are a meticulous analyst comparing several independent research reports answering the "
    "SAME task. Do NOT merge them. Produce a structured comparison:\n"
    "- consensus: points all or most reports agree on (higher confidence).\n"
    "- contradictions: where reports disagree, with each report's stance.\n"
    "- partial_coverage: important points only some reports cover.\n"
    "- unique_insights: valuable points unique to a single report.\n"
    "- blind_spots: aspects of the task NO report addressed.\n"
    "Reference reports as 'Response 1', 'Response 2', etc. Be specific and factual."
)

_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "fusion_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "consensus": {"type": "array", "items": {"type": "string"}},
                "contradictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string"},
                            "stances": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "response": {"type": "string"},
                                        "stance": {"type": "string"},
                                    },
                                    "required": ["response", "stance"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["topic", "stances"],
                        "additionalProperties": False,
                    },
                },
                "partial_coverage": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "responses": {"type": "array", "items": {"type": "string"}},
                            "point": {"type": "string"},
                        },
                        "required": ["responses", "point"],
                        "additionalProperties": False,
                    },
                },
                "unique_insights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "response": {"type": "string"},
                            "insight": {"type": "string"},
                        },
                        "required": ["response", "insight"],
                        "additionalProperties": False,
                    },
                },
                "blind_spots": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["consensus", "contradictions", "partial_coverage",
                         "unique_insights", "blind_spots"],
            "additionalProperties": False,
        },
    },
}

# ---------- synthesizer ----------
SYNTH_PROMPT = (
    "You are synthesizing a single definitive answer from several independent research reports "
    "and a structured analysis of their agreements, contradictions, and gaps.\n"
    "Write ONE comprehensive, accurate, well-structured report that:\n"
    "- treats consensus points as high-confidence;\n"
    "- resolves each contradiction, preferring the better-sourced / more precise claim and "
    "stating which you chose and why when it matters;\n"
    "- incorporates the unique insights;\n"
    "- addresses the blind spots if you can, or flags them;\n"
    "- cites sources inline with URLs; is precise with numbers, dates, entities.\n"
    "Fully answer the original task. Do not mention 'Response 1/2' — write a clean final report."
)


@dataclass
class PanelMember:
    model: str
    report: str
    ok: bool
    n_searches: int
    n_fetches: int
    n_steps: int
    n_blocked_fetches: int = 0
    n_fetch_errors: int = 0
    fetched_urls: list[str] = field(default_factory=list)
    blocked_fetch_urls: list[str] = field(default_factory=list)
    trajectory: list[dict] = field(default_factory=list)


@dataclass
class FusionResult:
    task_id: str
    panel: list[PanelMember]
    analysis: dict
    fused_report: str
    synth_model: str
    fused_ok: bool
    error: str = ""


def _responses_block(panel: list[PanelMember]) -> str:
    parts = []
    for i, m in enumerate(panel, 1):
        parts.append(f"### Response {i} (model: {m.model})\n{m.report}")
    return "\n\n".join(parts)


def _fetch_stats(result) -> dict:
    fetched_urls = []
    blocked_urls = []
    n_fetch_errors = 0
    for step in result.trajectory:
        if step.kind != "tool_result" or step.tool != "web_fetch":
            continue
        url = str(step.args.get("url") or "")
        if url:
            fetched_urls.append(url)
        preview = step.result_preview or step.error
        if preview.startswith("ERROR"):
            n_fetch_errors += 1
        if "ERROR: blocked benchmark/rubric domain" in preview:
            blocked_urls.append(url)
    return {
        "n_blocked_fetches": len(blocked_urls),
        "n_fetch_errors": n_fetch_errors,
        "fetched_urls": fetched_urls,
        "blocked_fetch_urls": blocked_urls,
    }


async def _analyze(client: ChatClient, task: Task, panel: list[PanelMember], judge_model: str) -> dict:
    user = (
        f"# Task\n{task.problem}\n\n"
        f"# {len(panel)} independent reports\n{_responses_block(panel)}\n\n"
        "Produce the structured comparison JSON."
    )
    payload = {
        "model": judge_model,
        "response_format": _ANALYSIS_SCHEMA,
        "messages": [
            {"role": "system", "content": FUSION_JUDGE_PROMPT},
            {"role": "user", "content": user},
        ],
    }
    data = await client.chat("fusion_judge", payload)
    parsed = extract_json(message_text(data))
    return parsed or {}


async def _synthesize(client, task, panel, analysis, synth_model) -> str:
    user = (
        f"# Original task\n{task.problem}\n\n"
        f"# Independent reports\n{_responses_block(panel)}\n\n"
        f"# Structured analysis\n{json.dumps(analysis, ensure_ascii=False, indent=2)}\n\n"
        "Now write the final synthesized report."
    )
    payload = {
        "model": synth_model,
        "messages": [
            {"role": "system", "content": SYNTH_PROMPT},
            {"role": "user", "content": user},
        ],
    }
    if config.AGENT_REASONING_EFFORT:  # 拉满
        payload["reasoning"] = {"effort": config.AGENT_REASONING_EFFORT}
    data = await client.chat("synthesizer", payload)
    return message_text(data).strip()


async def run_fusion(
    client: ChatClient,
    task: Task,
    panel_models: list[str],
    synth_model: str,
    judge_model: str,
    excluded_domains: list[str],
) -> FusionResult:
    # 1) panel 并行研究
    results = await asyncio.gather(
        *[run_research_agent(client, task, m, excluded_domains) for m in panel_models]
    )
    panel = []
    for m, r in zip(panel_models, results):
        stats = _fetch_stats(r)
        panel.append(
            PanelMember(
                m,
                r.report,
                r.ok,
                r.n_searches,
                r.n_fetches,
                r.n_steps,
                stats["n_blocked_fetches"],
                stats["n_fetch_errors"],
                stats["fetched_urls"],
                stats["blocked_fetch_urls"],
                r.trajectory_dicts(),
            )
        )
    good = [m for m in panel if m.ok and m.report]
    if not good:
        return FusionResult(task.id, panel, {}, "", synth_model, False, "no usable panel reports")

    # 2) 结构化分析
    try:
        analysis = await _analyze(client, task, good, judge_model)
    except Exception as e:  # noqa: BLE001 — 分析失败则降级：synthesizer 直接看原始报告
        analysis = {"_error": str(e)}

    # 3) 合成
    try:
        fused = await _synthesize(client, task, good, analysis, synth_model)
        return FusionResult(task.id, panel, analysis, fused, synth_model, bool(fused))
    except Exception as e:  # noqa: BLE001
        return FusionResult(task.id, panel, analysis, "", synth_model, False, str(e))


async def _analyze_messages(
    client: ChatClient,
    original_messages: list,
    panel: list[PanelMember],
    judge_model: str,
) -> dict:
    user = (
        f"# {len(panel)} independent reports\n{_responses_block(panel)}\n\n"
        "Produce the structured comparison JSON."
    )
    payload = {
        "model": judge_model,
        "response_format": _ANALYSIS_SCHEMA,
        "messages": [
            {"role": "system", "content": FUSION_JUDGE_PROMPT},
            *deepcopy(original_messages),
            {"role": "user", "content": user},
        ],
    }
    data = await client.chat("fusion_judge", payload)
    parsed = extract_json(message_text(data))
    return parsed or {}


async def _synthesize_messages(
    client: ChatClient,
    original_messages: list,
    panel: list[PanelMember],
    analysis: dict,
    synth_model: str,
) -> str:
    user = (
        f"# Independent reports\n{_responses_block(panel)}\n\n"
        f"# Structured analysis\n{json.dumps(analysis, ensure_ascii=False, indent=2)}\n\n"
        "Now write the final synthesized report."
    )
    payload = {
        "model": synth_model,
        "messages": [
            {"role": "system", "content": SYNTH_PROMPT},
            *deepcopy(original_messages),
            {"role": "user", "content": user},
        ],
    }
    if config.AGENT_REASONING_EFFORT:
        payload["reasoning"] = {"effort": config.AGENT_REASONING_EFFORT}
    data = await client.chat("synthesizer", payload)
    return message_text(data).strip()


async def run_fusion_messages(
    client: ChatClient,
    model_input: ModelInput,
    panel_models: list[str],
    synth_model: str,
    judge_model: str,
    excluded_domains: list[str],
    request_id: str = "",
) -> FusionResult:
    """Run the provider-compatible message-array Fusion pipeline."""
    original_messages = normalize_messages(model_input)
    results = await asyncio.gather(*[
        run_research_messages(client, deepcopy(original_messages), model, excluded_domains, request_id)
        for model in panel_models
    ])
    panel = []
    for model, result in zip(panel_models, results):
        stats = _fetch_stats(result)
        panel.append(PanelMember(
            model, result.report, result.ok, result.n_searches, result.n_fetches, result.n_steps,
            stats["n_blocked_fetches"], stats["n_fetch_errors"], stats["fetched_urls"],
            stats["blocked_fetch_urls"], result.trajectory_dicts(),
        ))
    good = [member for member in panel if member.ok and member.report]
    if not good:
        return FusionResult(request_id, panel, {}, "", synth_model, False, "no usable panel reports")
    try:
        analysis = await _analyze_messages(client, original_messages, good, judge_model)
    except Exception as e:  # noqa: BLE001
        analysis = {"_error": str(e)}
    try:
        fused = await _synthesize_messages(client, original_messages, good, analysis, synth_model)
        return FusionResult(request_id, panel, analysis, fused, synth_model, bool(fused))
    except Exception as e:  # noqa: BLE001
        return FusionResult(request_id, panel, analysis, "", synth_model, False, str(e))
