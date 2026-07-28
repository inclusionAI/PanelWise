"""LLM-as-judge：对一份报告按 rubric 逐条给 MET/UNMET。

- 一次调用判完全部标准（成本远低于逐条调用），用 structured output 强约束 JSON。
- 每题独立判 judge_runs 次（论文默认 5），上层取均值±std。
- 负权重标准：MET 表示“该缺陷/错误确实出现在报告里” → 计分时扣分。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .dataset import Task
from .openrouter import OpenRouterClient, extract_json, message_text

# 兜底：即便 JSON 被截断，也从文本里逐个抠出 {"id":..,"met":..} 对，
# 拿到多少算多少（缺的标准在 scoring 里会被跳过，不影响已判部分）。
_PAIR_RE = re.compile(r'"id"\s*:\s*"([^"]+)"\s*,\s*"met"\s*:\s*(true|false)', re.IGNORECASE)


def _salvage_verdicts(text: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for cid, met in _PAIR_RE.findall(text or ""):
        out[cid] = met.lower() == "true"
    return out

JUDGE_SYSTEM_PROMPT = (
    "You are a meticulous evaluator grading a research report against a rubric of "
    "independent criteria. For EACH criterion, return a strict binary verdict:\n"
    "  met=true  -> the criterion's stated condition/requirement IS satisfied by the report.\n"
    "  met=false -> it is NOT satisfied.\n"
    "Rules:\n"
    "- Judge ONLY from the report's content, fact-checked against reality. Do not give "
    "credit for things the report does not actually state.\n"
    "- A criterion with a NEGATIVE weight describes an error/flaw to penalize; for those, "
    "met=true means the flaw IS present in the report.\n"
    "- Be strict: if the report does not clearly satisfy a criterion, mark met=false.\n"
    "- Return a verdict for EVERY criterion id, no more and no fewer."
)

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "draco_grades",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "grades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "met": {"type": "boolean"},
                        },
                        "required": ["id", "met"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["grades"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class JudgeRun:
    verdicts: dict[str, bool]  # criterion_id -> met
    ok: bool
    error: str = ""


def _criteria_block(task: Task) -> str:
    lines = []
    for c in task.criteria:
        polarity = "NEGATIVE(flaw)" if c.negative else "POSITIVE"
        lines.append(f'- id="{c.id}" weight={c.weight} [{polarity}]: {c.requirement}')
    return "\n".join(lines)


def _build_payload(task: Task, report: str, model: str, seed_hint: int) -> dict:
    user = (
        f"# Research task\n{task.problem}\n\n"
        f"# Report to grade\n{report}\n\n"
        f"# Rubric criteria ({len(task.criteria)} total)\n{_criteria_block(task)}\n\n"
        "Return JSON: {\"grades\":[{\"id\":...,\"met\":true|false}, ...]} with one entry "
        "per criterion id above."
    )
    payload: dict = {
        "model": model,
        "response_format": _RESPONSE_FORMAT,
        # seed_hint 只为让多次运行有差异（真正的独立性来自采样）；注入到 system 里
        "messages": [
            {"role": "system", "content": f"{JUDGE_SYSTEM_PROMPT}\n(grading pass #{seed_hint})"},
            {"role": "user", "content": user},
        ],
    }
    if config.JUDGE_MAX_TOKENS > 0:  # 0 = 不传，用 provider 满额上限（推荐）
        payload["max_tokens"] = config.JUDGE_MAX_TOKENS
    return payload


async def judge_once(
    client: OpenRouterClient, task: Task, report: str, model: str, run_idx: int
) -> JudgeRun:
    payload = _build_payload(task, report, model, run_idx)
    try:
        data = await client.chat("judge", payload)
        text = message_text(data)
        verdicts: dict[str, bool] = {}
        parsed = extract_json(text)
        if parsed and isinstance(parsed.get("grades"), list):
            for g in parsed["grades"]:
                try:
                    verdicts[str(g["id"])] = bool(g["met"])
                except (KeyError, TypeError):
                    continue
        if not verdicts:  # JSON 解析失败或被截断 → 正则兜底抠出已完成的判决
            verdicts = _salvage_verdicts(text)
        if not verdicts:
            return JudgeRun({}, ok=False, error=f"unparseable judge output: {text[:200]}")
        return JudgeRun(verdicts, ok=True)
    except Exception as e:  # noqa: BLE001
        return JudgeRun({}, ok=False, error=str(e))
