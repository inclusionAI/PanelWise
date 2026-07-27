"""用官方 rubric 包(paper-instruments/rubric)评分，generate_fn 走我们的 Gemini(OpenRouter)。
用 PerCriterionOneShotGrader：官方详细 prompt、一次判全部(与我们的批量判同粒度，只差 prompt)。
"""
from __future__ import annotations

from rubric import OneShotOutput
from rubric.autograders.per_criterion_one_shot_grader import PerCriterionOneShotGrader
from rubric.rubric import Rubric
from rubric.types import Criterion as RCriterion

from . import config
from .dataset import Task
from .openrouter import OpenRouterClient, extract_json, message_text


def _make_oneshot_fn(client: OpenRouterClient, model: str):
    async def fn(system_prompt: str, user_prompt: str, **kwargs) -> OneShotOutput:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        last = ""
        for _ in range(3):
            data = await client.chat("official_judge", payload)
            text = message_text(data)
            last = text
            parsed = extract_json(text)
            if parsed and isinstance(parsed.get("criteria_evaluations"), list):
                try:
                    return OneShotOutput(**parsed)
                except Exception:  # noqa: BLE001
                    pass
        # 实在解析不出 → 返回空评估(grader 会判 0)；上层会重试整次
        raise ValueError(f"official judge unparseable: {last[:200]}")

    return fn


async def grade_official(client: OpenRouterClient, task: Task, report: str, model: str) -> float | None:
    """用官方 one-shot grader 评一次，返回 normalized 0-100；失败返回 None。"""
    if not report:
        return None
    criteria = [RCriterion(weight=c.weight, requirement=c.requirement) for c in task.criteria]
    rb = Rubric(criteria)
    grader = PerCriterionOneShotGrader(generate_fn=_make_oneshot_fn(client, model))
    try:
        rep = await rb.grade(to_grade=report, autograder=grader, query=task.problem)
        return round(rep.score * 100.0, 2)  # .score 是 0-1 归一化
    except Exception:  # noqa: BLE001
        return None
