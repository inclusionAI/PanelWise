"""DRACO 打分公式。

单次评分（一次 judge run 的 verdicts）：
  raw_score        = Σ weight_i  (over criteria where met_i is True)   # 负权重命中即扣分
  normalized_score = clamp(raw_score / Σ(weight_i where weight_i>0), 0, 1) * 100
  pass_rate        = mean over criteria of passed_i
                     passed_i = (weight>0 and met) or (weight<0 and not met)

多次评分（judge_runs 次）：对每次的 normalized / pass_rate 取 mean ± std。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .dataset import Task
from .judge import JudgeRun


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class SingleScore:
    raw_score: float
    normalized_score: float  # 0-100
    pass_rate: float         # 0-100
    graded_criteria: int


def score_run(task: Task, run: JudgeRun) -> SingleScore:
    pos_total = task.positive_weight_total or 1  # 防 0 除
    raw = 0.0
    passed = 0
    graded = 0
    for c in task.criteria:
        if c.id not in run.verdicts:
            continue  # 裁判漏判的标准不计入（保守）
        graded += 1
        met = run.verdicts[c.id]
        if met:
            raw += c.weight
        is_pass = (c.weight > 0 and met) or (c.weight < 0 and not met)
        if is_pass:
            passed += 1
    normalized = _clamp01(raw / pos_total) * 100.0
    pass_rate = (passed / graded * 100.0) if graded else 0.0
    return SingleScore(raw, normalized, pass_rate, graded)


@dataclass
class TaskScore:
    task_id: str
    domain: str
    normalized_mean: float
    normalized_std: float
    pass_rate_mean: float
    pass_rate_std: float
    n_runs: int
    per_run: list[SingleScore]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def aggregate_task(task: Task, runs: list[JudgeRun]) -> TaskScore:
    singles = [score_run(task, r) for r in runs if r.ok and r.verdicts]
    n_mean, n_std = _mean_std([s.normalized_score for s in singles])
    p_mean, p_std = _mean_std([s.pass_rate for s in singles])
    return TaskScore(
        task_id=task.id,
        domain=task.domain,
        normalized_mean=round(n_mean, 2),
        normalized_std=round(n_std, 2),
        pass_rate_mean=round(p_mean, 2),
        pass_rate_std=round(p_std, 2),
        n_runs=len(singles),
        per_run=singles,
    )


@dataclass
class BenchmarkSummary:
    n_tasks: int
    normalized_mean: float
    pass_rate_mean: float
    by_domain: dict[str, dict[str, float]]


def aggregate_benchmark(scores: list[TaskScore]) -> BenchmarkSummary:
    scored = [s for s in scores if s.n_runs > 0]
    n_overall, _ = _mean_std([s.normalized_mean for s in scored])
    p_overall, _ = _mean_std([s.pass_rate_mean for s in scored])

    by_domain: dict[str, list[TaskScore]] = {}
    for s in scored:
        by_domain.setdefault(s.domain, []).append(s)
    domain_stats = {}
    for d, lst in sorted(by_domain.items()):
        nm, _ = _mean_std([s.normalized_mean for s in lst])
        pm, _ = _mean_std([s.pass_rate_mean for s in lst])
        domain_stats[d] = {
            "normalized_mean": round(nm, 2),
            "pass_rate_mean": round(pm, 2),
            "n_tasks": len(lst),
        }
    return BenchmarkSummary(
        n_tasks=len(scored),
        normalized_mean=round(n_overall, 2),
        pass_rate_mean=round(p_overall, 2),
        by_domain=domain_stats,
    )
