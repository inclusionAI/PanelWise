"""小 ablation：同自建循环、同模型、同题，只换 Exa 搜索档 auto vs fast，看 DRACO 分变不变。"""
from __future__ import annotations

import asyncio
import os

from draco_eval import config, scoring
from draco_eval.dataset import load_all, select_subset
from draco_eval.judge import judge_once
from draco_eval.openrouter import CostTracker, OpenRouterClient
from draco_eval.research_agent import run_research_agent

MODEL = "openai/gpt-5.5"
DOMAINS = ["Finance", "Academic"]
TIERS = ["auto", "fast"]
JUDGE_RUNS = 3


async def score(client, task, report):
    if not report:
        return None
    runs = await asyncio.gather(
        *[judge_once(client, task, report, config.DEFAULT_JUDGE_MODEL, i) for i in range(JUDGE_RUNS)]
    )
    return scoring.aggregate_task(task, list(runs))


async def main():
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    sub = select_subset(load_all(), 10)
    tasks = [next(t for t in sub if t.domain == d) for d in DOMAINS]

    cost = CostTracker()
    res = {}  # (tier, domain) -> (norm, searches, fetches, chars, ok)
    async with OpenRouterClient(10, cost) as client:
        for tier in TIERS:
            config.EXA_SEARCH_TYPE = tier
            for task in tasks:
                print(f">>> tier={tier}  {task.domain} ...")
                ar = await run_research_agent(client, task, MODEL, config.BLOCKED_DOMAINS)
                s = await score(client, task, ar.report)
                res[(tier, task.domain)] = (
                    s.normalized_mean if s else None, ar.n_searches, ar.n_fetches, len(ar.report), ar.ok
                )

    print("\n" + "=" * 60)
    print(f"Exa 搜索档 ablation（{MODEL}, 自建循环, DRACO normalized）")
    print("=" * 60)
    print(f"{'题':12s} {'auto':>8s} {'fast':>8s} {'差(fast-auto)':>14s}")
    for d in DOMAINS:
        a = res[("auto", d)][0]
        f = res[("fast", d)][0]
        diff = (f - a) if (a is not None and f is not None) else None
        print(f"{d:12s} {('%.1f'%a) if a is not None else '失败':>8s} "
              f"{('%.1f'%f) if f is not None else '失败':>8s} "
              f"{('%+.1f'%diff) if diff is not None else '-':>14s}")
    print("\n研究量明细 (norm / 搜索 / 抓取 / 字数 / ok):")
    for k, v in res.items():
        print(f"   {k[0]:5s} {k[1]:12s} {v[0]} / {v[1]}s / {v[2]}f / {v[3]} / {v[4]}")
    print(f"\n成本: ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
