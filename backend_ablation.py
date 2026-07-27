"""你的点子的正确实现：自建 loop 里，搜索后端 = Exa vs "模型自带 native 搜索(做成一次专门调用)"。
外层 ReAct 循环不变(我们的)，只换搜索原语。同模型同题对比。
"""
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
BACKENDS = ["exa", "native"]
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
    res = {}
    async with OpenRouterClient(10, cost) as client:
        for backend in BACKENDS:
            config.RESEARCH_SEARCH_BACKEND = backend
            for task in tasks:
                print(f">>> backend={backend}  {task.domain} ...")
                ar = await run_research_agent(client, task, MODEL, config.BLOCKED_DOMAINS)
                s = await score(client, task, ar.report)
                res[(backend, task.domain)] = (
                    s.normalized_mean if s else None, ar.n_searches, ar.n_fetches, len(ar.report), ar.ok
                )

    print("\n" + "=" * 64)
    print(f"自建 loop 搜索后端 ablation（{MODEL}, DRACO normalized）")
    print("=" * 64)
    print(f"{'题':12s} {'Exa':>9s} {'native(自带)':>14s} {'差(native-exa)':>16s}")
    for d in DOMAINS:
        e = res[("exa", d)][0]
        n = res[("native", d)][0]
        diff = (n - e) if (e is not None and n is not None) else None
        print(f"{d:12s} {('%.1f'%e) if e is not None else '失败':>9s} "
              f"{('%.1f'%n) if n is not None else '失败':>14s} "
              f"{('%+.1f'%diff) if diff is not None else '-':>16s}")
    print("\n研究量明细 (norm / 搜索 / 抓取 / 字数 / ok):")
    for k, v in res.items():
        print(f"   {k[0]:7s} {k[1]:12s} {v[0]} / {v[1]}s / {v[2]}f / {v[3]} / {v[4]}")
    print(f"\n成本: ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
