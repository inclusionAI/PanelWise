"""同 harness(OpenRouter server-tool)、同模型、同题，只换搜索引擎：native vs exa。
隔离"搜索后端"本身对 DRACO 分的影响。
"""
from __future__ import annotations

import asyncio
import os

from draco_eval import agent, config, scoring
from draco_eval.dataset import load_all, select_subset
from draco_eval.judge import judge_once
from draco_eval.openrouter import CostTracker, OpenRouterClient

MODEL = "openai/gpt-5.5"
DOMAINS = ["Finance", "Academic"]
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
    config.AGENT_WEB_MODE = "server_tool"
    sub = select_subset(load_all(), 10)
    tasks = [next(t for t in sub if t.domain == d) for d in DOMAINS]

    cost = CostTracker()
    results = {}  # (engine, domain) -> (norm, ok, chars)
    async with OpenRouterClient(10, cost) as client:
        for engine in ["native", "exa"]:
            config.WEB_ENGINE = engine
            for task in tasks:
                print(f">>> {MODEL}  engine={engine}  {task.domain} ...")
                ar = await agent.run_agent(client, task, MODEL, config.BLOCKED_DOMAINS)
                s = await score(client, task, ar.report)
                results[(engine, task.domain)] = (
                    s.normalized_mean if s else None, ar.ok, len(ar.report)
                )

    print("\n" + "=" * 60)
    print(f"搜索引擎对比（{MODEL}, server-tool, DRACO normalized）")
    print("=" * 60)
    print(f"{'题':12s} {'native':>10s} {'exa':>10s} {'差(exa-native)':>14s}")
    for d in DOMAINS:
        nv = results[("native", d)][0]
        ex = results[("exa", d)][0]
        diff = (ex - nv) if (nv is not None and ex is not None) else None
        print(f"{d:12s} {('%.1f'%nv) if nv is not None else '失败':>10s} "
              f"{('%.1f'%ex) if ex is not None else '失败':>10s} "
              f"{('%+.1f'%diff) if diff is not None else '-':>14s}")
    print("\n字数/ok 明细:")
    for k, v in results.items():
        print(f"   {k[0]:7s} {k[1]:12s} norm={v[0]} ok={v[1]} chars={v[2]}")
    print(f"\n成本: ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
