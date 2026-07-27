"""干净重测 native vs exa（直接构造 server-tool payload，精确控制 engine 和 excluded_domains）。

三臂：
  native_nofilter : engine=native, 不带 excluded_domains  → 真·native
  exa_nofilter    : engine=exa,    不带 excluded_domains  → 对照
  native_filter   : engine=native, 带 excluded_domains    → 验证"是否被回退成 Exa"
"""
from __future__ import annotations

import asyncio
import os

from draco_eval import config, scoring
from draco_eval.agent import AGENT_SYSTEM_PROMPT
from draco_eval.dataset import load_all, select_subset
from draco_eval.judge import judge_once
from draco_eval.openrouter import CostTracker, OpenRouterClient, message_text

MODEL = "openai/gpt-5.5"
DOMAINS = ["Finance", "Academic"]
JUDGE_RUNS = 3


def build_payload(task, engine, with_filter):
    params = {"engine": engine, "max_results": 5, "search_context_size": "high"}
    if with_filter:
        params["excluded_domains"] = config.BLOCKED_DOMAINS
    return {
        "model": MODEL,
        "tools": [
            {"type": "openrouter:web_search", "parameters": params},
            {"type": "openrouter:web_fetch"},
        ],
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": task.problem},
        ],
    }


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

    arms = [("native_nofilter", "native", False),
            ("exa_nofilter", "exa", False),
            ("native_filter", "native", True)]
    cost = CostTracker()
    res = {}
    dumped = False
    async with OpenRouterClient(10, cost) as client:
        for name, engine, wf in arms:
            for task in tasks:
                print(f">>> {name}  {task.domain} ...")
                try:
                    data = await client.chat("retest", build_payload(task, engine, wf))
                    rep = message_text(data).strip()
                    # 抓一次原始 usage/server_tool 字段，看能否看出引擎
                    if not dumped:
                        u = data.get("usage", {})
                        print("    [debug] usage keys:", list(u.keys()),
                              "| server_tool_use:", u.get("server_tool_use"))
                        dumped = True
                except Exception as e:  # noqa: BLE001
                    rep = ""
                    print("    ERR:", e)
                s = await score(client, task, rep)
                res[(name, task.domain)] = (s.normalized_mean if s else None, len(rep))

    print("\n" + "=" * 64)
    print(f"native vs exa 干净重测（{MODEL}, server-tool, DRACO normalized）")
    print("=" * 64)
    print(f"{'臂':18s} {'Finance':>9s} {'Academic':>9s}")
    for name, _, _ in arms:
        f = res.get((name, "Finance"), (None,))[0]
        a = res.get((name, "Academic"), (None,))[0]
        print(f"{name:18s} {('%.1f'%f) if f is not None else '失败':>9s} "
              f"{('%.1f'%a) if a is not None else '失败':>9s}")
    print("\n字数明细:")
    for k, v in res.items():
        print(f"   {k[0]:16s} {k[1]:10s} norm={v[0]} chars={v[1]}")
    print(f"\n成本: ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
