"""直接调 OpenRouter 官方 Fusion，同题对上我们自建 fusion，看谁好。

openrouter/fusion 默认 Quality panel ≈ [claude-latest, gpt-latest, gemini-pro-latest]，
和我们自建 panel 基本一致 —— 这就是"他们的成品 vs 我们的复现"。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from draco_eval import config, scoring
from draco_eval.dataset import load_all, select_subset
from draco_eval.judge import judge_once
from draco_eval.openrouter import CostTracker, OpenRouterClient, message_text

JUDGE_RUNS = 3


async def score(client, task, report):
    if not report:
        return None
    runs = await asyncio.gather(
        *[judge_once(client, task, report, config.DEFAULT_JUDGE_MODEL, i) for i in range(JUDGE_RUNS)]
    )
    return scoring.aggregate_task(task, list(runs))


def our_fused(domain):
    p = f"output/fusion/{domain.replace('/','_')}.json"
    if os.path.exists(p):
        return json.load(open(p)).get("fused_score")
    return None


async def main():
    domains = (sys.argv[1] if len(sys.argv) > 1 else "Finance,Academic").split(",")
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

    sub = select_subset(load_all(), 10)
    tasks = [next(t for t in sub if t.domain == d) for d in domains]

    cost = CostTracker()
    rows = []
    async with OpenRouterClient(10, cost) as client:
        for task in tasks:
            print(f">>> 调 openrouter/fusion: {task.domain} ...")
            payload = {
                "model": "openrouter/fusion",
                "messages": [{"role": "user", "content": task.problem}],
            }
            try:
                data = await client.chat("or_fusion", payload)
                report = message_text(data).strip()
                served = data.get("model", "?")
            except Exception as e:  # noqa: BLE001
                report, served = "", f"ERROR: {e}"
            s = await score(client, task, report)
            rows.append({
                "domain": task.domain,
                "or_fusion": s.normalized_mean if s else None,
                "served_by": served, "chars": len(report),
                "ours": our_fused(task.domain),
            })
            os.makedirs("output/or_fusion", exist_ok=True)
            json.dump({"task_id": task.id, "domain": task.domain, "served_by": served,
                       "report": report, "score": s.normalized_mean if s else None},
                      open(f"output/or_fusion/{task.domain.replace('/','_')}.json", "w"),
                      ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("OpenRouter 官方 Fusion  vs  我们自建 Fusion（DRACO normalized）")
    print("=" * 70)
    print(f"{'题':14s} {'官方Fusion':>12s} {'我们自建':>10s} {'差(我们-官方)':>14s}  served_by")
    for r in rows:
        o, m = r["or_fusion"], r["ours"]
        d = (m - o) if (o is not None and m is not None) else None
        print(f"{r['domain']:14s} {('%.1f'%o) if o is not None else '失败':>12s} "
              f"{('%.1f'%m) if m is not None else '-':>10s} "
              f"{('%+.1f'%d) if d is not None else '-':>14s}  {r['served_by'][:30]} ({r['chars']}字)")
    print(f"\n本次成本(含官方fusion调用+评分): ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
