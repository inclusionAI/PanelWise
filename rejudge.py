"""复用 output/sweep/*/records_*.jsonl 里已生成的报告，用修好的裁判重新评分。

不重跑 agent（省下前沿模型的 input 大头），只重判 + 重算分。
按 (模型 × 题) 全异步并发。
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import sys

from draco_eval import config, scoring
from draco_eval.dataset import load_all
from draco_eval.judge import judge_once
from draco_eval.openrouter import CostTracker, OpenRouterClient

JUDGE_RUNS = int(os.getenv("REJUDGE_RUNS", "3"))
JUDGE_MODEL = config.DEFAULT_JUDGE_MODEL
CONCURRENCY = int(os.getenv("REJUDGE_CONCURRENCY", "30"))


async def rejudge_task(client, sem, task, report):
    async with sem:
        runs = await asyncio.gather(
            *[judge_once(client, task, report, JUDGE_MODEL, i) for i in range(JUDGE_RUNS)]
        )
    ts = scoring.aggregate_task(task, list(runs))
    return ts


async def main():
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    tasks_by_id = {t.id: t for t in load_all()}

    # 收集所有 (模型目录, 报告) ——只取 agent 成功且有报告的
    jobs = []  # (model_name, task, report)
    for rec_path in glob.glob("output/sweep/*/records_*.jsonl"):
        model_dir = os.path.basename(os.path.dirname(rec_path))
        for line in open(rec_path):
            r = json.loads(line)
            rep = r.get("report", "")
            if not rep or r.get("status") == "agent_failed":
                continue
            t = tasks_by_id.get(r["task_id"])
            if t:
                jobs.append((model_dir, t, rep))

    print(f"重判 {len(jobs)} 份报告 ({len(set(j[0] for j in jobs))} 个模型), "
          f"裁判={JUDGE_MODEL} ×{JUDGE_RUNS}, 并发={CONCURRENCY}")

    cost = CostTracker()
    sem = asyncio.Semaphore(CONCURRENCY)
    results: dict[str, list] = {}
    async with OpenRouterClient(CONCURRENCY + 5, cost) as client:
        coros = [rejudge_task(client, sem, t, rep) for (_, t, rep) in jobs]
        scored = await asyncio.gather(*coros)
    for (model_dir, _, _), ts in zip(jobs, scored):
        results.setdefault(model_dir, []).append(ts)

    # 汇总每个模型
    rows = []
    for model_dir, ts_list in results.items():
        bench = scoring.aggregate_benchmark(ts_list)
        model_name = model_dir.replace("_", "/", 1)
        rows.append({
            "agent_model": model_name,
            "n_reports": len(ts_list),
            "n_scored": bench.n_tasks,
            "normalized_mean": bench.normalized_mean,
            "pass_rate_mean": bench.pass_rate_mean,
            "by_domain": bench.by_domain,
        })
        out = os.path.join("output", "sweep", model_dir, "summary_rejudged.json")
        json.dump(rows[-1], open(out, "w"), ensure_ascii=False, indent=2)

    rows.sort(key=lambda r: r["normalized_mean"], reverse=True)
    print(f"\n{'model':34s} {'norm':>6s} {'pass':>6s} {'scored':>7s}/{'rep':<4s}")
    print("-" * 64)
    for r in rows:
        print(f"{r['agent_model']:34s} {r['normalized_mean']:6.1f} {r['pass_rate_mean']:6.1f} "
              f"{r['n_scored']:7d}/{r['n_reports']:<4d}")

    print("\n按领域 normalized：")
    domains = sorted({d for r in rows for d in r["by_domain"]})
    print("model".ljust(28) + "".join(f"{d[:9]:>10s}" for d in domains))
    for r in rows:
        line = r["agent_model"].split("/")[-1][:27].ljust(28)
        for d in domains:
            v = r["by_domain"].get(d, {}).get("normalized_mean")
            line += f"{'-' if v is None else f'{v:.0f}':>10s}"
        print(line)

    print(f"\n重判成本: ${cost.cost_usd:.2f}  ({cost.calls} 次裁判调用)")


if __name__ == "__main__":
    asyncio.run(main())
