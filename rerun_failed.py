"""定点补跑：server-tool 模式下出空/稀薄报告的 (模型×题)，用更大 max_tokens 重跑，
结果合并回原 __st 记录并重建汇总，便于 compare_modes 直接对比。
"""
from __future__ import annotations

import asyncio
import glob
import json
import os

from draco_eval import config, scoring
from draco_eval.agent import run_agent
from draco_eval.dataset import load_all
from draco_eval.judge import judge_once
from draco_eval.openrouter import CostTracker, OpenRouterClient

config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
config.AGENT_WEB_MODE = "server_tool"  # 强制 agentic
JUDGE_RUNS = 3
JUDGE_MODEL = config.DEFAULT_JUDGE_MODEL

# 要补跑的 (模型 -> __st 目录)
TARGETS = {
    "google/gemini-3.1-pro-preview": "output/sweep/google_gemini-3.1-pro-preview__st",
    "anthropic/claude-opus-4.8": "output/sweep/anthropic_claude-opus-4.8__st",
}
# 两个翻车的题（subset-10 里的 Finance / General Knowledge）
TASK_IDS = [
    "10e75d21-9e92-4878-b691-02fc87ae5d54",  # Finance（上限全关后验证）
]


def _summary_from_records(model, recs):
    scored = [
        scoring.TaskScore(
            task_id=r["task_id"], domain=r["domain"],
            normalized_mean=r.get("normalized_mean", 0.0),
            normalized_std=r.get("normalized_std", 0.0),
            pass_rate_mean=r.get("pass_rate_mean", 0.0),
            pass_rate_std=r.get("pass_rate_std", 0.0),
            n_runs=r.get("judge_runs_ok", 0), per_run=[],
        )
        for r in recs if r.get("status") == "scored"
    ]
    bench = scoring.aggregate_benchmark(scored)
    return {
        "agent_model": model, "subset": 10, "n_selected": len(recs),
        "n_scored": bench.n_tasks,
        "n_agent_failed": sum(1 for r in recs if r.get("status") == "agent_failed"),
        "normalized_mean": bench.normalized_mean,
        "pass_rate_mean": bench.pass_rate_mean,
        "by_domain": bench.by_domain,
    }


async def rerun_one(client, model, task):
    ar = await run_agent(client, task, model, config.BLOCKED_DOMAINS)
    rec = {
        "task_id": task.id, "domain": task.domain, "n_criteria": len(task.criteria),
        "agent_ok": ar.ok, "agent_error": ar.error, "report_chars": len(ar.report),
    }
    if not ar.ok:
        rec["status"] = "agent_failed"
        return rec
    runs = await asyncio.gather(
        *[judge_once(client, task, ar.report, JUDGE_MODEL, i) for i in range(JUDGE_RUNS)]
    )
    ts = scoring.aggregate_task(task, list(runs))
    rec.update({
        "status": "scored" if ts.n_runs > 0 else "judge_failed",
        "normalized_mean": ts.normalized_mean, "normalized_std": ts.normalized_std,
        "pass_rate_mean": ts.pass_rate_mean, "pass_rate_std": ts.pass_rate_std,
        "judge_runs_ok": ts.n_runs, "report": ar.report,
    })
    return rec


async def main():
    tasks_by_id = {t.id: t for t in load_all()}
    targets = [tasks_by_id[i] for i in TASK_IDS]
    cost = CostTracker()
    async with OpenRouterClient(20, cost) as client:
        for model, d in TARGETS.items():
            print(f"\n>>> 补跑 {model}  ({len(targets)} 题, max_tokens={config.AGENT_MAX_TOKENS})")
            new = await asyncio.gather(*[rerun_one(client, model, t) for t in targets])
            for r in new:
                print(f"    {r['domain']:20s} {r['status']:13s} chars={r.get('report_chars')} norm={r.get('normalized_mean')}")

            recf = glob.glob(d + "/records_*.jsonl")[0]
            recs = [json.loads(l) for l in open(recf)]
            merged = {r["task_id"]: r for r in recs}
            for r in new:
                merged[r["task_id"]] = r  # 替换翻车的那条
            recs = list(merged.values())
            with open(recf, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            summ = _summary_from_records(model, recs)
            sumf = glob.glob(d + "/summary_*.json")
            sumf = [x for x in sumf if "rejudged" not in x][0]
            json.dump(summ, open(sumf, "w"), ensure_ascii=False, indent=2)
            print(f"    -> 更新后总分 norm={summ['normalized_mean']} (scored {summ['n_scored']}/10)")

    print(f"\n补跑成本: ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
