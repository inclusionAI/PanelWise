"""单模型(无 fusion)排名:自建 agent(Exa, 参数拉满)跑 10 子集，官方 grader 评分。
断点续跑。
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics as st
import time

from draco_eval import config
from draco_eval.dataset import load_all, select_subset
from draco_eval.official_judge import grade_official
from draco_eval.openrouter import CostTracker, OpenRouterClient
from draco_eval.research_agent import run_research_agent

MODELS = [
    "google/gemini-3.5-flash",
    "qwen/qwen3.7-max",
    "minimax/minimax-m3",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
]
OUT = os.getenv("SOLO_RANKING_OUT", "output/solo_ranking")
TASK_CONCURRENCY = int(os.getenv("TASK_CONCURRENCY", "8"))


async def one(client, model, task, sem):
    safe = model.replace("/", "_")
    p = os.path.join(OUT, f"{safe}__{task.id}.json")
    if os.path.exists(p):
        with open(p) as f:
            r = json.load(f)
        if r.get("score") is not None:
            if r.get("direct_fetch_enabled", True) is not config.RESEARCH_ENABLE_DIRECT_FETCH:
                raise RuntimeError(
                    "cached record direct_fetch_enabled does not match this run; "
                    "set SOLO_RANKING_OUT to a separate directory"
                )
            return r
    async with sem:
        t0 = time.monotonic()
        ar = await run_research_agent(client, task, model, config.BLOCKED_DOMAINS)
        score = None
        for _ in range(3):  # 官方裁判判空重试
            score = await grade_official(client, task, ar.report, config.DEFAULT_JUDGE_MODEL)
            if score is not None:
                break
    r = {"model": model, "task_id": task.id, "domain": task.domain, "score": score,
         "ok": ar.ok, "chars": len(ar.report), "searches": ar.n_searches,
         "fetches": ar.n_fetches, "elapsed_s": round(time.monotonic() - t0, 1),
         "direct_fetch_enabled": config.RESEARCH_ENABLE_DIRECT_FETCH}
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(p, "w"), ensure_ascii=False)
    return r


async def main():
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    tasks = select_subset(load_all(), 10)
    for task in tasks:
        for model in MODELS:
            path = os.path.join(OUT, f"{model.replace('/', '_')}__{task.id}.json")
            if os.path.exists(path):
                with open(path) as f:
                    cached = json.load(f)
                if cached.get("score") is not None and (
                    cached.get("direct_fetch_enabled", True) is not config.RESEARCH_ENABLE_DIRECT_FETCH
                ):
                    raise RuntimeError(
                        "cached record direct_fetch_enabled does not match this run; "
                        "set SOLO_RANKING_OUT to a separate directory"
                    )
    print(f"单模型排名: {len(MODELS)} 模型 × {len(tasks)} 题 | reasoning={config.AGENT_REASONING_EFFORT} "
          f"max_steps={config.RESEARCH_MAX_STEPS} results={config.RESEARCH_SEARCH_RESULTS} | 官方grader")

    cost = CostTracker()
    sem = asyncio.Semaphore(TASK_CONCURRENCY)
    results = []
    async with OpenRouterClient(config.DEFAULT_MAX_INFLIGHT, cost) as client:
        coros = [one(client, m, t, sem) for m in MODELS for t in tasks]
        done = 0
        for fut in asyncio.as_completed(coros):
            r = await fut
            results.append(r)
            done += 1
            print(f"[{done}/{len(coros)}] {r['model'].split('/')[-1]:22s} {r['domain']:18s} "
                  f"score={r['score']} ok={r['ok']} cost=${cost.cost_usd:.1f}")

    by_model = {}
    for r in results:
        if r["score"] is not None:
            by_model.setdefault(r["model"], []).append(r["score"])

    print("\n" + "=" * 60)
    print(f"单模型排名（10 题，官方 grader，参数拉满，DRACO normalized）")
    print("=" * 60)
    rank = sorted(by_model.items(), key=lambda kv: -st.mean(kv[1]))
    for i, (m, vs) in enumerate(rank, 1):
        print(f"  {i}. {m:34s} {st.mean(vs):.1f}   (n={len(vs)})")
    summary = {m: round(st.mean(vs), 2) for m, vs in by_model.items()}
    summary["direct_fetch_enabled"] = config.RESEARCH_ENABLE_DIRECT_FETCH
    with open(OUT + "/summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n成本: ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
