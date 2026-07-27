"""异步编排：每题一个协程（agent → judge×N → 打分），全部并发跑。

并发模型：
  - task 级信号量 = concurrency（默认等于子集大小，即“一次跑完”）。
  - 每题内部把 judge_runs 次裁判并发发出。
  - HTTP 级信号量（OpenRouterClient 内部，= max_inflight）限制同时在飞的请求总数，
    防止 100 题 × (1 agent + 5 judge) 一起打爆 OpenRouter 限速。
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict

from . import agent as agent_mod
from . import dataset as ds
from . import judge as judge_mod
from . import scoring
from .config import RunConfig
from .openrouter import CostTracker, OpenRouterClient


async def _run_one(
    client: OpenRouterClient,
    task: ds.Task,
    cfg: RunConfig,
    task_sem: asyncio.Semaphore,
) -> dict:
    async with task_sem:
        t0 = time.monotonic()
        ar = await agent_mod.run_agent(client, task, cfg.agent_model, cfg.blocked_domains)

        record: dict = {
            "task_id": task.id,
            "domain": task.domain,
            "n_criteria": len(task.criteria),
            "agent_ok": ar.ok,
            "agent_error": ar.error,
            "report_chars": len(ar.report),
        }

        if not ar.ok:
            record["status"] = "agent_failed"
            record["elapsed_s"] = round(time.monotonic() - t0, 1)
            return record

        if cfg.no_judge:
            record["status"] = "agent_only"
            record["report"] = ar.report
            record["elapsed_s"] = round(time.monotonic() - t0, 1)
            return record

        runs = await asyncio.gather(
            *[
                judge_mod.judge_once(client, task, ar.report, cfg.judge_model, i)
                for i in range(cfg.judge_runs)
            ]
        )
        ts = scoring.aggregate_task(task, list(runs))
        record.update(
            {
                "status": "scored" if ts.n_runs > 0 else "judge_failed",
                "normalized_mean": ts.normalized_mean,
                "normalized_std": ts.normalized_std,
                "pass_rate_mean": ts.pass_rate_mean,
                "pass_rate_std": ts.pass_rate_std,
                "judge_runs_ok": ts.n_runs,
                "judge_runs_requested": cfg.judge_runs,
                "report": ar.report,
                "judge_errors": [r.error for r in runs if not r.ok][:3],
                "elapsed_s": round(time.monotonic() - t0, 1),
            }
        )
        return record


async def run(cfg: RunConfig) -> dict:
    cfg.validate()

    all_tasks = ds.load_all()
    tasks = ds.select_subset(all_tasks, cfg.subset)

    print(f"[dataset] 总 {len(all_tasks)} 题，选中 {len(tasks)} 题")
    domains: dict[str, int] = {}
    for t in tasks:
        domains[t.domain] = domains.get(t.domain, 0) + 1
    print(f"[dataset] 领域分布: {dict(sorted(domains.items()))}")
    print(
        f"[config] agent={cfg.agent_model} judge={cfg.judge_model} "
        f"judge_runs={cfg.judge_runs} concurrency={cfg.concurrency} "
        f"max_inflight={cfg.max_inflight} no_judge={cfg.no_judge}"
    )

    if cfg.dry_run:
        print("[dry-run] 不调用任何 API。下面是将被评测的题目 id：")
        for t in tasks:
            print(f"  {t.domain:28s} {t.id}  ({len(t.criteria)} criteria)")
        return {"dry_run": True, "selected": [t.id for t in tasks]}

    cost = CostTracker()
    task_sem = asyncio.Semaphore(cfg.concurrency)

    t_start = time.monotonic()
    records: list[dict] = []
    async with OpenRouterClient(cfg.max_inflight, cost) as client:
        coros = [_run_one(client, t, cfg, task_sem) for t in tasks]
        done = 0
        for fut in asyncio.as_completed(coros):
            rec = await fut
            records.append(rec)
            done += 1
            tag = rec.get("status", "?")
            nm = rec.get("normalized_mean")
            extra = f" norm={nm}" if nm is not None else ""
            print(
                f"[{done}/{len(tasks)}] {rec['domain']:24s} {tag}{extra} "
                f"({rec.get('elapsed_s','?')}s)  cost=${cost.cost_usd:.2f}"
            )

    elapsed = round(time.monotonic() - t_start, 1)

    # 汇总
    scored = [
        scoring.TaskScore(
            task_id=r["task_id"],
            domain=r["domain"],
            normalized_mean=r.get("normalized_mean", 0.0),
            normalized_std=r.get("normalized_std", 0.0),
            pass_rate_mean=r.get("pass_rate_mean", 0.0),
            pass_rate_std=r.get("pass_rate_std", 0.0),
            n_runs=r.get("judge_runs_ok", 0),
            per_run=[],
        )
        for r in records
        if r.get("status") == "scored"
    ]
    bench = scoring.aggregate_benchmark(scored)

    summary = {
        "agent_model": cfg.agent_model,
        "judge_model": cfg.judge_model if not cfg.no_judge else None,
        "judge_runs": cfg.judge_runs,
        "subset": cfg.subset,
        "n_selected": len(tasks),
        "n_scored": bench.n_tasks,
        "n_agent_failed": sum(1 for r in records if r.get("status") == "agent_failed"),
        "normalized_mean": bench.normalized_mean,
        "pass_rate_mean": bench.pass_rate_mean,
        "by_domain": bench.by_domain,
        "elapsed_s": elapsed,
        "cost": cost.summary(),
    }

    os.makedirs(cfg.out_dir, exist_ok=True)
    stamp = f"{cfg.agent_model.split('/')[-1]}_n{len(tasks)}"
    rec_path = os.path.join(cfg.out_dir, f"records_{stamp}.jsonl")
    sum_path = os.path.join(cfg.out_dir, f"summary_{stamp}.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"DRACO 结果  agent={cfg.agent_model}")
    if not cfg.no_judge:
        print(f"  Normalized score : {bench.normalized_mean:.1f}  (n={bench.n_tasks})")
        print(f"  Pass rate        : {bench.pass_rate_mean:.1f}")
        print("  按领域:")
        for d, st in bench.by_domain.items():
            print(f"    {d:26s} norm={st['normalized_mean']:.1f}  pass={st['pass_rate_mean']:.1f}  (n={st['n_tasks']})")
    print(f"  agent 失败题数    : {summary['n_agent_failed']}")
    print(f"  总耗时           : {elapsed}s")
    print(f"  总成本           : ${cost.cost_usd:.2f}")
    for stage, s in cost.summary()["by_stage"].items():
        print(f"    {stage:7s} ${s['cost_usd']:.2f}  ({s['calls']} calls, "
              f"{s['prompt_tokens']}+{s['completion_tokens']} tok)")
    print(f"  明细 -> {rec_path}")
    print(f"  汇总 -> {sum_path}")
    print("=" * 60)

    return summary
