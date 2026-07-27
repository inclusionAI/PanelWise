"""跑自建研究 agent（复现 server-tool），打印并保存完整 trajectory。

用法:
  EXA_API_KEY=... OPENROUTER_API_KEY=... \
  .venv/bin/python research_demo.py --domain Finance --model openai/gpt-5.5

  # 默认跑 Finance（之前在 OpenRouter server-tool 下翻车的那题）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from draco_eval import config
from draco_eval.dataset import load_all
from draco_eval.clients import capture_responses, make_client
from draco_eval.openrouter import CostTracker
from draco_eval.research_agent import run_research_agent


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="Finance")
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--model", default="openai/gpt-5.5")
    args = ap.parse_args()

    config.validate_provider_config()
    if not config.EXA_API_KEY:
        print("⚠️  EXA_API_KEY 未设置 —— web_search 会失败。web_fetch 仍可用。\n")

    tasks = load_all()
    if args.task_id:
        task = next(t for t in tasks if t.id == args.task_id)
    else:
        # 子集-10 里该领域那题（按 id 排序取第一个）
        from draco_eval.dataset import select_subset
        sub = select_subset(tasks, 10)
        task = next(t for t in sub if t.domain == args.domain)

    print(f"任务: {task.domain} / {task.id}")
    print(f"模型: {args.model}  | 步数上限: {config.RESEARCH_MAX_STEPS}")
    print(f"问题: {task.problem[:200]}...\n")

    cost = CostTracker()
    async with make_client(10, cost) as client:
        with capture_responses() as response_ids:
            res = await run_research_agent(client, task, args.model, config.BLOCKED_DOMAINS)

    print("=" * 70)
    print("TRAJECTORY（AI 的研究轨迹）")
    print("=" * 70)
    for s in res.trajectory:
        if s.kind == "assistant":
            calls = s.args.get("tool_calls") or []
            tag = f"调用 {calls}" if calls else ("【最终报告】" if s.text else "(空)")
            extra = " [forced]" if s.args.get("forced_finalize") else ""
            print(f"\n[step {s.step}] assistant {tag}{extra}")
            if s.text:
                print(f"    思考/正文: {s.text[:200]}")
        else:
            print(f"    └─ {s.tool}({json.dumps(s.args, ensure_ascii=False)[:100]}) -> {s.result_preview[:150]}")

    print("\n" + "=" * 70)
    print(f"结果: ok={res.ok}  报告 {len(res.report)} 字  | 搜索 {res.n_searches} 次, 抓取 {res.n_fetches} 次, 共 {res.n_steps} 步")
    print(f"成本: ${cost.cost_usd:.3f}")
    print(f"err: {res.error}")
    print("\n--- 报告开头 600 字 ---")
    print(res.report[:600])

    os.makedirs("output/trajectories", exist_ok=True)
    safe = args.model.split("/")[-1]
    provider_safe = config.MODEL_PROVIDER.strip().lower().replace("/", "_")
    out = f"output/trajectories/{task.domain.replace('/','_')}_{provider_safe}_{safe}.json"
    json.dump({
        "task_id": task.id, "domain": task.domain, "model": args.model,
        "provider": config.MODEL_PROVIDER, "response_ids": response_ids,
        "problem": task.problem, "report": res.report,
        "n_searches": res.n_searches, "n_fetches": res.n_fetches, "n_steps": res.n_steps,
        "ok": res.ok, "error": res.error, "cost_usd": round(cost.cost_usd, 4),
        "trajectory": res.trajectory_dicts(),
    }, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n完整 trajectory 已存 -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
