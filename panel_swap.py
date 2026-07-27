"""换 panel 第三人:复用 budget_panel 里的 GLM+MiniMax 报告,只重跑第三个模型,
重新 analyze + Opus 合成 + 官方 grader。SWAP_MODEL 由环境变量指定。
用法: SWAP_MODEL=deepseek/deepseek-v4-pro .venv/bin/python panel_swap.py
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics as st

from draco_eval import config
from draco_eval.dataset import load_all
from draco_eval.fusion import PanelMember, _analyze, _synthesize
from draco_eval.official_judge import grade_official
from draco_eval.openrouter import CostTracker, OpenRouterClient
from draco_eval.research_agent import run_research_agent

BASE = ["z-ai/glm-5.1", "minimax/minimax-m3"]   # 复用
SWAP = os.environ["SWAP_MODEL"]
SYNTH = "anthropic/claude-opus-4.8"
FUSION_JUDGE = "google/gemini-3.1-pro-preview"
JUDGE_RUNS = 3
PANEL_DIR = "output/budget_panel"
SWAP_REPORT_DIR = "output/panel_swap/reports"   # 存第三人报告(可复用)
OUT_DIR = "output/panel_swap"
SAFE = SWAP.replace("/", "_")


async def swap_report(client, task):
    p = os.path.join(SWAP_REPORT_DIR, f"{SAFE}__{task.id}.json")
    if os.path.exists(p):
        return json.load(open(p))["report"]
    ar = await run_research_agent(client, task, SWAP, config.BLOCKED_DOMAINS)
    os.makedirs(SWAP_REPORT_DIR, exist_ok=True)
    json.dump({"report": ar.report, "ok": ar.ok}, open(p, "w"), ensure_ascii=False)
    return ar.report


async def grade(client, task, report):
    if not report:
        return None
    for _ in range(JUDGE_RUNS):
        s = await grade_official(client, task, report, config.DEFAULT_JUDGE_MODEL)
        if s is not None:
            return s
    return None


async def run_task(client, task, base_rec, sem):
    rp = os.path.join(OUT_DIR, f"{SAFE}__{task.id}.json")
    if os.path.exists(rp):
        r = json.load(open(rp))
        if r.get("fused") is not None:
            return r
    async with sem:
        # 复用 GLM+MiniMax
        panel = [PanelMember(m, base_rec["panel_reports"][m], True, 0, 0, 0)
                 for m in BASE if base_rec["panel_ok"].get(m) and base_rec["panel_reports"].get(m)]
        rep3 = await swap_report(client, task)
        if rep3:
            panel.append(PanelMember(SWAP, rep3, True, 0, 0, 0))
        analysis = await _analyze(client, task, panel, FUSION_JUDGE) if panel else {}
        fused_report = await _synthesize(client, task, panel, analysis, SYNTH) if panel else ""
        fused = await grade(client, task, fused_report)
    r = {"task_id": task.id, "domain": task.domain, "swap": SWAP, "fused": fused,
         "n_panel": len(panel)}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(r, open(rp, "w"), ensure_ascii=False)
    return r


async def main():
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    tasks = load_all()
    base = {}
    for f in __import__("glob").glob(PANEL_DIR + "/*.json"):
        r = json.load(open(f)); base[r["task_id"]] = r
    tasks = [t for t in tasks if t.id in base]  # 只跑有 GLM+MiniMax 基线的题
    print(f"换第三人 = {SWAP} | 复用 GLM+MiniMax | {len(tasks)} 题 | synth=Opus 官方grader")

    cost = CostTracker()
    sem = asyncio.Semaphore(int(os.getenv("TASK_CONCURRENCY", "8")))
    recs = []
    async with OpenRouterClient(config.DEFAULT_MAX_INFLIGHT, cost) as client:
        coros = [run_task(client, t, base[t.id], sem) for t in tasks]
        done = 0
        for fut in asyncio.as_completed(coros):
            r = await fut; recs.append(r); done += 1
            print(f"[{done}/{len(tasks)}] {r['domain']:22s} fused={r.get('fused')} cost=${cost.cost_usd:.1f}")

    fused = [r["fused"] for r in recs if r.get("fused") is not None]
    print("\n" + "=" * 56)
    print(f"panel = GLM + MiniMax + {SWAP.split('/')[-1]}  (Opus 合成)")
    print(f"  ★ FUSION: {st.mean(fused):.1f}  (n={len(fused)})")
    json.dump({"swap": SWAP, "fusion": round(st.mean(fused), 2), "n": len(fused),
               "cost_usd": round(cost.cost_usd, 2)},
              open(OUT_DIR + f"/summary_{SAFE}.json", "w"), ensure_ascii=False, indent=2)
    print(f"  成本: ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
