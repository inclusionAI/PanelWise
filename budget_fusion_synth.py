"""budget panel = [GLM5.1 + MiniMax M3 + Qwen3.7-max]，比哪个合成器(merge)最好(含 Opus)。

Phase1: 这三个模型在 10 题上各跑自建 agent(存报告) + fusion judge 分析(只做一次,复用)。
Phase2: 多个 synthesizer 候选在固定 panel 报告上合成 → 官方 grader 评分 → 排名。
断点续跑。
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics as st

from draco_eval import config
from draco_eval.dataset import load_all, select_subset
from draco_eval.fusion import PanelMember, _analyze, _synthesize
from draco_eval.official_judge import grade_official
from draco_eval.openrouter import CostTracker, OpenRouterClient
from draco_eval.research_agent import run_research_agent

PANEL = ["z-ai/glm-5.1", "minimax/minimax-m3", "qwen/qwen3.7-max"]
SYNTHS = [
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "z-ai/glm-5.1",
    "minimax/minimax-m3",
    "qwen/qwen3.7-max",
]
FUSION_JUDGE = "google/gemini-3.1-pro-preview"
JUDGE_RUNS = 3
PANEL_DIR = os.getenv("BUDGET_PANEL_DIR", "output/budget_panel")
SYNTH_DIR = os.getenv("BUDGET_SYNTH_DIR", "output/budget_synth")


def validate_resume_cache(tasks) -> None:
    for task in tasks:
        p = os.path.join(PANEL_DIR, f"{task.id}.json")
        if os.path.exists(p):
            with open(p) as f:
                cached = json.load(f)
            # Pre-flag panel records were produced when direct fetch was always enabled.
            if cached.get("direct_fetch_enabled", True) is not config.RESEARCH_ENABLE_DIRECT_FETCH:
                raise RuntimeError(
                    "cached panel direct_fetch_enabled does not match this run; "
                    "choose a separate PANEL_DIR directory"
                )
        for synth in SYNTHS:
            synth_path = os.path.join(SYNTH_DIR, f"{synth.replace('/', '_')}__{task.id}.json")
            if not os.path.exists(synth_path):
                continue
            with open(synth_path) as f:
                synth_record = json.load(f)
            if synth_record.get("score") is not None and (
                synth_record.get("direct_fetch_enabled", True) is not config.RESEARCH_ENABLE_DIRECT_FETCH
            ):
                raise RuntimeError(
                    "cached synth direct_fetch_enabled does not match this run; "
                    "set BUDGET_SYNTH_DIR to a separate directory"
                )


async def build_panel(client, task, sem):
    """跑 budget panel 三个 agent + 分析,存盘复用。"""
    p = os.path.join(PANEL_DIR, f"{task.id}.json")
    if os.path.exists(p):
        with open(p) as f:
            cached = json.load(f)
        # Pre-flag panel records were produced when direct fetch was always enabled.
        if cached.get("direct_fetch_enabled", True) is not config.RESEARCH_ENABLE_DIRECT_FETCH:
            raise RuntimeError(
                "cached panel direct_fetch_enabled does not match this run; "
                "choose a separate PANEL_DIR directory"
            )
        return cached
    async with sem:
        results = await asyncio.gather(
            *[run_research_agent(client, task, m, config.BLOCKED_DOMAINS) for m in PANEL]
        )
        panel = [PanelMember(m, r.report, r.ok, r.n_searches, r.n_fetches, r.n_steps)
                 for m, r in zip(PANEL, results)]
        good = [m for m in panel if m.ok and m.report]
        analysis = await _analyze(client, task, good, FUSION_JUDGE) if good else {}
    rec = {"task_id": task.id, "domain": task.domain,
           "direct_fetch_enabled": config.RESEARCH_ENABLE_DIRECT_FETCH,
           "panel_reports": {m.model: m.report for m in panel},
           "panel_ok": {m.model: m.ok for m in panel}, "analysis": analysis}
    os.makedirs(PANEL_DIR, exist_ok=True)
    json.dump(rec, open(p, "w"), ensure_ascii=False)
    return rec


async def grade(client, task, report):
    for _ in range(JUDGE_RUNS):
        s = await grade_official(client, task, report, config.DEFAULT_JUDGE_MODEL)
        if s is not None:
            return s
    return None


async def synth_one(client, synth, task, rec, sem):
    p = os.path.join(SYNTH_DIR, f"{synth.replace('/','_')}__{task.id}.json")
    if os.path.exists(p):
        with open(p) as f:
            r = json.load(f)
        if r.get("score") is not None:
            if r.get("direct_fetch_enabled", True) is not config.RESEARCH_ENABLE_DIRECT_FETCH:
                raise RuntimeError(
                    "cached synth direct_fetch_enabled does not match this run; "
                    "set BUDGET_SYNTH_DIR to a separate directory"
                )
            return r
    panel = [PanelMember(m, rep, True, 0, 0, 0)
             for m, rep in rec["panel_reports"].items() if rec["panel_ok"].get(m) and rep]
    async with sem:
        fused = await _synthesize(client, task, panel, rec.get("analysis", {}), synth)
        score = await grade(client, task, fused)
    r = {"synth": synth, "task_id": task.id, "domain": task.domain, "score": score,
         "chars": len(fused), "direct_fetch_enabled": config.RESEARCH_ENABLE_DIRECT_FETCH}
    os.makedirs(SYNTH_DIR, exist_ok=True)
    json.dump(r, open(p, "w"), ensure_ascii=False)
    return r


async def main():
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    tasks = select_subset(load_all(), 10)
    validate_resume_cache(tasks)
    print(f"budget panel={PANEL}\n比 {len(SYNTHS)} 个合成器 × {len(tasks)} 题 | reasoning={config.AGENT_REASONING_EFFORT}")

    cost = CostTracker()
    sem = asyncio.Semaphore(8)
    async with OpenRouterClient(config.DEFAULT_MAX_INFLIGHT, cost) as client:
        print(">>> Phase1: 生成 budget panel 报告 + 分析 ...")
        panel_recs = await asyncio.gather(*[build_panel(client, t, sem) for t in tasks])
        print(f"    panel 就绪(cost=${cost.cost_usd:.1f}). 各题 panel 报告字数:")
        for r in panel_recs:
            print(f"      {r['domain']:22s} " + " ".join(f"{m.split('/')[-1]}={len(rep)}" for m, rep in r["panel_reports"].items()))

        print(">>> Phase2: 扫合成器 ...")
        recmap = {r["task_id"]: r for r in panel_recs}
        coros = [synth_one(client, s, t, recmap[t.id], sem) for s in SYNTHS for t in tasks]
        results = []
        done = 0
        for fut in asyncio.as_completed(coros):
            r = await fut
            results.append(r); done += 1
            print(f"  [{done}/{len(coros)}] synth={r['synth'].split('/')[-1]:20s} {r['domain']:16s} score={r['score']} cost=${cost.cost_usd:.1f}")

    by = {}
    for r in results:
        if r["score"] is not None:
            by.setdefault(r["synth"], []).append(r["score"])
    print("\n" + "=" * 62)
    print(f"合成器排名（panel=GLM+MiniMax+Qwen 固定，官方 grader，融合稿分）")
    print("=" * 62)
    for i, (m, vs) in enumerate(sorted(by.items(), key=lambda kv: -st.mean(kv[1])), 1):
        print(f"  {i}. {m:34s} {st.mean(vs):.1f}  (n={len(vs)})")
    summary = {m: round(st.mean(vs), 2) for m, vs in by.items()}
    summary["direct_fetch_enabled"] = config.RESEARCH_ENABLE_DIRECT_FETCH
    with open(SYNTH_DIR + "/summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n成本: ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
