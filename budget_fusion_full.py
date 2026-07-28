"""全量 budget fusion：panel=[GLM5.1+MiniMax M3+Qwen3.7-max]，synth=Opus 4.8，官方 grader，单 seed。
复用 output/budget_panel/ 已有的 10 题 panel 报告;其余 90 题新生成。断点续跑。
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics as st
import time

from draco_eval import config
from draco_eval.dataset import load_all
from draco_eval.fusion import PanelMember, _analyze, _synthesize
from draco_eval.official_judge import grade_official
from draco_eval.openrouter import CostTracker, OpenRouterClient
from draco_eval.research_agent import run_research_agent

PANEL = ["z-ai/glm-5.1", "minimax/minimax-m3", "qwen/qwen3.7-max"]
SYNTH = "anthropic/claude-opus-4.8"
FUSION_JUDGE = "google/gemini-3.1-pro-preview"
JUDGE_RUNS = 3
PANEL_DIR = os.getenv("BUDGET_PANEL_DIR", "output/budget_panel")
REC_DIR = os.getenv("BUDGET_FULL_REC_DIR", "output/budget_full/records")
TASK_CONCURRENCY = int(os.getenv("TASK_CONCURRENCY", "6"))


def _validate_cached_panel(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        cached = json.load(f)
    # Pre-flag panel records were produced when direct fetch was always enabled.
    cached_direct_fetch = cached.get("direct_fetch_enabled", True)
    if cached_direct_fetch is not config.RESEARCH_ENABLE_DIRECT_FETCH:
        raise RuntimeError(
            "cached panel direct_fetch_enabled does not match this run; "
            "choose a separate PANEL_DIR directory"
        )


def _validate_cached_record(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        old = json.load(f)
    if old.get("fused") is None:
        return
    cached_direct_fetch = old.get("direct_fetch_enabled", True)
    if cached_direct_fetch is not config.RESEARCH_ENABLE_DIRECT_FETCH:
        raise RuntimeError(
            "cached record direct_fetch_enabled does not match this run; "
            "choose a separate REC_DIR directory"
        )


def validate_resume_cache(tasks) -> None:
    for task in tasks:
        _validate_cached_panel(os.path.join(PANEL_DIR, f"{task.id}.json"))
        _validate_cached_record(os.path.join(REC_DIR, f"{task.id}.json"))


def summary_path() -> str:
    return os.path.join(os.path.dirname(os.path.normpath(REC_DIR)) or ".", "summary.json")


async def build_panel(client, task):
    p = os.path.join(PANEL_DIR, f"{task.id}.json")
    if os.path.exists(p):
        _validate_cached_panel(p)
        with open(p) as f:
            cached = json.load(f)
        return cached
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
    if not report:
        return None
    for _ in range(JUDGE_RUNS):
        s = await grade_official(client, task, report, config.DEFAULT_JUDGE_MODEL)
        if s is not None:
            return s
    return None


async def run_task(client, task, sem, cost):
    rp = os.path.join(REC_DIR, f"{task.id}.json")
    if os.path.exists(rp):
        with open(rp) as f:
            old = json.load(f)
        if old.get("fused") is not None:
            _validate_cached_record(rp)
            return old
    async with sem:
        t0 = time.monotonic()
        pr = await build_panel(client, task)
        panel = [PanelMember(m, rep, True, 0, 0, 0)
                 for m, rep in pr["panel_reports"].items() if pr["panel_ok"].get(m) and rep]
        fused_report = await _synthesize(client, task, panel, pr.get("analysis", {}), SYNTH) if panel else ""
        fused = await grade(client, task, fused_report)
        solo = {}
        for m, rep in pr["panel_reports"].items():
            solo[m] = await grade(client, task, rep)
    rec = {"task_id": task.id, "domain": task.domain, "fused": fused, "solo": solo,
           "direct_fetch_enabled": config.RESEARCH_ENABLE_DIRECT_FETCH,
           "fused_report": fused_report, "elapsed_s": round(time.monotonic() - t0, 1)}
    os.makedirs(REC_DIR, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(rec, f, ensure_ascii=False)
    return rec


async def main():
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    tasks = load_all()
    validate_resume_cache(tasks)
    print(f"全量 budget fusion: {len(tasks)} 题 | panel={PANEL} | synth={SYNTH} | 官方grader | 并发{TASK_CONCURRENCY}")
    cost = CostTracker()
    sem = asyncio.Semaphore(TASK_CONCURRENCY)
    recs = []
    t0 = time.monotonic()
    async with OpenRouterClient(config.DEFAULT_MAX_INFLIGHT, cost) as client:
        coros = [run_task(client, t, sem, cost) for t in tasks]
        done = 0
        for fut in asyncio.as_completed(coros):
            r = await fut
            recs.append(r); done += 1
            print(f"[{done}/{len(tasks)}] {r['domain']:22s} fused={r.get('fused')} ({r.get('elapsed_s','-')}s) cost=${cost.cost_usd:.1f}")

    fused = [r["fused"] for r in recs if r.get("fused") is not None]
    by_model = {}
    for r in recs:
        for m, v in r.get("solo", {}).items():
            if v is not None:
                by_model.setdefault(m, []).append(v)
    by_dom = {}
    for r in recs:
        if r.get("fused") is not None:
            by_dom.setdefault(r["domain"], []).append(r["fused"])

    print("\n" + "=" * 62)
    print(f"全量 budget fusion（{len(fused)} 题,官方grader,单seed）")
    print("=" * 62)
    print(f"  ★ BUDGET FUSION (Opus合成)   {st.mean(fused):.1f}")
    for m, vs in sorted(by_model.items(), key=lambda kv: -st.mean(kv[1])):
        print(f"    solo {m:30s} {st.mean(vs):.1f}  (n={len(vs)})")
    best = max((st.mean(vs) for vs in by_model.values()), default=0)
    print(f"  融合 vs 最强单: {st.mean(fused)-best:+.1f}")
    print("  按领域:")
    for d, vs in sorted(by_dom.items()):
        print(f"    {d:26s} {st.mean(vs):.1f} (n={len(vs)})")
    summ = {"budget_fusion": round(st.mean(fused), 2), "n": len(fused),
            "solo": {m: round(st.mean(vs), 2) for m, vs in by_model.items()},
            "by_domain": {d: round(st.mean(vs), 2) for d, vs in by_dom.items()},
            "elapsed_min": round((time.monotonic()-t0)/60, 1), "cost_usd": round(cost.cost_usd, 2),
            "direct_fetch_enabled": config.RESEARCH_ENABLE_DIRECT_FETCH}
    output_summary_path = summary_path()
    os.makedirs(os.path.dirname(output_summary_path) or ".", exist_ok=True)
    with open(output_summary_path, "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(f"\n  耗时 {summ['elapsed_min']}分 | 本次成本 ${cost.cost_usd:.2f} | -> {output_summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
