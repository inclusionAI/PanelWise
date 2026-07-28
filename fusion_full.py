"""全量 Fusion 评测（100 题，单 seed，断点续跑）。

panel = [gpt-5.5, claude-opus-4.8, gemini-3.1-pro] 自建小agent(直连 Exa)
  → fusion judge(gemini)结构化分析 → synthesizer(claude)合成 → DRACO 评分。
每题评:融合稿 + 3 个 panel 成员 solo(都 ×1 seed)。

断点续跑:每题结果写 output/fusion_full/records/<task_id>.json，重跑自动跳过已完成。
用法:
  .venv/bin/python fusion_full.py            # 全 100
  .venv/bin/python fusion_full.py --limit 3  # 只跑前 3 题(冒烟)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from draco_eval import config, scoring
from draco_eval.clients import capture_responses, make_client
from draco_eval.dataset import load_all
from draco_eval.fusion import run_fusion
from draco_eval.judge import judge_once
from draco_eval.openrouter import CostTracker

PANEL = ["openai/gpt-5.5", "anthropic/claude-opus-4.8", "google/gemini-3.1-pro-preview"]
SYNTH = "anthropic/claude-opus-4.8"
FUSION_JUDGE = "google/gemini-3.1-pro-preview"
JUDGE_RUNS = 1                       # 单 seed
TASK_CONCURRENCY = int(os.getenv("TASK_CONCURRENCY", "6"))
_provider = config.MODEL_PROVIDER.strip().lower()
_default_out = "output/fusion_full_v2" if _provider == "openrouter" else f"output/fusion_full_v2_{_provider}"
OUT_BASE = os.getenv("FUSION_OUT", _default_out)  # v2 = 存答案版
RECORDS_DIR = OUT_BASE + "/records"
SAVE_TRAJECTORY = os.getenv("FUSION_SAVE_TRAJECTORY", "").strip().lower() in {"1", "true", "yes", "on"}


def _validate_cached_record(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        old = json.load(f)
    if old.get("fused") is None:
        return
    # Pre-flag records were produced when direct fetch was always enabled.
    cached_direct_fetch = old.get("direct_fetch_enabled", True)
    if cached_direct_fetch is not config.RESEARCH_ENABLE_DIRECT_FETCH:
        raise RuntimeError(
            "cached record direct_fetch_enabled does not match this run; "
            "choose a separate FUSION_OUT directory"
        )


def validate_resume_cache(tasks) -> None:
    for task in tasks:
        _validate_cached_record(os.path.join(RECORDS_DIR, f"{task.id}.json"))


async def score(client, task, report, max_attempts=3):
    if not report:
        return None
    for _ in range(max_attempts):  # 单 seed 裁判偶发 whiff → 判空就重试
        runs = await asyncio.gather(
            *[judge_once(client, task, report, config.DEFAULT_JUDGE_MODEL, i) for i in range(JUDGE_RUNS)]
        )
        ts = scoring.aggregate_task(task, list(runs))
        if ts.n_runs:
            return ts.normalized_mean
    return None


async def run_task(client, task, sem, cost):
    rec_path = os.path.join(RECORDS_DIR, f"{task.id}.json")
    if os.path.exists(rec_path):
        with open(rec_path) as f:
            old = json.load(f)
        if old.get("fused") is not None:   # 续跑:只跳过成功的；失败的(fused=None)重跑
            _validate_cached_record(rec_path)
            return old

    async with sem:
        t0 = time.monotonic()
        with capture_responses() as response_ids:
            fr = await run_fusion(client, task, PANEL, SYNTH, FUSION_JUDGE, config.BLOCKED_DOMAINS)
            panel_scores = await asyncio.gather(*[score(client, task, m.report) for m in fr.panel])
            fused = await score(client, task, fr.fused_report)
        panel_fetch_stats = {
            m.model: {
                "n_searches": m.n_searches,
                "n_fetches": m.n_fetches,
                "n_blocked_fetches": m.n_blocked_fetches,
                "n_fetch_errors": m.n_fetch_errors,
                "fetched_urls": m.fetched_urls,
                "blocked_fetch_urls": m.blocked_fetch_urls,
            }
            for m in fr.panel
        }
        rec = {
            "task_id": task.id, "domain": task.domain,
            "provider": _provider,
            "base_url": config.ZENMUX_BASE_URL if _provider == "zenmux" else config.OPENROUTER_BASE_URL,
            "panel_models": PANEL,
            "synth_model": SYNTH,
            "fusion_judge_model": FUSION_JUDGE,
            "draco_judge_model": config.DEFAULT_JUDGE_MODEL,
            "search_backend": config.RESEARCH_SEARCH_BACKEND,
            "search_results": config.RESEARCH_SEARCH_RESULTS,
            "direct_fetch_enabled": config.RESEARCH_ENABLE_DIRECT_FETCH,
            "blocked_domains": config.BLOCKED_DOMAINS,
            "max_steps": config.RESEARCH_MAX_STEPS,
            "reasoning_effort": config.AGENT_REASONING_EFFORT,
            "analysis_schema_mode": (
                "prompt" if _provider == "zenmux" and config.ZENMUX_JSON_SCHEMA_FALLBACK
                else "native_strict"
            ),
            "response_ids": response_ids,
            "cost_reconciled": False,
            "fetch_stats": {
                "n_searches": sum(m.n_searches for m in fr.panel),
                "n_fetches": sum(m.n_fetches for m in fr.panel),
                "n_blocked_fetches": sum(m.n_blocked_fetches for m in fr.panel),
                "n_fetch_errors": sum(m.n_fetch_errors for m in fr.panel),
            },
            "panel_fetch_stats": panel_fetch_stats,
            "fused": fused,
            "solo": {m.model: ps for m, ps in zip(fr.panel, panel_scores)},
            "panel_ok": {m.model: m.ok for m in fr.panel},
            "fused_chars": len(fr.fused_report),
            # 存答案版：报告原文 + 结构化分析，便于以后用官方 grader 重评/复现
            "fused_report": fr.fused_report,
            "panel_reports": {m.model: m.report for m in fr.panel},
            "analysis": fr.analysis,
            "elapsed_s": round(time.monotonic() - t0, 1),
            "cost_so_far": round(cost.cost_usd, 2),
        }
        if SAVE_TRAJECTORY:
            rec["panel_trajectories"] = {m.model: m.trajectory for m in fr.panel}
        os.makedirs(RECORDS_DIR, exist_ok=True)
        with open(rec_path, "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        return rec


def aggregate_and_print(records):
    import statistics as st
    fused = [r["fused"] for r in records if r.get("fused") is not None]
    by_model = {}
    for r in records:
        for m, v in r.get("solo", {}).items():
            if v is not None:
                by_model.setdefault(m, []).append(v)
    by_domain = {}
    for r in records:
        if r.get("fused") is not None:
            by_domain.setdefault(r["domain"], []).append(r["fused"])

    print("\n" + "=" * 64)
    print(f"全量 Fusion 结果（{len(records)} 题，单 seed，DRACO normalized）")
    print("=" * 64)
    print(f"  ★ FUSION (融合)      {st.mean(fused):.1f}   (n={len(fused)})")
    for m, vs in sorted(by_model.items(), key=lambda kv: -st.mean(kv[1])):
        print(f"    solo {m:32s} {st.mean(vs):.1f}   (n={len(vs)})")
    best = max((st.mean(vs) for vs in by_model.values()), default=0)
    print(f"  融合 vs 最强单模型: {st.mean(fused) - best:+.1f}")
    print("\n  按领域(融合):")
    for d, vs in sorted(by_domain.items()):
        print(f"    {d:28s} {st.mean(vs):.1f}  (n={len(vs)})")
    return {
        "fusion": round(st.mean(fused), 2) if fused else None,
        "n": len(fused),
        "solo": {m: round(st.mean(vs), 2) for m, vs in by_model.items()},
        "by_domain": {d: round(st.mean(vs), 2) for d, vs in by_domain.items()},
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="validate the command without loading data or calling APIs")
    args = ap.parse_args()
    config.validate_provider_config(dry_run=args.dry_run)
    if args.dry_run:
        print("Dry run: configuration is valid. No dataset, model, search, or grader calls were made.")
        return

    tasks = load_all()
    if args.limit:
        tasks = tasks[: args.limit]
    validate_resume_cache(tasks)
    print(f"全量 fusion: {len(tasks)} 题 | panel={PANEL} | synth={SYNTH} | 并发={TASK_CONCURRENCY}")

    cost = CostTracker()
    sem = asyncio.Semaphore(TASK_CONCURRENCY)
    records = []
    t_start = time.monotonic()
    async with make_client(config.DEFAULT_MAX_INFLIGHT, cost) as client:
        coros = [run_task(client, t, sem, cost) for t in tasks]
        done = 0
        for fut in asyncio.as_completed(coros):
            rec = await fut
            records.append(rec)
            done += 1
            print(f"[{done}/{len(tasks)}] {rec['domain']:24s} fused={rec.get('fused')} "
                  f"({rec.get('elapsed_s','-')}s) cost=${cost.cost_usd:.1f}")

    summ = aggregate_and_print(records)
    summ["elapsed_min"] = round((time.monotonic() - t_start) / 60, 1)
    summ["total_cost_usd"] = round(cost.cost_usd, 2)
    summ["direct_fetch_enabled"] = config.RESEARCH_ENABLE_DIRECT_FETCH
    os.makedirs(OUT_BASE, exist_ok=True)
    with open(OUT_BASE + "/summary.json", "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(f"\n  耗时 {summ['elapsed_min']} 分 | 本次新增成本 ${cost.cost_usd:.2f}")
    print(f"  汇总 -> {OUT_BASE}/summary.json")


if __name__ == "__main__":
    asyncio.run(main())
