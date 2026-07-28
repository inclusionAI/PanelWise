"""用官方 rubric grader(one-shot, 官方 prompt)重评 v2 已存的 100 份报告(融合 + 3 panel),
不重跑 agent。断点续跑。
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import statistics as st

from draco_eval import config
from draco_eval.dataset import load_all
from draco_eval.official_judge import grade_official
from draco_eval.openrouter import CostTracker, OpenRouterClient

FUSION_OUT = os.getenv("FUSION_OUT", "output/fusion_full_v2")
V2_RECORDS = FUSION_OUT + "/records"
OUT_DIR = os.getenv("REGRADE_OUT_DIR", FUSION_OUT + "/official_grades")
CONCURRENCY = int(os.getenv("REGRADE_CONCURRENCY", "12"))


async def regrade_task(client, task, rec, sem):
    out_path = os.path.join(OUT_DIR, f"{task.id}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            og = json.load(f)
        if og.get("fused") is not None:
            source_direct_fetch = rec.get("direct_fetch_enabled", True)
            if og.get("direct_fetch_enabled", True) is source_direct_fetch:
                return og
    async with sem:
        fused = await grade_official(client, task, rec.get("fused_report", ""), config.DEFAULT_JUDGE_MODEL)
        solo = {}
        for m, rep in rec.get("panel_reports", {}).items():
            solo[m] = await grade_official(client, task, rep, config.DEFAULT_JUDGE_MODEL)
    og = {"task_id": task.id, "domain": task.domain, "fused": fused, "solo": solo,
          "direct_fetch_enabled": rec.get("direct_fetch_enabled", True)}
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(og, open(out_path, "w"), ensure_ascii=False, indent=2)
    return og


async def main():
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    tasks_by_id = {t.id: t for t in load_all()}
    recs = [json.load(open(f)) for f in glob.glob(V2_RECORDS + "/*.json")]
    print(f"官方 grader 重评 {len(recs)} 题(融合+3panel),并发 {CONCURRENCY}")

    cost = CostTracker()
    sem = asyncio.Semaphore(CONCURRENCY)
    out = []
    async with OpenRouterClient(config.DEFAULT_MAX_INFLIGHT, cost) as client:
        coros = [regrade_task(client, tasks_by_id[r["task_id"]], r, sem) for r in recs if r["task_id"] in tasks_by_id]
        done = 0
        for fut in asyncio.as_completed(coros):
            og = await fut
            out.append(og)
            done += 1
            print(f"[{done}/{len(coros)}] {og['domain']:22s} fused(官方)={og.get('fused')}  cost=${cost.cost_usd:.1f}")

    fused = [o["fused"] for o in out if o.get("fused") is not None]
    by_model = {}
    for o in out:
        for m, v in o.get("solo", {}).items():
            if v is not None:
                by_model.setdefault(m, []).append(v)
    by_domain = {}
    for o in out:
        if o.get("fused") is not None:
            by_domain.setdefault(o["domain"], []).append(o["fused"])

    print("\n" + "=" * 60)
    print(f"官方 grader 重评结果（{len(fused)} 题，one-shot，DRACO normalized）")
    print("=" * 60)
    print(f"  ★ FUSION (官方裁判)   {st.mean(fused):.1f}")
    for m, vs in sorted(by_model.items(), key=lambda kv: -st.mean(kv[1])):
        print(f"    solo {m:32s} {st.mean(vs):.1f}")
    best = max((st.mean(vs) for vs in by_model.values()), default=0)
    print(f"  融合 vs 最强单: {st.mean(fused) - best:+.1f}")
    print("  按领域(融合,官方):")
    for d, vs in sorted(by_domain.items()):
        print(f"    {d:28s} {st.mean(vs):.1f} (n={len(vs)})")
    summ = {"fusion_official": round(st.mean(fused), 2), "n": len(fused),
            "solo_official": {m: round(st.mean(vs), 2) for m, vs in by_model.items()},
            "by_domain_official": {d: round(st.mean(vs), 2) for d, vs in by_domain.items()},
            "cost_usd": round(cost.cost_usd, 2)}
    fetch_settings = sorted({o.get("direct_fetch_enabled", True) for o in out})
    summ["direct_fetch_enabled"] = fetch_settings[0] if len(fetch_settings) == 1 else None
    summ["direct_fetch_settings"] = fetch_settings
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "summary_official.json")
    with open(summary_path, "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(f"\n成本: ${cost.cost_usd:.2f}  | 汇总 -> {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
