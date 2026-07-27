"""跑完整 Fusion 并评分：panel 各自 solo 分 vs 融合分，看融合是否 > 最强单模型。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from draco_eval import config, scoring
from draco_eval.clients import capture_responses, make_client
from draco_eval.dataset import load_all, select_subset
from draco_eval.fusion import run_fusion
from draco_eval.judge import judge_once
from draco_eval.openrouter import CostTracker

JUDGE_RUNS = 3


async def score(client, task, report):
    if not report:
        return None
    runs = await asyncio.gather(
        *[judge_once(client, task, report, config.DEFAULT_JUDGE_MODEL, i) for i in range(JUDGE_RUNS)]
    )
    return scoring.aggregate_task(task, list(runs))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="Finance,Academic")
    ap.add_argument("--panel", default="openai/gpt-5.5,anthropic/claude-opus-4.8,google/gemini-3.1-pro-preview")
    ap.add_argument("--synth", default="anthropic/claude-opus-4.8")
    ap.add_argument("--fusion-judge", default="google/gemini-3.1-pro-preview")
    args = ap.parse_args()

    config.validate_provider_config()
    panel_models = args.panel.split(",")
    domains = args.domains.split(",")

    all_tasks = load_all()
    sub = select_subset(all_tasks, 10)
    tasks = [next(t for t in sub if t.domain == d) for d in domains]

    print(f"Panel: {panel_models}")
    print(f"Synth: {args.synth} | Fusion-judge: {args.fusion_judge} | DRACO judge: {config.DEFAULT_JUDGE_MODEL} ×{JUDGE_RUNS}")
    print(f"题目: {domains}\n")

    cost = CostTracker()
    rows = []
    async with make_client(20, cost) as client:
        for task in tasks:
            print(f">>> 跑 fusion: {task.domain} ...")
            with capture_responses() as response_ids:
                fr = await run_fusion(client, task, panel_models, args.synth, args.fusion_judge, config.BLOCKED_DOMAINS)
                # 评每个 panel 成员 + 融合
                panel_scores = await asyncio.gather(*[score(client, task, m.report) for m in fr.panel])
                fused_score = await score(client, task, fr.fused_report)
            solo = {m.model: (s.normalized_mean if s else None) for m, s in zip(fr.panel, panel_scores)}
            fused_n = fused_score.normalized_mean if fused_score else None
            best_single = max([v for v in solo.values() if v is not None], default=None)
            rows.append({
                "domain": task.domain, "solo": solo, "fused": fused_n,
                "best_single": best_single,
                "panel_stats": {m.model: f"{m.n_searches}s/{m.n_fetches}f/{m.n_steps}st ok={m.ok}" for m in fr.panel},
                "analysis_keys": {k: len(v) if isinstance(v, list) else v for k, v in (fr.analysis or {}).items()},
            })
            # 存详情
            os.makedirs("output/fusion", exist_ok=True)
            provider_safe = config.MODEL_PROVIDER.strip().lower().replace("/", "_")
            out_path = f"output/fusion/{task.domain.replace('/','_')}_{provider_safe}.json"
            json.dump({
                "task_id": task.id, "domain": task.domain,
                "provider": config.MODEL_PROVIDER, "response_ids": response_ids,
                "panel": [{"model": m.model, "report": m.report, "ok": m.ok,
                           "searches": m.n_searches, "fetches": m.n_fetches, "steps": m.n_steps}
                          for m in fr.panel],
                "analysis": fr.analysis, "fused_report": fr.fused_report,
                "solo_scores": solo, "fused_score": fused_n,
            }, open(out_path, "w"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("FUSION 效果（DRACO normalized）")
    print("=" * 78)
    for r in rows:
        print(f"\n【{r['domain']}】")
        for m, v in r["solo"].items():
            print(f"   solo {m:34s} {('%.1f'%v) if v is not None else '失败':>6s}   [{r['panel_stats'][m]}]")
        bs = r["best_single"]; fu = r["fused"]
        delta = (fu - bs) if (fu is not None and bs is not None) else None
        print(f"   {'最强单模型':38s} {('%.1f'%bs) if bs is not None else '-':>6s}")
        print(f"   {'★ FUSION 融合':38s} {('%.1f'%fu) if fu is not None else '-':>6s}   "
              f"{('vs best %+.1f'%delta) if delta is not None else ''}  "
              f"{'✅ 融合>最强单' if (delta or 0)>0 else '❌ 没超过' if delta is not None else ''}")
        print(f"   分析: {r['analysis_keys']}")
    print(f"\n总成本: ${cost.cost_usd:.2f}")
    print("详情 -> output/fusion/<domain>.json")


if __name__ == "__main__":
    asyncio.run(main())
