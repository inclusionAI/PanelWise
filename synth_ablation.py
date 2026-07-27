"""合成器(merge)选型 ablation:固定 panel 报告(复用 v2 的 Opus/GPT/Gemini 三份报告+分析)，
只换 synthesizer 模型，官方 grader 评融合稿，看谁合成得最好。
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import statistics as st

from draco_eval import config
from draco_eval.dataset import load_all, select_subset
from draco_eval.fusion import PanelMember, _synthesize
from draco_eval.official_judge import grade_official
from draco_eval.openrouter import CostTracker, OpenRouterClient

SYNTHS = [
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "z-ai/glm-5.1",
    "minimax/minimax-m3",
    "deepseek/deepseek-v4-pro",
]
JUDGE_RUNS = 3  # 官方裁判判空重试上限
OUT = "output/synth_ablation"


def load_panel_inputs():
    """从 v2 记录里取 10-子集那 10 题的 panel 报告 + 分析。"""
    sub = {t.id: t for t in select_subset(load_all(), 10)}
    items = []
    for f in glob.glob("output/fusion_full_v2/records/*.json"):
        r = json.load(open(f))
        if r["task_id"] in sub:
            panel = [PanelMember(m, rep, True, 0, 0, 0) for m, rep in r["panel_reports"].items()]
            items.append((sub[r["task_id"]], panel, r.get("analysis", {})))
    return items


async def grade(client, task, report):
    for _ in range(JUDGE_RUNS):
        s = await grade_official(client, task, report, config.DEFAULT_JUDGE_MODEL)
        if s is not None:
            return s
    return None


async def one(client, synth, task, panel, analysis, sem):
    safe = synth.replace("/", "_")
    p = os.path.join(OUT, f"{safe}__{task.id}.json")
    if os.path.exists(p):
        r = json.load(open(p))
        if r.get("score") is not None:
            return r
    async with sem:
        fused = await _synthesize(client, task, panel, analysis, synth)
        score = await grade(client, task, fused)
    r = {"synth": synth, "task_id": task.id, "domain": task.domain, "score": score, "chars": len(fused)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(r, open(p, "w"), ensure_ascii=False)
    return r


async def main():
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    items = load_panel_inputs()
    print(f"合成器 ablation: {len(SYNTHS)} 个 synth × {len(items)} 题(固定 frontier panel 报告) | reasoning={config.AGENT_REASONING_EFFORT}")

    cost = CostTracker()
    sem = asyncio.Semaphore(10)
    results = []
    async with OpenRouterClient(config.DEFAULT_MAX_INFLIGHT, cost) as client:
        coros = [one(client, s, t, panel, an, sem) for s in SYNTHS for (t, panel, an) in items]
        done = 0
        for fut in asyncio.as_completed(coros):
            r = await fut
            results.append(r); done += 1
            print(f"[{done}/{len(coros)}] synth={r['synth'].split('/')[-1]:22s} {r['domain']:16s} score={r['score']} cost=${cost.cost_usd:.1f}")

    by = {}
    for r in results:
        if r["score"] is not None:
            by.setdefault(r["synth"], []).append(r["score"])
    print("\n" + "=" * 60)
    print("合成器排名（固定 panel，官方 grader，融合稿分）")
    print("=" * 60)
    for i, (m, vs) in enumerate(sorted(by.items(), key=lambda kv: -st.mean(kv[1])), 1):
        print(f"  {i}. {m:34s} {st.mean(vs):.1f}  (n={len(vs)})")
    json.dump({m: round(st.mean(vs), 2) for m, vs in by.items()}, open(OUT + "/summary.json", "w"), ensure_ascii=False, indent=2)
    print(f"\n成本: ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
