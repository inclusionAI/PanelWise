"""测裁判方法 delta：同一份报告，我们的裁判 vs 官方 rubric 裁判，各 ×3 取均值。

先用已存的 trajectory 报告(output/trajectories/*.json)，零额外 agent 成本。
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import statistics as st

from draco_eval import config, scoring
from draco_eval.dataset import load_all
from draco_eval.judge import judge_once
from draco_eval.official_judge import grade_official
from draco_eval.openrouter import CostTracker, OpenRouterClient

RUNS = 3


async def ours(client, task, report):
    vals = []
    for i in range(RUNS):
        r = await judge_once(client, task, report, config.DEFAULT_JUDGE_MODEL, i)
        ts = scoring.aggregate_task(task, [r])
        if ts.n_runs:
            vals.append(ts.normalized_mean)
    return st.mean(vals) if vals else None


async def official(client, task, report):
    vals = []
    for _ in range(RUNS):
        v = await grade_official(client, task, report, config.DEFAULT_JUDGE_MODEL)
        if v is not None:
            vals.append(v)
    return st.mean(vals) if vals else None


async def main():
    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    tasks_by_id = {t.id: t for t in load_all()}

    reports = []  # (domain, task, report)
    for f in sorted(glob.glob("output/trajectories/*.json")):
        d = json.load(open(f))
        t = tasks_by_id.get(d["task_id"])
        if t and d.get("report"):
            reports.append((d["domain"], t, d["report"]))

    print(f"用 {len(reports)} 份已存报告测裁判 delta（各裁判 ×{RUNS} 取均值）\n")
    cost = CostTracker()
    rows = []
    async with OpenRouterClient(10, cost) as client:
        for dom, t, rep in reports:
            o = await ours(client, t, rep)
            of = await official(client, t, rep)
            rows.append((dom, o, of))
            print(f"  {dom:12s} 我们={o}  官方={of}  delta(我们-官方)={None if (o is None or of is None) else round(o-of,1)}")

    print("\n" + "=" * 50)
    print(f"{'域':12s} {'我们':>8s} {'官方':>8s} {'delta':>8s}")
    for dom, o, of in rows:
        dd = None if (o is None or of is None) else round(o - of, 1)
        print(f"{dom:12s} {('%.1f'%o) if o else '-':>8s} {('%.1f'%of) if of else '-':>8s} {('%+.1f'%dd) if dd is not None else '-':>8s}")
    print(f"\n成本: ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
