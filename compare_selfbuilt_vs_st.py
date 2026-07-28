"""验证：自建复现版 跑同一(模型×题) 的 DRACO 分 ≥ OpenRouter server-tool。

复用 output/trajectories/<domain>_<model>.json 里已生成的自建报告（不重跑 agent），
用同款裁判(gemini-3.1-pro ×3、无 max_tokens 上限)评分，对上 __st 里 server-tool 的同题分。
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import sys

from draco_eval import config, scoring
from draco_eval.dataset import load_all
from draco_eval.judge import judge_once
from draco_eval.openrouter import CostTracker, OpenRouterClient

JUDGE_RUNS = 3


def st_baseline(model_safe: str, domain: str):
    d = f"output/sweep/{model_safe}__st"
    fs = glob.glob(d + "/records_*.jsonl")
    if not fs:
        return None
    for line in open(fs[0]):
        r = json.loads(line)
        if r["domain"] == domain:
            return r.get("normalized_mean"), r.get("status")
    return None


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-5.5"
    domain = sys.argv[2] if len(sys.argv) > 2 else "Finance"
    model_short = model.split("/")[-1]
    model_safe = model.replace("/", "_")

    config.OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    traj_path = f"output/trajectories/{domain.replace('/','_')}_{model_short}.json"
    tj = json.load(open(traj_path))
    report = tj["report"]
    if not report:
        print(f"自建报告为空（{traj_path}），无法对比"); return

    task = next(t for t in load_all() if t.id == tj["task_id"])

    cost = CostTracker()
    async with OpenRouterClient(10, cost) as client:
        runs = await asyncio.gather(
            *[judge_once(client, task, report, config.DEFAULT_JUDGE_MODEL, i) for i in range(JUDGE_RUNS)]
        )
    ts = scoring.aggregate_task(task, list(runs))

    base = st_baseline(model_safe, domain)
    print("\n" + "=" * 60)
    print(f"验证 {model}  /  {domain}")
    print("=" * 60)
    print(f"自建复现版 normalized : {ts.normalized_mean}  (±{ts.normalized_std}, pass={ts.pass_rate_mean}, {len(report)}字)")
    if base:
        bn, bstatus = base
        print(f"server-tool 同题     : {bn}  (status={bstatus})")
        if bn is not None:
            verdict = "✅ 自建 ≥ server-tool" if ts.normalized_mean >= bn else "❌ 自建 < server-tool"
            print(f"判定                 : {verdict}  (差 {ts.normalized_mean - bn:+.1f})")
    else:
        print("server-tool 同题     : 无基准记录")
    print(f"评分成本             : ${cost.cost_usd:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
