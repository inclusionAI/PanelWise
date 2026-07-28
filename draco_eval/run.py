"""CLI 入口。

示例：
  # 10 子集，10 并发，一次跑完（DeepSeek V4 Pro + Gemini 裁判 5 次）
  python -m draco_eval.run --subset 10 --concurrency 10

  # 100 全量，100 task 并发（注意 max-inflight 限制 HTTP 在飞数防限速）
  python -m draco_eval.run --subset 100 --concurrency 100 --max-inflight 50

  # 只生成不打分（省裁判钱），裁判跑 1 次（省 80% 裁判成本）
  python -m draco_eval.run --subset 10 --no-judge
  python -m draco_eval.run --subset 10 --judge-runs 1

  # 干跑：只看会评哪些题，不花钱
  python -m draco_eval.run --subset 10 --dry-run
"""
from __future__ import annotations

import argparse
import asyncio

from . import config
from .config import RunConfig
from .pipeline import run as run_pipeline


def parse_args(argv: list[str] | None = None) -> RunConfig:
    p = argparse.ArgumentParser(prog="draco_eval", description="DRACO 复现（异步并发）")
    p.add_argument("--subset", type=int, default=10, help="评测题数：10 / 100 / 任意 N（默认 10）")
    p.add_argument("--concurrency", type=int, default=None,
                   help="task 级并发，默认等于 subset（即一次跑完）")
    p.add_argument("--max-inflight", type=int, default=config.DEFAULT_MAX_INFLIGHT,
                   help=f"HTTP 在飞请求上限，防限速（默认 {config.DEFAULT_MAX_INFLIGHT}）")
    p.add_argument("--agent-model", default=config.DEFAULT_AGENT_MODEL,
                   help=f"被测模型 slug（默认 {config.DEFAULT_AGENT_MODEL}）")
    p.add_argument("--judge-model", default=config.DEFAULT_JUDGE_MODEL,
                   help=f"裁判模型 slug（默认 {config.DEFAULT_JUDGE_MODEL}）")
    p.add_argument("--judge-runs", type=int, default=config.DEFAULT_JUDGE_RUNS,
                   help=f"每题独立打分次数（默认 {config.DEFAULT_JUDGE_RUNS}）")
    p.add_argument("--out-dir", default="output", help="结果输出目录（默认 output）")
    p.add_argument("--no-judge", action="store_true", help="只跑生成，不打分")
    p.add_argument("--dry-run", action="store_true", help="只加载数据、打印选择，不调 API")
    a = p.parse_args(argv)

    return RunConfig(
        subset=a.subset,
        concurrency=a.concurrency if a.concurrency is not None else a.subset,
        max_inflight=a.max_inflight,
        agent_model=a.agent_model,
        judge_model=a.judge_model,
        judge_runs=a.judge_runs,
        out_dir=a.out_dir,
        no_judge=a.no_judge,
        dry_run=a.dry_run,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    asyncio.run(run_pipeline(cfg))


if __name__ == "__main__":
    main()
