"""DRACO deep-research benchmark 复现管线（异步并发版）。

阶段：
  1) dataset  —— 从 HuggingFace 拉取 perplexity-ai/draco（100 题 + 加权 rubric）
  2) agent    —— 被测系统（默认 DeepSeek V4 Pro，经 OpenRouter，开启联网）生成研究报告
  3) judge    —— LLM-as-judge（默认 Gemini 3 Pro），每题对每条标准给 MET/UNMET，跑 N 次
  4) scoring  —— 套 DRACO 官方公式算 normalized_score / pass_rate，N 次取均值±std

设计目标：全程 asyncio 并发，task 级并发 = 子集大小（10 或 100），一次跑完。
"""

__version__ = "0.1.0"
