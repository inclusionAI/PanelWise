#!/usr/bin/env bash
# 多模型横扫：每个模型在 10 子集上跑一遍（plugin 联网，裁判 1 次，10 并发）。
# 单个模型失败不影响其他。结果各自落 output/sweep/<model>/。
set -u

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?need key}"
export AGENT_WEB_MODE=plugin

MODELS=(
  "deepseek/deepseek-v4-pro"
  "minimax/minimax-m3"
  "z-ai/glm-5.1"
  "google/gemini-3.1-pro-preview"
  "openai/gpt-5.5"
  "anthropic/claude-opus-4.8"
  "x-ai/grok-4.3"
)

mkdir -p output/sweep
for m in "${MODELS[@]}"; do
  safe="${m//\//_}"
  echo ">>>>> [$(date +%H:%M:%S)] START $m"
  .venv/bin/python -m draco_eval.run \
    --subset 10 --concurrency 10 --judge-runs 1 \
    --agent-model "$m" \
    --out-dir "output/sweep/$safe" \
    > "output/sweep/${safe}.log" 2>&1
  echo "<<<<< [$(date +%H:%M:%S)] DONE  $m (exit $?)"
done
echo "ALL SWEEP DONE"
