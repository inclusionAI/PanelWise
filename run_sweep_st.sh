#!/usr/bin/env bash
# server-tool（agentic 多步联网）横扫，仅原生搜索四家。
# 输出放 output/sweep/<model>__st/，便于和 plugin 基线对比。
set -u
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?need key}"
export AGENT_WEB_MODE=server_tool

MODELS=(
  "google/gemini-3.1-pro-preview"
  "openai/gpt-5.5"
  "anthropic/claude-opus-4.8"
  "x-ai/grok-4.3"
)

mkdir -p output/sweep
for m in "${MODELS[@]}"; do
  safe="${m//\//_}__st"
  echo ">>>>> [$(date +%H:%M:%S)] START $m (server_tool)"
  .venv/bin/python -m draco_eval.run \
    --subset 10 --concurrency 10 --judge-runs 3 \
    --agent-model "$m" \
    --out-dir "output/sweep/$safe" \
    > "output/sweep/${safe}.log" 2>&1
  echo "<<<<< [$(date +%H:%M:%S)] DONE  $m (exit $?)"
done
echo "ALL ST SWEEP DONE"
