#!/usr/bin/env bash
# Qwen3-30B-A3B를 vLLM OpenAI 호환 서버로 서빙 (H100에서 실행).
# ★공유 서버 주의: 실행 전 GPU 여유 확인.  nvidia-smi
set -euo pipefail

VENV="${VENV:-$HOME/vllm-venv}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"   # timmyoo HF 캐시에 이미 있음(풀 정밀)
PORT="${PORT:-8000}"
TP="${TP:-2}"                          # 2×H100 텐서병렬(풀모델 bf16 ~60GB)

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "GPU 현황:"; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "→ $MODEL 서빙 (TP=$TP, port=$PORT)"

exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --served-model-name "$MODEL" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port "$PORT"

# ── 단일 GPU로 돌리고 싶으면(다른 GPU를 동료에게 양보) AWQ 양자화판 사용:
#   pip install autoawq  # 또는 이미 양자화된 repo 다운로드
#   MODEL=<AWQ-repo> TP=1  \
#   vllm serve "$MODEL" --quantization awq_marlin --tensor-parallel-size 1 ...
