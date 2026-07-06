#!/usr/bin/env bash
# H100에 vLLM 서빙 환경 구성 (이 박스에서 실행).
# ★드라이버 550.144.03 = CUDA 12.4 상한. cu126/cu128/cu130 휠은 못 씀(예전 실패 원인).
#   → vLLM 0.8.5.post1 (torch 2.6.0+cu124 기본) 로 핀. Qwen3 지원 + cu124 정합.
set -euo pipefail

VENV="${VENV:-$HOME/vllm-venv}"

# 드라이버/CUDA 상한 확인
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "driver=$DRV  (CUDA 12.4 상한: cu124 휠만 사용)"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel

# vLLM 0.8.5.post1 = torch 2.6.0+cu124 기본 → 드라이버 550.144와 정합
pip install "vllm==0.8.5.post1"

# 혹시 torch가 cu126/cu128로 딸려오면 cu124로 강제 복원(예전에 통한 패턴)
python - <<'PY'
import torch
v = torch.__version__
print("torch:", v)
assert "cu124" in v, f"torch가 cu124가 아님({v}) → 아래 명령으로 복원 필요"
PY

echo "✅ setup 완료: $VENV"
echo "다음: bash serve/serve_vllm.sh"
