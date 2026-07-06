# vLLM 서빙 (H100)

Qwen3-30B-A3B를 vLLM OpenAI 호환 서버로 서빙한다. 에이전트는 백엔드 독립이라
`LLM_BASE_URL`만 이 서버로 바꾸면 된다(Ollama→vLLM 교체 = 설정 2줄).

## ★전제 / 주의
- **드라이버 550.144.03 = CUDA 12.4 상한.** cu126/cu128/cu130 휠은 못 씀(예전 vLLM
  init 실패 원인). → `vllm==0.8.5.post1`(torch 2.6.0+**cu124**)로 핀. `setup_vllm.sh`가 처리.
- **이 H100은 공유 서버**(여러 사용자). 실행 전 `nvidia-smi`로 GPU 여유 확인하고,
  동료 작업을 밀어내지 않도록 주의. (2장을 오래 점유하는 서버라 특히.)
- `Qwen/Qwen3-30B-A3B` 풀 정밀 모델은 timmyoo HF 캐시에 이미 있음 → TP=2로 바로 서빙.
  단일 GPU로 돌리려면(1장 양보) AWQ 양자화판 사용(`serve_vllm.sh` 하단 주석).

## 실행
```bash
# H100에서:
bash serve/setup_vllm.sh      # 최초 1회 (venv + vllm 0.8.5 cu124)
bash serve/serve_vllm.sh      # 서빙 (TP=2, :8000)
```

## 로컬(WSL)에서 접속
vLLM이 H100 :8000에 뜨면, run.py가 있는 곳에서 접근하려면 SSH 터널:
```bash
ssh -i ~/silicon-cube-key -p 50023 -N -L 8000:127.0.0.1:8000 root@182.208.83.156
```
그다음 `eval/configs/baseline.yaml`의 `LLM_BASE_URL=http://127.0.0.1:8000/v1` 그대로 사용.

## 실측 파이프라인 (서버 뜬 뒤)
```bash
python -m eval.run    --domain qc_report --config eval/configs/baseline.yaml
python -m eval.report --domain qc_report --pred eval/predictions_qc_report.json --model Qwen3-30B-A3B
python -m eval.gate   --domain qc_report --pred eval/predictions_qc_report.json --update  # 최초 baseline
```
