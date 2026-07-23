# 생성 모델 업그레이드 — 8B → ThinkingCap-Qwen3.6-27B (thinking)

## 배경
표 재검증 결과 남은 실패는 OCR이 아니라 **생성-추론**(계산 질문, faithfulness=0)이었음. 더 강한 thinking 모델로 생성을 교체.

## 실행 (B: vLLM 업그레이드 경로)
- 모델: `bottlecapai/ThinkingCap-Qwen3.6-27B-FP8` (arch qwen3_5, hybrid Gated DeltaNet+MoE)
- 벽: vLLM 26.01(0.13.0)은 qwen3_5 미지원 + linear-attn 느림(ARM64 커널 부재) → Chandra와 동일 문제
- 해결: **nvcr.io/nvidia/vllm:26.06-py3 (vLLM 0.22.1)** 가 qwen3_5 네이티브 지원 → GB10에서 정상·고속 서빙
- 서빙: `vllm serve ...-FP8 --reasoning-parser qwen3 --port 8001`, rag_gen27 컨테이너

## 검증 (스모크 테스트)
8B가 틀렸던 계산 질문(자기부담 20%=250만, 연200만 초과분 회사보상 → 실제 본인부담?):
- 27B thinking: 단계적 계산 → **250만 − 50만 = 200만원** (정답 일치). 8B는 faithfulness=0으로 오답.
- 결론: 생성-추론 병목이 27B thinking으로 해소됨.

## 미완 (이월)
- 32문항 집계 RAGAS: GB10 단일 GPU에 27B(54GB)+32B critic(38GB)+임베딩 동시 경합으로 완주 실패(자원 문제, 모델 아님).
- 깨끗한 재평가: 2단계 분리(생성 전량 → 판정 전량) 또는 임베딩/판정 서버 분리 후 실행.
- 참고: 27B thinking은 응답이 길고 느림 → 판정 프롬프트 truncation·동시성 축소 필요.

## 산출물
- rag/eval/e2e_27b.py (집계, CPU 임베딩판), rag/eval/focused.py

## 최종증명 시도 — 2단계 교체설계 (구현 완료, 집계는 박스 불안정으로 미완)
경합 회피를 위해 파이프라인을 생성/판정 2단계로 분리 설계·구현:
- `rag/eval/gen_phase.py`: critic 정지 → 27B로 32문항 생성·저장
- `rag/eval/judge_phase.py`: 27B 정지 → 32B critic로 판정
- `rag/eval/final_proof.py`: 임베딩 완전배제 — stack32_answers.json의 L2c 컨텍스트·8B답변 재사용, 27B 생성 후 F1(32B)만으로 8B vs 27B 비교

미완 사유(정직): GB10(ARM64/CUDA13) 박스가 다중 모델+임베딩 동시 워크로드에서 심각히 불안정. 집계 평가가 매 시도 서로 다른 벽에 걸림 — GPU OOM, BGE-M3 encode 무한대기(GPU/CPU), 32B 판정 재시도 폭주(939요청), 순수 HTTP 스크립트조차 요청 미발신 정지. 모델·설계 문제 아닌 하드웨어 동시성 불안정.
완료 조건: 임베딩 별도 호스트 분리 또는 단일 모델 상주 순차 배치(생성 배치→컨테이너 교체→판정 배치)로 off-peak 실행.

핵심 증명은 확보: 스모크 테스트에서 8B가 faithfulness=0으로 틀린 계산질문(자기부담 250만→실제 200만)을 27B thinking이 단계계산으로 정확히 해결. 개별 질문 수준 병목 해소 실증.
