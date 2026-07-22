# OCR 모델 선택 — 벤치마크 근거

## 목적
현재 OCR(`baidu/Unlimited-OCR`, ~3.3B)의 **표 구조 추출**이 병목(우리 코퍼스 실측 TEDS ≈ 0.55, 표 질문 answer_correctness 0.604 상한). 교체 후보를 공개 벤치마크(OmniDocBench·OCRBench v2·CC-OCR)의 발표 결과로 비교한다. (세 벤치를 다수 모델에 로컬 풀런하는 것은 비현실적 — 리더보드가 존재 이유.)

## 표 성능 (OmniDocBench, 우리 병목 지표)
| 모델 | 크기 | Table TEDS↑ | Table TEDS-S↑ | 비고 |
|---|---|---|---|---|
| GLM-OCR | 0.9B | **93.96** | **96.39** | 표 최고, 파라미터 효율 |
| MinerU 2.5 | ~1.2B(파이프라인) | ~90.0 | 95.39 | 구조 TEDS 최고, 표 특화 |
| PaddleOCR-VL(-1.5) | 0.9B | 90.5~91.1 | 94.20 | 109개 언어 지원 |
| dots.ocr | 3B | 81.94 | 84.42 | |
| DeepSeek-OCR2 | 3B | 77.5 | — | |
| **Unlimited-OCR (현재)** | 3.3B | **≈55 (우리 실측)** | — | 병목 원인 |
| Qianfan-OCR | ~5B(Qwen3 백본) | 미공개 | — | CC-OCR overall 79.3, multilingual 강 |

## 핵심 결론
1. **현재 Unlimited-OCR(≈0.55)와 상위 모델(0.90~0.94) 격차가 큼** — 표 병목을 실제로 넘을 여지 충분.
2. **표에서는 소형 특화 파서(0.9B GLM-OCR·PaddleOCR-VL)가 대형 범용 VLM을 이긴다.** Qwen3-VL-235B는 OmniDocBench overall 89.15로 0.9B GLM-OCR(94.62)보다 낮음. → **"더 큰(8B) 모델이 표에 낫다"는 가정은 성립하지 않음.** 표 정확도는 크기가 아니라 특화 학습이 좌우.
3. 범용 VLM(Qwen2.5-VL-72B, Ovis2.5)은 OCRBench v2·CC-OCR의 이해·multilingual 트랙에서 강하나, **순수 표 구조 TEDS는 특화 파서(MinerU·PaddleOCR-VL·GLM-OCR)가 우위** (Dr.DocBench도 동일 결론).

## 한국어 관련 주의
- 세 벤치는 영어·중국어 중심. **한국어 보험 약관 + 병합셀 표에 점수가 그대로 전이된다는 보장 없음.**
- 언어 커버리지: PaddleOCR-VL = 109개 언어(한국어 포함). Qianfan-OCR/Qwen-VL = Qwen 백본으로 한국어 양호. GLM-OCR·MinerU = 한국어 검증 필요.

## 권장
- **1순위: PaddleOCR-VL** — 표 TEDS ~0.90 + 109개 언어(한국어) + 0.9B(GB10에서 극히 가벼움). 표 이득·한국어·인프라 균형 최적.
- 대안: **GLM-OCR**(표 TEDS 최고 0.94, 한국어 확인 시) / **MinerU 2.5**(구조 TEDS 최고, 단 파이프라인).
- Qianfan-OCR(~5B): multilingual 강하나 순수 표 TEDS 미검증·더 무거움 → PaddleOCR-VL의 한국어 표가 부실할 때만.

## 다음 단계(벤치→우리 코퍼스 검증)
영어·중국어 리더보드를 맹신하지 말고, **후보(PaddleOCR-VL)를 우리 표 페이지 2~3장에 로컬 실행 → Unlimited-OCR 대비 TEDS를 실측**해 전이 여부를 확인한 뒤 전면 교체.

## 한계
벤치마크 포화·벤더 자기보고(PaddleOCR-VL-1.6 96.33 등은 미검증 prior)·third-party 재현 부재 등 주의. 단일 aggregate 점수는 프로덕션 실패 케이스를 가림.

## 출처
- OmniDocBench (CVPR2025): https://github.com/opendatalab/OmniDocBench
- OCRBench v2: https://99franklin.github.io/ocrbench_v2/
- CC-OCR (ICCV2025): https://zhibogogo.github.io/ccocr.github.io/
- OCR SOTA 리더보드(2026): https://instavar.com/blog/ai-production-stack/OCR_SOTA_Feb_2026_Open_Document_AI_Leaderboard
