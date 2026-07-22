# OCR 업그레이드 조사 보고서 — 표·수치 추출 병목 해소

## TL;DR
- **문제**: RAG의 마지막 병목은 표·수치 정밀추출(32B 심판 기준 표 질문 answer_correctness 0.604; context_recall=1·faithfulness=1인데 숫자가 틀림).
- **표현 재조합으론 못 넘음**: HTML/markdown/행-records/수치 자기검증 노드 모두 실패(records는 0.604→0.439로 악화). 병목은 표현 형식이 아니라 **표 구조 파싱 품질**(현재 OCR의 TEDS≈0.55).
- **벤치마크는 참고만**: OmniDocBench/OCRBench v2/CC-OCR에서 표 TEDS 1위는 소형 특화 파서(GLM-OCR 0.9B, PaddleOCR-VL). 단 전부 영어·중국어 중심.
- **우리 한국어 표 실측 bake-off가 결론**: **Chandra v2(Qwen3-VL 8B)가 최고**. 영/중 벤치 1위 GLM-OCR은 한국어 글자 오류로 오히려 최악 → 벤치 순위가 한국어로 전이되지 않음.
- **결정**: Chandra v2로 전면 재-OCR. **적용은 다음 세션**(본 보고서는 결정+계획까지).

## 1. 배경 — 무엇이 병목인가
RAG 계층별 실측(README 참조) 결과, 검색·에이전틱·생성은 개선되었으나 표 질문만 정체.
- 32B 심판: 표 질문 ac=0.604 (8B 심판 0.853은 과대평가).
- 실패 유형: context_recall=1·faithfulness=1인데 answer_correctness 낮음 = 검색·환각 아님. **표의 특정 셀 숫자를 잘못 읽음**.

## 2. 조사 1 — 표현 재조합 (실패)
같은 OCR 출력으로 표를 어떻게 "보여주느냐"만 바꿔 시도:
| 방식 | 표 ac(32B) |
|---|---|
| HTML grid (L2b) | 0.604 |
| markdown (L2c) | 0.604 |
| 수치 자기검증 노드 (L5) | 0.604 (무효) |
| 행 단위 records (L6) | 0.439 (악화) |

- records 악화 원인: colspan/rowspan 병합셀을 위치기반 파서가 오정렬 → 확신에 찬 오답.
- 수치 자기검증 무효: 같은 모델의 체계적 오독은 자기검증으로 안 잡힘.
- **결론: 병목은 표현이 아니라 표 구조 파싱(TEDS≈0.55)이다.** → OCR 자체를 바꿔야 함.

## 3. 조사 2 — 벤치마크 (참고, 한계 명확)
표 TEDS(OmniDocBench):
| 모델 | 크기 | Table TEDS |
|---|---|---|
| GLM-OCR | 0.9B | 93.96 |
| MinerU 2.5 | ~1.2B | 구조 95.39 |
| PaddleOCR-VL | 0.9B | ~91 |
| dots.ocr / DeepSeek-OCR2 | 3B | 82 / 77.5 |
| Unlimited-OCR(현재) | 3.3B | ≈55(우리 실측) |

- 표에서는 **소형 특화 파서(0.9B)가 대형 범용 VLM을 능가**(Qwen3-VL-235B 89.15 < GLM-OCR 0.9B 94.62). "8B가 낫다"는 표엔 성립 안 함.
- **치명적 한계**: 세 벤치 모두 영어·중국어 중심. 한국어 보험 표에 전이 보장 없음. → 실측 필요.

## 4. 조사 3 — 우리 한국어 표 실측 bake-off (결정)
후보를 실제 표(p21/p23, 통원 공제금액, 병합셀·계층구조)에 직접 실행.
정답 셀: 표준형 외래 **의원 1만원 / 병원 1.5만원 / 상급종합병원 2만원 / 약제 8천원**.

| 모델 | 표 구조 | 숫자 4/4 | 한국어 글자 | 판정 |
|---|---|---|---|---|
| **Chandra v2** (8B, Qwen3-VL) | rowspan/colspan 정확 | ✅ | 깨끗 | **최고** |
| Unlimited-OCR (3.3B, 현재) | 양호 | ✅ | 깨끗 | 양호 |
| GLM-OCR (0.9B) | 정확 | ✅ | 오류 심함(의료→의로, 병원→복원) | 한국어 열세 |
| PaddleOCR-VL (0.9B) | — | — | — | 런타임 실패 |

- **GLM-OCR(영/중 벤치 1위)이 한국어 글자에서 최악** — 벤치 전이 실패 실증.
- **Chandra v2가 표 구조·숫자·한국어 모두 최고** + 레이아웃 라벨(bbox/element type) 보너스.
- 백본 언어 커버리지(Qwen3-VL의 한국어)가 벤치 순위보다 결정적.

## 5. 결정
**Chandra v2 (`datalab-to/chandra-ocr-2`, ~17.5GB, Qwen3-VL 8B)** 채택.

## 6. 적용 계획 (다음 세션)
1. **전면 재-OCR**: 60p 전체를 Chandra v2로 재판독 → 새 마크다운/HTML(레이아웃 라벨 포함).
2. **다운스트림 재실행**: 파싱 → 검증(TEDS 재측정, ≈0.55 → 목표 0.9+) → 청킹(small-to-big) → 재인덱싱(신규 Qdrant 컬렉션 silson_v3).
3. **재평가**: 검색 hit@k/MRR + RAGAS(32B 심판) + 표 질문 ac. 목표: 표 ac 0.604 → 유의미 상승.
4. **엔드투엔드 델타**: Unlimited-OCR 대비 표 질문 answer_correctness 개선폭 측정.

## 7. 실행 런타임 노트 (재현용)
- GB10(ARM64)에서 신규 아키텍처(qwen3_5, glm_ocr)는 vLLM 26.01 컨테이너 transformers가 미인식 → **NVIDIA 컨테이너 안에서 `pip install transformers==5.14.1 accelerate timm einops tiktoken qwen-vl-utils` 후 transformers-direct 실행**.
- **cuDNN conv2d 에러**(GB10 bf16 vision encoder) → `torch.backends.cudnn.enabled=False`.
- **이미지 로딩**: url 전달 시 torchvision `read_file` 실패 → **PIL 이미지 객체 직접 전달**.
- Triton JIT(.venv-rag)는 dev 헤더 부재로 컴파일 실패 → 반드시 컨테이너에서 실행.
- 추론: `AutoProcessor` + `AutoModelForImageTextToText`, `apply_chat_template(..., add_generation_prompt=True, return_dict=True)`, `generate(max_new_tokens=4096, do_sample=False)`.
- 스크립트: `rag/eval/bake_ocr.py`(GLM/멀티), `rag/eval/bake_chandra.py`(Chandra, cudnn 우회).

## 8. 리스크 / 한계
- Chandra v2 17.5GB → 전면 60p 재-OCR은 수십 분 소요. VRAM 관리(생성/심판 컨테이너 정지 후 실행).
- PaddleOCR-VL은 transformers 5.14에서 `PaddleOCRVLConfig.text_config` 부재로 미실행(런타임 정비 시 재시도 가능).
- bake-off는 표 2장(p21/p23) 기준 — 전면 재-OCR 후 TEDS/표 ac로 정량 확정 필요.
- 컨테이너 상태: 재-OCR 위해 `rag_llm`/`rag_critic` 정지했었음(현재 rag_llm 복구). 적용 시 재조정.

## 9. 산출물
- 벤치 분석: `docs/ocr-benchmark.md`
- bake-off 출력: `data/ocr_bake/{glmocr,chandra2}_p{21,23}.png.md`
- 스크립트: `rag/eval/bake_ocr.py`, `rag/eval/bake_chandra.py`
