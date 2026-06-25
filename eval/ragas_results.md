# RAGAS 평가 — LangGraph 문서 추출 에이전트

- judge: `qwen2.5:7b-instruct` (Ollama) · 추출 에이전트: `qwen3:8b`
- 골든셋: 3건 (정상 영수증/세금계산서 + 결함 문서)
- 지표: **Faithfulness**(추출이 원본 문서에 근거하는가), **FactualCorrectness**(골든과 사실 일치)

## 집계 (평균)

- **faithfulness**: 0.833
- **factual_correctness(mode=f1)**: 0.850

## 문서별

| id | category | faithfulness | factual_correctness(mode=f1) |
|---|---|---|---|
| doc-01 | 정상-영수증 | 1.000 | 0.800 |
| doc-02 | 정상-세금계산서 | 1.000 | 0.860 |
| doc-03 | 결함-공급자/날짜 없음 | 0.500 | 0.890 |

## 한계 (정직성)

- N=3 합성셋, 작성자 1인 라벨 → 정밀 벤치가 아니라 방향성 sanity check.
- judge LLM(qwen2.5:7b) 기반이라 절대 점수는 judge 능력에 의존.
- 추출 에이전트는 RAG가 아니라 단일 문서 추출이므로, context_precision/recall(랭킹 지표)은 적용하지 않고 근거성(faithfulness)·정답일치(factual correctness)만 측정한다.
