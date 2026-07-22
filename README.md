# Agentic OCR+RAG — 실손보험 약관 질의응답

DB손해보험 실손의료비보험 약관(60p)을 대상으로 한 검색증강생성 시스템.
`OCR → 후처리·검증 → 하이브리드 검색 → LangGraph 자기교정 그래프` 구조이며,
32문항 골든셋으로 검색·답변 품질을 계층별로 측정한다.

## 구성
| 계층 | 기술 |
|---|---|
| OCR | Unlimited-OCR (3B VLM) |
| 임베딩 | BGE-M3 (dense 1024 + sparse) |
| 벡터DB | Qdrant (named-vector 하이브리드 RRF, payload 인덱스) |
| 리랭커 | BGE-reranker-v2-m3 (cross-encoder) |
| 생성 | Qwen3-8B-AWQ (vLLM, OpenAI 호환) |
| 오케스트레이션 | LangGraph |
| 인프라 | DGX Spark GB10 (ARM64), OCR은 NVIDIA vLLM 컨테이너 |

## 인제스트 파이프라인
```
docs/*.pdf → OCR(60p) → 파싱(1285블록) → 검증(CER 84% / TEDS 0.55 / KIE)
           → 청킹(코사인급락 + small-to-big) → BGE-M3(dense+sparse) → Qdrant
```
- 파싱: `<|det|>type[bbox]<|/det|>content` 를 블록 단위로 분류(text/title/table 등).
- 검증: CER(문자정확도, OCR vs 텍스트레이어), TEDS(표 구조), KIE(담보종목/조항/관 + money/pct/period).
- 청킹: 문장 코사인급락 경계, 리스트 항목 비분리, 표 원형 보존. 검색은 작은 청크, 생성은 부모 페이지(small-to-big).

## LangGraph 그래프
```
START → route ──(잡담)── direct ── END
          │(검색)
       retrieve → grade(CRAG) ──(관련0)── transform ─→ retrieve  (재시도 ≤2)
          │(관련)
       generate(부모 페이지) → Self-RAG ──(비근거)── transform
          │(근거)
         END
```
2-way LLM 라우터, CRAG 재검색 루프, small-to-big 생성, Self-RAG 근거검증. 조건부 엣지 + MemorySaver 체크포인터.

## 검색 평가 (골든셋 32문항, page-level relevance)
| 청킹 | 방법 | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| 섹션 | 하이브리드 | 0.562 | 0.844 | 1.0 | 0.725 |
| 섹션 | +리랭커 | 0.625 | 0.875 | 0.938 | 0.760 |
| 코사인급락 | 하이브리드 | 0.656 | 0.906 | 0.969 | 0.785 |
| 코사인급락 | +리랭커 | 0.531 | 0.844 | 0.938 | 0.701 |
| 코사인급락 | +contextual(약식) | 0.469 | 0.812 | 0.875 | 0.641 |
| 코사인급락 | +contextual(페이지맥락) | 0.531 | 0.938 | 0.969 | 0.721 |

- 코사인급락 청킹이 섹션 청킹보다 상위검색 우세.
- 리랭커·contextual retrieval의 효과는 청킹 방식에 의존적(상호작용 존재).
- Contextual retrieval은 본 코퍼스에서 marginal — 청크 텍스트가 조항 헤더를 이미 포함하여 situating이 중복.

## 답변 평가 — Ablation ladder (RAGAS 지표, 골든셋 32문항)
| 단계 | context_recall | faithfulness | answer_correctness |
|---|---|---|---|
| L0 dense | 0.694 | 0.847 | 0.636 |
| L1 +하이브리드 | 0.743 | 0.859 | 0.716 |
| L2 +small-to-big | 0.902 | 0.838 | 0.758 |
| L3 +리랭커 | 0.943 | 0.847 | 0.737 |
| L4 +에이전틱 | 0.927 | 0.906 | 0.701 |

- small-to-big이 answer_correctness 기여 최대(0.636→0.758).
- 리랭커는 context_recall을 최고(0.943)로 올리나 answer_correctness로 전환되지 않음(0.758→0.737).
- 에이전틱(CRAG/Self-RAG)은 faithfulness 최고(0.906), answer_correctness는 감소(신중성·거부 증가).
- Failure localization(L4): 검색실패 1, 생성환각 1, 표·수치 정밀추출 9(context_recall=1·faithfulness=1인데 answer_correctness 낮음). 잔여 실패는 검색·환각이 아닌 표 수치 추출 문제이며 리랭커·에이전틱으로 해소되지 않음.

## 청킹 결함 사례 — 면책 항목 유실
비만(E66) 면책 규정 질의에서 드러난 4단계 결함과 수정:

| 단계 | 증상 | 수정 |
|---|---|---|
| 청킹 필터 | 30자 미만 폐기로 `⑤비만(E66)`(9자) 삭제 | 최소 길이 10→2, 잔여 조각 병합 |
| 헤더 분리 | 리스트만 검색되어 면책을 보상으로 오독 | — |
| 검색·문맥 상충 | 헤더 병합 시 청크 희석으로 미검색 | small-to-big 도입 |
| 부모 품질 | raw OCR 절단·bbox 노이즈 | 파싱블록 기반 정제 페이지 |

## 실행
```bash
# 사전: Qdrant(:6333), Qwen3-8B(vLLM :8001)
.venv-rag/bin/python rag/ingest/chunk_fix.py     # 청킹·인덱싱 (silson_v2_sem)
.venv-rag/bin/python rag/eval/eval.py            # 검색 hit@k/MRR + 리랭커 델타
.venv-rag/bin/python rag/eval/qdrant_tune.py     # ef_search 튜닝 + payload 필터검색
.venv-rag/bin/python rag/graph/agentic_rag.py    # LangGraph 에이전틱 RAG
.venv-rag/bin/python rag/eval/build_ladder.py    # RAGAS ablation 데이터셋 L0~L4
.venv-rag/bin/python rag/eval/custom_ragas.py    # 답변품질 3지표
```

## 측정 환경 및 한계
- 인덱싱: 벡터 193개 규모에서 Qdrant가 exact search로 폴백, ef_search 튜닝 효과 없음(recall@10 전 구간 1.0). payload 인덱스(dambo/btype/pages)로 담보종목·표·페이지 스코프 필터검색 제공.
- RAGAS 라이브러리는 로컬 vLLM에 대해 순차 호출(동시 1)로 비실용적(~60분). 지표 정의(faithfulness, context_recall, answer_correctness=0.75·F1+0.25·의미유사도)를 스레드 병렬로 직접 구현.
- 심판 모델이 로컬 Qwen3-8B로 answer_correctness에 노이즈. 표·수치 정밀추출 미해결.
