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

## 검색 스택
단일 모델·단일 컬렉션으로 dense+sparse 하이브리드를 구성하고, 각 요소의 기여를 골든셋으로 측정했다.

**임베딩 — BGE-M3 (단일 모델, 다중 표현)**
- 한 모델이 dense(의미, 1024차원)와 sparse(어휘 가중치)를 동시 생성 — 별도 SPLADE/BM25 파이프라인 없이 하이브리드 구성.
- sparse 표현이 조항 번호·담보종목명·금액 등 정확 일치 토큰을 포착하여 dense 단독이 놓치는 법률·수치 텍스트를 보완.
- 다국어·롱컨텍스트 임베딩으로 한국어 약관 용어 처리.

**하이브리드 검색 — Qdrant named vectors + 서버측 RRF**
- 한 컬렉션의 각 포인트가 dense·sparse 두 벡터를 함께 보유(named vectors) — 인덱스 이중화 없음.
- 질의 시 dense top-30 + sparse top-30을 서버측 Reciprocal Rank Fusion으로 융합(FusionQuery) — 클라이언트 병합·점수 정규화 불필요.
- payload 인덱스(dambo/btype/pages)로 담보종목·표·페이지 구간 스코프 필터검색(filterable HNSW). HNSW의 ef_search·exact 폴백 특성까지 검증.

**리랭킹 — BGE-reranker-v2-m3 (cross-encoder)**
- 질의-문서 결합 어텐션으로 bi-encoder 검색 결과를 top-20→5 재정렬, 상위 정밀도 보강.
- 효과를 청킹별로 측정: 섹션 청킹엔 MRR +0.035, 코사인급락 청킹엔 손해 — "리랭커는 항상 이득"이 아님을 실측(청킹×리랭커 상호작용). 검색 개선이 답변정확도로 전환되지 않는 지점도 RAGAS로 포착.
- FlagReranker의 slow-tokenizer 비호환(transformers 5.x)을 AutoModelForSequenceClassification 직접 로드로 우회.

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
| L2b +구조화 표(small-to-big) | 0.812 | 0.925 | 0.799 |
| L2c +markdown 표 | 0.938 | 0.831 | 0.780 |

- small-to-big이 answer_correctness 기여 최대(0.636→0.758).
- 리랭커는 context_recall을 최고(0.943)로 올림. answer_correctness 전환 여부는 심판 의존적 — 8B 심판에선 하락(0.758→0.737)이나 32B 심판 재채점에선 상승(0.670→0.699). 아래 심판 검증 참조.
- 에이전틱(CRAG/Self-RAG)은 faithfulness 최고(0.906), answer_correctness는 감소(신중성·거부 증가).
- 표·수치 정밀추출 개선: 부모 문맥에서 표 HTML 구조를 보존하면 answer_correctness 0.758→0.799(최고), faithfulness 0.838→0.925 상승. context_recall은 HTML 장황함으로 0.902→0.812 감소(구조·예산 트레이드오프). 표를 markdown으로 압축(L2c)하면 context_recall 0.938(최고)로 회복하며 표 정확도 0.853 유지 — 균형 최적.
- Failure localization(L4): 검색실패 1, 생성환각 1, 표·수치 정밀추출 9(context_recall=1·faithfulness=1인데 answer_correctness 낮음). 잔여 실패는 검색·환각이 아닌 표 수치 추출 문제이며 리랭커·에이전틱으로 해소되지 않음.

## 심판 모델 검증 — 8B vs 32B
answer_correctness의 로컬 8B 심판 노이즈를 검증하기 위해, 동일 생성 답변을 Qwen3-32B-AWQ 심판(별도 vLLM 인스턴스)으로 재채점했다.

| 단계 | ac(8B) | ac(32B) | cr(8B) | cr(32B) |
|---|---|---|---|---|
| L0 dense | 0.636 | 0.609 | 0.694 | 0.861 |
| L1 +하이브리드 | 0.716 | 0.661 | 0.743 | 1.00 |
| L2 +small-to-big | 0.758 | 0.670 | 0.902 | 0.972 |
| L3 +리랭커 | 0.737 | 0.699 | 0.943 | 0.953 |
| L4 +에이전틱 | 0.701 | 0.687 | 0.927 | 0.972 |

- 심판이 바뀌면 결론이 바뀐다: 8B 심판은 answer_correctness가 L2에서 정점 후 리랭커·에이전틱에서 하락한다고 봤으나, 32B 심판에선 리랭커(L3)에서 상승(0.670→0.699). "리랭커가 답변으로 전환되지 않는다"는 8B 결론은 로버스트하지 않음.
- 로버스트한 신호: 두 심판 모두 전체 스택(L3/L4)이 naive(L0)보다 context_recall·answer_correctness 모두 우세. 개별 컴포넌트의 answer_correctness 기여 순위는 심판에 민감.
- 심판 편향: 8B는 answer_correctness를 후하게(L2 0.758 vs 32B 0.670), 32B는 context_recall을 후하게 매김. 소규모 로컬 심판의 절대값·순위는 신뢰구간이 넓다.

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
- 심판 검증: 8B 심판의 결론 일부(리랭커·에이전틱의 answer_correctness 영향)가 32B 심판 재채점에서 뒤집힘 — 컴포넌트 단위 답변정확도 결론은 심판 의존적. 표·수치 정밀추출 미해결.
