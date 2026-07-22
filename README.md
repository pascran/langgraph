# Agentic Self-Correcting OCR+RAG — 실손보험 약관

DB손해보험 실손의료비보험 약관(60p)에 대한 **에이전틱 자기교정 RAG**.
OCR → 후처리(검증) → 하이브리드 검색 → LangGraph 자기교정 그래프.
**각 기법을 32문항 골든셋으로 실측·A/B** 한 것이 핵심 — "논문 복붙"이 아니라 무엇이 실제로 검색/답변을 올리는지 데이터로 증명.

## 스택
| 계층 | 기술 |
|---|---|
| OCR | Unlimited-OCR (3B VLM) |
| 임베딩 | BGE-M3 (dense 1024 + sparse, 한 모델) |
| 벡터DB | Qdrant (named vectors 하이브리드 RRF, payload 인덱스) |
| 리랭커 | BGE-reranker-v2-m3 (cross-encoder) |
| 생성 LLM | Qwen3-8B-AWQ (vLLM :8001) |
| 오케스트레이션 | **LangGraph** (route→retrieve→CRAG→generate→Self-RAG) |

## 인제스트 파이프라인
`docs/*.pdf → OCR(60p) → 파싱(1285블록) → CER 84% / TEDS 0.55 / KIE → 청킹(small-to-big) → BGE-M3 → Qdrant`

## LangGraph 에이전틱 그래프
```
        START
          │
        route ──(잡담)── direct ── END      # 2-way LLM 라우터
          │(검색)
       retrieve (작은 청크)
          │
        grade ──(관련0)── transform ─┐      # CRAG: 부실검색 감지→쿼리재작성
          │(관련)                    └─→ retrieve (재시도 최대2)
       generate (부모 페이지 전체)           # small-to-big: 검색은 작게, 생성은 크게
          │
       Self-RAG ──(비근거)── transform      # 근거검증→환각시 재시도
          │(근거)
         END
```

## 검색 A/B 연구 (골든셋 32문항, page-level relevance)
| 청킹 | 방법 | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| 섹션 | 하이브리드 | 0.562 | 0.844 | **1.0** | 0.725 |
| 섹션 | +리랭커 | 0.625 | 0.875 | 0.938 | 0.760 |
| 코사인급락 | 하이브리드 | 0.656 | 0.906 | 0.969 | **0.785** |
| 코사인급락 | +리랭커 | 0.531 | 0.844 | 0.938 | 0.701 |
| 코사인급락 | +contextual(약식) | 0.469 | 0.812 | 0.875 | 0.641 |
| 코사인급락 | +contextual(페이지맥락) | 0.531 | **0.938** | 0.969 | 0.721 |

## 실측 발견 (정직)
1. **코사인급락 청킹 > 섹션 청킹** (상위검색).
2. **리랭커·contextual은 "무조건 좋다"가 거짓** — 청킹에 따라 도움/해로움이 갈리는 상호작용.
3. **Contextual Retrieval은 이 코퍼스선 marginal** — 청크 텍스트가 이미 조항 헤더를 품어 situating이 중복.
4. **비만(E66) 4층 failure-localization**: 청킹 필터가 짧은 면책항목 삭제 → 헤더 분리로 LLM 오독 → 검색↔문맥 상충 → **small-to-big으로 해결**(작게 검색+깨끗한 부모페이지로 생성).
5. **ef_search 튜닝은 이 규모(193청크)선 무의미** — recall@10 전 구간 1.0(Qdrant exact 폴백). **payload 인덱스**(dambo/btype/pages)로 스코프 필터검색.
6. **OCR이 최대 레버** — 텍스트레이어 garbage → hit@3 0.9+.

## 실행
```bash
# 사전: Qdrant(:6333), Qwen3-8B(vLLM :8001) 기동
.venv-rag/bin/python rag/ingest/chunk_fix.py       # 청킹+인덱싱 (silson_v2_sem)
.venv-rag/bin/python rag/eval/eval.py              # 검색 hit@k/MRR + 리랭커 델타
.venv-rag/bin/python rag/graph/agentic_rag.py      # LangGraph 에이전틱 RAG
```

## 이월
- RAGAS(faithfulness/answer_correctness) 답변품질 정량화 → 에이전틱 그래프 vs 직선 RAG 델타
- 라우터 tool 경로, transform 다양화 고도화
