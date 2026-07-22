# Agentic Self-Correcting OCR+RAG — 실손보험 약관

DB손해보험 실손의료비보험 약관(60p)에 대한 **에이전틱 자기교정 RAG**.
`OCR → 후처리(검증) → 하이브리드 검색 → LangGraph 자기교정 그래프`.

핵심은 **각 기법을 32문항 골든셋으로 실측·A/B** 한 것 — "논문 복붙"이 아니라
*무엇이 실제로 검색/답변 품질을 올리는지*, 그리고 *안 올리는지*를 데이터로 정직하게 증명.

## 스택
| 계층 | 기술 |
|---|---|
| OCR | Unlimited-OCR (3B VLM), `<det>` 블록 + HTML 표 |
| 임베딩 | BGE-M3 (dense 1024 + sparse, 단일 모델) |
| 벡터DB | Qdrant (named vectors 하이브리드 RRF, payload 인덱스) |
| 리랭커 | BGE-reranker-v2-m3 (cross-encoder) |
| 생성 LLM | Qwen3-8B-AWQ (vLLM, OpenAI 호환 :8001) |
| 오케스트레이션 | **LangGraph** (route→retrieve→CRAG→generate→Self-RAG) |
| 인프라 | DGX Spark GB10 (ARM64), OCR은 NVIDIA vLLM 컨테이너 |

## 인제스트 파이프라인
```
docs/*.pdf → OCR(60p) → 파싱(1285블록) → 검증(CER 84% / TEDS 0.55 / KIE)
           → 청킹(코사인급락 + small-to-big) → BGE-M3(dense+sparse) → Qdrant
```
- **파싱**: `<|det|>type[bbox]<|/det|>content` → 1285블록(text/title/table/…)
- **검증**: CER(문자정확도, OCR vs 텍스트레이어) · TEDS(표 구조) · KIE(담보종목/조항/관 + money/pct/period)
- **청킹**: 문장 코사인급락 경계 + 리스트 항목 분리금지 + 표 통째보존

## LangGraph 에이전틱 그래프
```
        START
          │
        route ──(잡담)── direct ── END        # 2-way LLM 라우터
          │(검색)
       retrieve  (작은 청크로 검색)
          │
        grade ──(관련0)── transform ─┐        # CRAG: 부실검색 감지→쿼리재작성
          │(관련)                    └→ retrieve   (재시도 ≤2)
       generate  (부모 페이지 전체)             # small-to-big: 검색은 작게, 생성은 크게
          │
       Self-RAG ──(비근거)── transform         # 근거검증→환각시 재시도
          │(근거)
         END
```
MemorySaver 체크포인터. LLM 노드는 Qwen3-8B(:8001).

## 검색 A/B 연구 (골든셋 32문항, page-level relevance)
| 청킹 | 방법 | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|
| 섹션 | 하이브리드 | 0.562 | 0.844 | **1.0** | 0.725 |
| 섹션 | +리랭커 | 0.625 | 0.875 | 0.938 | 0.760 |
| **코사인급락** | **하이브리드** | 0.656 | 0.906 | 0.969 | **0.785** |
| 코사인급락 | +리랭커 | 0.531 | 0.844 | 0.938 | 0.701 |
| 코사인급락 | +contextual(약식) | 0.469 | 0.812 | 0.875 | 0.641 |
| 코사인급락 | +contextual(페이지맥락) | 0.531 | **0.938** | 0.969 | 0.721 |

## 실측 발견 (정직)
1. **코사인급락 청킹 > 섹션 청킹** — 더 촘촘한 의미경계가 상위검색을 올림.
2. **리랭커·contextual은 "무조건 좋다"가 거짓** — 청킹에 따라 도움/해로움이 갈리는 상호작용(리랭커는 섹션엔 +MRR, 코사인급락엔 해로움).
3. **Contextual Retrieval은 이 코퍼스선 marginal** — 청크 텍스트가 이미 조항 헤더를 품어 situating이 중복.
4. **비만(E66) 4층 failure-localization** — 에이전틱 층이 상류 버그를 드러낸 대표 사례:

   | 층 | 증상 | 수정 |
   |---|---|---|
   | ① 청킹 필터 | 30자 미만 폐기 → `⑤비만(E66)` 9자 삭제 | 최소 10→2자 + 조각병합 |
   | ② 헤더 분리 | 리스트만 검색 → LLM이 면책을 **보상으로 오독** | — |
   | ③ 검색↔문맥 상충 | 헤더 합치니 희석돼 검색 안 됨 | **small-to-big** |
   | ④ 부모 품질 | raw OCR 잘림+bbox노이즈 | 깨끗한 파싱블록 페이지전체 |

   → 최종: 작은 청크로 검색 + 깨끗한 부모페이지로 생성 → *"비만으로 입원하면 보상하지 않습니다"* 정답.
5. **인덱싱**: ef_search 튜닝은 이 규모(193청크)선 무의미(recall@10 전 구간 1.0, Qdrant exact 폴백). **payload 인덱스**(dambo/btype/pages)로 담보종목·표·페이지 스코프 필터검색.
6. **OCR이 최대 레버** — 텍스트레이어 garbage → hit@3 0.9+.

## 실행
```bash
# 사전: Qdrant(:6333), Qwen3-8B(vLLM :8001) 기동
.venv-rag/bin/python rag/ingest/chunk_fix.py     # 청킹+인덱싱 (silson_v2_sem, small-to-big)
.venv-rag/bin/python rag/eval/eval.py            # 검색 hit@k/MRR + 리랭커 델타
.venv-rag/bin/python rag/eval/qdrant_tune.py     # ef_search 튜닝 + payload 필터검색
.venv-rag/bin/python rag/graph/agentic_rag.py    # LangGraph 에이전틱 RAG
```

## RAGAS 답변품질 — Ablation ladder (골든셋 32문항)
| 단계 | context_recall(검색) | faithfulness(근거) | answer_correctness(정답) |
|---|---|---|---|
| L0 dense | 0.694 | 0.847 | 0.636 |
| L1 +하이브리드 | 0.743 | 0.859 | 0.716 |
| L2 +small-to-big | 0.902 | 0.838 | **0.758** |
| L3 +리랭커 | **0.943** | 0.847 | 0.737 |
| L4 +에이전틱 | 0.927 | **0.906** | 0.701 |

**정직한 결론:**
- **small-to-big = 답변품질 최대 레버** (ac 0.636→0.758, cr +0.16).
- **리랭커: 검색 올리나(cr 0.943 최고) 답변엔 전환 안 됨(ac −0.02)** = "검색 승리 ≠ 답변 승리"의 실측.
- **에이전틱: faithfulness 최고(0.906, 환각 최소)지만 correctness는 trade** (신중·거부↑).

**Failure localization** (L4): 검색실패 1 · 생성환각 1 · **표/수치정밀 9**(cr=1·f=1인데 ac 낮음).
→ 남은 실패는 검색/환각이 아니라 **표·수치 정밀추출**(선택형 공제금액 등). 리랭커·에이전틱으로 안 고쳐짐 → 표 특화/수치검증/강한 모델이 다음.

> 심판=로컬 Qwen3-8B(노이즈 有, answer_correctness는 표현 민감). RAGAS 라이브러리는 로컬 vLLM에 순차호출(1 concurrent)로 ~60분 → **RAGAS 지표 정의(faithfulness/context_recall/answer_correctness=0.75·F1+0.25·유사도)를 스레드 병렬로 직접 구현**.

## 이월
- **RAGAS 답변품질 정량화** — 컴포넌트 ablation(naive→+hybrid→+rerank→+small-to-big→+agentic) × failure-localization(context_recall vs faithfulness)
- 라우터 tool 경로, transform 다양화 고도화

---
*이 repo는 원래 LangGraph 문서추출 에이전트(영수증/QC) 연습 하네스였고, 현재 실손 RAG가 최종본. 구 하네스 코드(`agent.py`,`domains/`,`eval/`)는 히스토리에 보존.*
