# LangGraph 문서 추출 에이전트

로컬 LLM(Ollama/vLLM) 위에서 도는 **LangGraph 에이전트** 예제.
문서를 받아 **분류 → 추출 → 검증 → (실패 시 재추출 루프) → 구조화 JSON**으로 변환한다.

상태(State)·조건 분기·**사이클(루프)** 을 최소 코드로 보여주는 게 목표다.
일반 체인(DAG)으로는 표현 못 하는 *"검증 실패 → 추출로 되돌아가 재시도"* 를 그래프로 구현했다.

---

## 1. LangGraph가 뭔가 (LangChain·n8n과 비교)

| | 정체 | 빌드 방식 |
|---|---|---|
| **n8n / Zapier** | 비주얼 노코드 워크플로 | UI 드래그&드롭 |
| **LangChain** | LLM 앱 빌딩블록 **코드 라이브러리** | Python으로 조립 |
| **LangGraph** | LangChain 위에서 **에이전트 워크플로를 "그래프/상태머신"으로** | Python (시각화는 가능) |

**LangGraph의 멘탈모델:**
- **State**: 모든 노드가 읽고 갱신하는 공유 dict
- **Node**: 함수 하나 (한 가지 책임)
- **Edge**: 노드 간 흐름. **조건 분기**(상태에 따라 다음 노드 결정)와 **사이클**(되돌아가기)이 가능
- **사이클이 핵심 차별점**: 일반 체인(DAG)은 한 방향. LangGraph는 "검증 실패 → 추출로 되돌아가 재시도" 같은 **루프**가 됨 = 에이전트의 본질(재시도·반성·반복)

---

## 2. 이 에이전트의 그래프

```
        START
          │
      ┌───────┐
      │classify│  문서 종류 분류 (영수증/계약서/기타)
      └───────┘
          │
      ┌───────┐ ◄──────────────┐
      │extract │  구조화 필드 추출 │ (재시도: 직전 누락 필드를
      └───────┘  (Pydantic 스키마)│  프롬프트에 넣어 보강)
          │                      │
      ┌───────┐                  │
      │validate│  필수 필드 검증   │
      └───────┘                  │
          │                      │
   조건 분기 ─ 실패 & retries<2 ──┘   (← 사이클)
          │
       성공 / 재시도 소진
          │
      ┌────────┐
      │finalize │  최종 구조화 JSON 조립
      └────────┘
          │
         END
```

- **노드**: `classify` → `extract` → `validate` → (`extract` 루프 | `finalize`)
- **조건 분기**(`route`): 검증 실패 + 재시도 여유 → `extract`로 되돌아감, 아니면 `finalize`
- **구조화 출력**: `llm.with_structured_output(도메인 스키마)` — Pydantic 스키마로 강제
- **도메인-주입**: `build_app(domain)` — 스키마·필수필드·업무규칙을 갈아끼움(영수증 · QC 성적서). `validate`는 필수필드 + 업무규칙(예: QC 판정 = 측정값 vs 규격)을 검사한다.
- **백엔드 독립**: `base_url`만 바꾸면 vLLM/Ollama/상용 API 무엇이든

---

## 3. 실행

### A) 로컬 LLM 띄우기 (둘 중 하나)

**vLLM (처리량 높음):**
```bash
pip install vllm
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384 --gpu-memory-utilization 0.9
# → OpenAI 호환 엔드포인트 http://localhost:8000/v1
```
> 주의: vLLM 최신 휠은 새 CUDA로 빌드된 torch를 끌고 온다. GPU 드라이버가 그보다 낮으면
> (예: driver 550.144 = CUDA 12.4 상한) 엔진 init에서 `RuntimeError: The NVIDIA driver on your
> system is too old` 로 죽는다. → **드라이버에 맞는 torch가 든 vLLM으로 핀**해야 한다.
> 이 레포는 `serve/`에 그 셋업을 스크립트로 둔다(vLLM 0.8.5 = torch 2.6.0+**cu124**):
> `bash serve/setup_vllm.sh && bash serve/serve_vllm.sh` (H100, `Qwen3-30B-A3B` TP=2). 자세히는 `serve/README.md`.

**Ollama (빠른 시작 — 이 데모는 Ollama로 돌렸다):**
```bash
ollama pull qwen3:8b    # 또는 qwen3:30b-a3b
ollama serve            # http://localhost:11434/v1
```
> Ollama는 호환 CUDA 런타임을 자체 번들 → 드라이버 12.4에서도 그대로 GPU 추론.

### B) 에이전트 실행
```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt

# vLLM이면:
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL="Qwen/Qwen3-30B-A3B"
# Ollama면:
# export LLM_BASE_URL=http://localhost:11434/v1
# export LLM_MODEL="qwen3:30b-a3b"

python agent.py            # sample_doc.txt 처리
DOC=다른문서.txt python agent.py
```

출력 예:
```json
{
  "doc_type": "영수증",
  "vendor": "...",
  "date": "2026-06-24",
  "total": 18500,
  "items": ["..."],
  "validation": "ok"
}
```

---

## 4. 검증 결과

H100에 Ollama(`qwen3:8b`)를 띄워 end-to-end로 돌린 로그.

**(1) 정상 영수증 → 한 번에 통과**
```
[classify] doc_type=영수증
[extract] {'vendor': '한빛문구 강남점', 'date': '2026-06-24', 'total': 24100, 'items': [...]}
[validate] issues=[] retries=0
→ validation: ok
```

**(2) 상호·날짜 없는 결함 문서 → 사이클(재추출 루프)이 실제로 발동**
```
[extract]  {'vendor': None, 'date': None, 'total': 19600, ...}
[validate] issues=['vendor 누락', 'date 누락'] retries=1   ← 검증 실패
[extract]  {'vendor': None, 'date': None, 'total': 19600, ...}   ← extract로 되돌아옴(사이클)
[validate] issues=['vendor 누락', 'date 누락'] retries=2
→ 재시도 소진 → finalize, validation: ['vendor 누락','date 누락']
```
`extract → validate → extract → validate → finalize` — DAG 체인은 못 하는 **되돌아가기**가 로그에 그대로 찍힌다.

> **버그 메모**: 모델이 누락 필드를 `null`이 아니라 `"not found"` 문자열로 채워, falsy 검사
> (`if not e.get(k)`)를 통과시켜 루프가 안 돌았다. → 검증 노드에서 placeholder 문자열
> (`"not found"`, `"없음"` 등)을 누락으로 정규화하는 `_is_missing()`을 추가.

---

## 5. 평가: RAGAS로 추출 품질 측정 (`eval/`)

RAGAS로 추출 결과의 품질을 수치화한다. 추출 에이전트는 RAG가 아니므로 랭킹 지표
(context precision/recall)는 빼고, 추출에 맞는 둘만 쓴다:
- **Faithfulness** — 추출한 값이 원본 문서에 근거하는가 (= 환각 안 했는가)
- **FactualCorrectness** — 추출 결과가 골든 정답과 사실적으로 일치하는가

| 문서 | faithfulness | factual_correctness |
|---|---|---|
| doc-01 정상 영수증 | 1.000 | 0.800 |
| doc-02 정상 세금계산서 | 1.000 | 0.860 |
| doc-03 결함(공급자·날짜 없음) | **0.500** | 0.890 |
| **평균** | **0.833** | **0.850** |

> doc-03의 faithfulness가 0.5인 건 에이전트 오류가 아니라 지표 성질이다. "공급자는 명시되지
> 않음" 같은 부재 진술은 문서에 entailment되지 않아 faithfulness가 깎인다 — faithfulness는
> '모름' 답변을 잘 보상하지 못한다. 추출/거절 과제에선 factual correctness가 더 맞는 신호다.
> N=3 합성셋·judge 1개라 정밀 벤치가 아니라 방향성 확인용이다.

**2단계 디커플링**: RAGAS(0.4.x)가 아직 langchain 1.x와 비호환이라, 에이전트(`venv-agent`)는
예측을 `predictions.json`으로 떨구고, RAGAS(`venv-ragas`, langchain 0.3 핀)는 그걸 읽어 평가한다 —
두 의존성 스택을 안 섞는다.

```bash
# 1) 에이전트로 예측 덤프 (langgraph 스택)
PYTHONPATH=. python eval/dump_predictions.py
# 2) RAGAS로 평가 (별도 venv, eval/requirements-ragas.txt)
python eval/ragas_eval.py          # judge: qwen2.5:7b-instruct
```

---

## 6. 실패 진단 평가 하네스 (`eval/`, schema-driven)

RAGAS(§5)가 LLM-judge 기반 시맨틱 점수라면, 이쪽은 **결정론 채점 + 실패 원인 분해**다.
추출이 틀렸을 때 "얼마나 틀렸나"가 아니라 **"어디서·왜 틀렸나"** 를 분류한다.

- **도메인-무관**: `domains/`의 스키마·필수필드·업무규칙 + `eval/golden/*.json`만 갈아끼우면 새 도메인에 붙는다. 현재 영수증(baseline) + **MES 품질검사 성적서** 2도메인.
- **결정론 채점**(`score.py`): 필드별 정규화 비교(날짜·수치+단위·set-F1). exact-match · field accuracy · **abstention 정확도**(결함 필드에서 올바로 null 냈나).
- **실패유형 분류**(`diagnose.py`): 오답마다 규칙으로 태깅 — `missing`(주의) · `hallucinated`(과생성) · `wrong_value`(추출오류) · `format`(스키마) · `ambiguous`(데이터 모호) · **`rule_violation`**(업무기준: 판정 ≠ 측정값 vs 규격).
- **리포트**(`report.py`): 실패유형 분포를 헤드라인으로 — "실패 X건 중 format 40%·rule_violation 25%…" → 프롬프트를 고칠지, 출력 스키마를 고칠지, 업무규칙을 고칠지 지목.
- **회귀 게이트**(`gate.py`): baseline 대비 품질 하락/특정 실패유형 급증 시 CI 차단.
- **A/B**(`run.py` + `configs/`): 모델·재시도·프롬프트 변형별 지표·실패유형 델타.

채점·분류는 규칙 기반이라 재현 가능하다(LLM 없이 검증).

**실측(qwen3:8b, H100 Ollama):**
- QC 성적서(13건): exact-match 1.0 · field 1.0 · abstention 1.0 · 실패 0 — 누락 필드 abstention과 판정 규칙까지 통과(`eval/results_qc_report.md`).
- 영수증(6건): exact-match 1.0 · field 0.79 · 실패 5건 전부 `items`의 **과다추출**(수량·가격을 함께 뽑음) → `wrong_value`로 분류(`eval/results_receipt.md`).
- 실패유형이 주입 오류에서 어떻게 갈리는지는 `eval/results_qc_report.demo.md`(합성) 참고.

> 서빙 노트: 원래 vLLM(§3)으로 Qwen3-30B-A3B를 서빙하려 했으나, 이 H100 드라이버(CUDA 12.4 상한)에서
> 쓸 수 있는 vLLM 0.8.5가 Qwen3 토크나이저와 충돌해(새 vLLM은 CUDA 12.8+ 요구) 실측은 Ollama로 돌렸다.
> 에이전트는 백엔드 독립이라 `LLM_BASE_URL`만 바꾸면 된다.

```bash
# 결정론 코어 검증 (LLM 불필요)
python3 tests/test_eval.py

# 실측 (LLM 서빙 필요): 실행 → 리포트 → 게이트
python -m eval.run    --domain qc_report --config eval/configs/baseline.yaml
python -m eval.report --domain qc_report --pred eval/predictions_qc_report.json --model Qwen3-30B-A3B
python -m eval.gate   --domain qc_report --pred eval/predictions_qc_report.json
```

---

## 7. 설계 노트

- **왜 LangGraph인가** — 단순 추출이면 일반 체인으로 충분하다. "검증 실패 시 추출로 되돌아가는" 흐름은 상태에 따라 경로가 갈리고 되돌아가는 사이클이라 그래프/상태머신으로 짜는 게 맞다.
- **확장 포인트** — 추출 노드에 외부 도구(DB 조회·검색) tool-calling, LangSmith 관찰성, 문서 타입별 서브그래프.

## 기술 스택
`LangGraph` · `LangChain` · `Pydantic` · `Ollama`/`vLLM` (로컬 LLM, OpenAI 호환) · `Qwen3` (검증: `qwen3:8b` on H100, 운영 권장: `Qwen3-30B-A3B`)
