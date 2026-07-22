# LangGraph 문서 추출 에이전트

문서에서 구조화 필드를 뽑는 LangGraph 에이전트. `classify → extract → validate → (검증 실패 시 재추출 루프) → JSON`.
Pydantic 구조화 출력, 도메인 주입(영수증 / MES 품질검사 성적서), OpenAI 호환 백엔드(vLLM·Ollama·API)면 `LLM_BASE_URL`만 바꿔 쓴다.

추출만 하지 않고, 추출이 틀렸을 때 어디서·왜 틀렸는지 분류하는 평가 하네스를 같이 둔다(`eval/`).

## 그래프

```
START → classify → extract → validate ─┬─ (검증 실패 & retries<2) → extract   # 사이클
                                       └─ finalize → END
```

- `route`: 검증 실패 + 재시도 여유면 `extract`로 되돌아가고, 아니면 `finalize`.
- `extract`: `llm.with_structured_output(도메인 스키마)`. 재시도 시 직전 누락 필드를 프롬프트에 넣는다.
- `validate`: 필수 필드 + 도메인 업무규칙(예: QC 판정 = 측정값 vs 규격) 검사.
- `build_app(domain)`으로 스키마·필수필드·규칙을 교체한다(`domains/`).

## 실행

로컬 LLM (둘 중 하나):

```bash
# Ollama
ollama pull qwen3:8b && ollama serve                       # http://localhost:11434/v1
# vLLM
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384                                    # http://localhost:8000/v1
```

에이전트:

```bash
pip install -r requirements.txt
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen3:8b
python agent.py                     # sample_doc.txt
DOC=other.txt python agent.py
```

출력:

```json
{"doc_type":"영수증","vendor":"...","date":"2026-06-24","total":18500,"items":["..."],"validation":"ok"}
```

정상 문서는 한 번에 통과하고, 상호·날짜가 없는 문서는 `validate`가 누락을 잡아 `extract`로 되돌아간다(재시도 2회 후 `finalize`).
모델이 누락 필드를 `null` 대신 `"not found"` 같은 문자열로 채우는 경우가 있어, `validate`에서 placeholder를 누락으로 정규화한다(`_is_missing`).

## 평가 하네스 (`eval/`)

추출 결과를 골든셋과 비교해 필드별로 채점하고, 틀린 필드마다 원인을 분류한다. 채점·분류는 규칙 기반이라 LLM 없이 재현된다(`python3 tests/test_eval.py`).

- `score.py` — 필드별 정규화 비교(날짜·수치+단위·set-F1). exact-match, field accuracy, abstention(결함 필드에서 null을 냈는지).
- `diagnose.py` — 실패유형 분류: `missing` / `hallucinated` / `wrong_value` / `format` / `ambiguous` / `rule_violation`(판정이 측정값·규격 규칙과 어긋남).
- `report.py` — 실패유형 분포를 낸다("실패 N건 중 format 40%, rule_violation 25% …"). 프롬프트·출력 스키마·업무규칙 중 무엇을 고칠지 판단용.
- `gate.py` — baseline 대비 하락 또는 특정 실패유형 급증 시 `exit 1`(CI).
- `run.py` + `configs/` — 모델·재시도·프롬프트 변형 A/B.

도메인은 `domains/`의 스키마·규칙 + `eval/golden/*.json`만 바꾸면 붙는다(현재 영수증, MES 품질검사 성적서).

실측(qwen3:8b):

- QC 성적서 13건 — exact-match 1.0, field 1.0, abstention 1.0, 실패 0 (`eval/results_qc_report.md`).
- 영수증 6건 — exact-match 1.0, field 0.79. items 5건이 수량·가격까지 함께 뽑혀 `wrong_value`로 분류 (`eval/results_receipt.md`).
- 합성 오류를 주입했을 때의 분포는 `eval/results_qc_report.demo.md`.

```bash
python -m eval.run    --domain qc_report --config eval/configs/baseline.yaml
python -m eval.report --domain qc_report --pred eval/predictions_qc_report.json --model qwen3:8b
python -m eval.gate   --domain qc_report --pred eval/predictions_qc_report.json
```

평가 run은 **MLflow**로 추적한다(`eval/track.py`) — 학습이 아니라 **평가** 추적이다. config별 지표(exact-match·field·abstention·**실패유형별 count**)를 run으로 로깅해, A/B(baseline↔no_retry↔small_model)와 회귀를 대시보드로 비교한다. 채점·진단은 하네스가, 저장·비교·시각화는 MLflow가 맡는 구성.

```bash
python -m eval.track --domain qc_report --pred eval/predictions_qc_report.json \
    --model qwen3:8b --config eval/configs/small_model.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db     # → run 비교·지표 추세 (localhost:5000)
```

MLflow가 config 단위 A/B라면, 개별 문서·노드 단위 추적은 **Langfuse**로 한다(`eval/run_traced.py`). 문서 1건 = 트레이스 1개, LangGraph 내부 노드(classify·extract·validate·재시도)가 span으로 중첩 기록된다. 추출 직후 그 문서를 하네스로 채점·진단해 트레이스에 score를 붙인다 — `exact_match`·`field_accuracy`·`abstention`·`retries`(수치), 그리고 틀린 필드마다 `fail:<field> = <실패유형>`(범주). Langfuse UI에서 "어느 문서가 어느 노드에서 왜 틀렸나"를 호출 단위로 필터·조회한다. MLflow(run 단위)와 Langfuse(call 단위)는 상보적이다.

Langfuse는 OSS self-host라 폐쇄망에서 쓴다(Docker: postgres·clickhouse·redis·minio·web·worker, `localhost:3000`). SaaS를 못 쓰는 납품 환경 기준.

```bash
# Langfuse self-host 기동 후 (키는 self-host 프로젝트에서 발급)
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=...  LANGFUSE_SECRET_KEY=...
python -m eval.run_traced --domain qc_report --config eval/configs/o_ollama_qwen25.yaml
python -m eval.run_traced --domain receipt   --config ... --limit 3   # CPU 추론시 건수 제한
```

실측(qwen2.5:3b, WSL Ollama CPU): QC 성적서 13건 → 21개 필드 실패를 taxonomy로 분해해 트레이스에 첨부(`missing` 3 · `hallucinated` 3 · `rule_violation` 1 등). 결함 주입 문서(part_no 누락·판정 규격위반)는 `validate`가 잡아 재시도 2회까지 도는 사이클이 트레이스에 그대로 남는다.

RAGAS 기반 시맨틱 지표(faithfulness / factual correctness)는 `eval/ragas_eval.py`에 따로 있다(langchain 버전 충돌 때문에 venv 분리).

## 스택

LangGraph, LangChain, Pydantic, Ollama/vLLM(OpenAI 호환), Qwen. 평가 관측: MLflow(run 단위) + Langfuse(call 단위, self-host).
