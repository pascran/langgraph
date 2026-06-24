# LangGraph 문서 추출 에이전트

로컬 LLM(vLLM/Ollama) 위에서 도는 **LangGraph 에이전트** 데모.
문서를 받아 **분류 → 추출 → 검증 → (실패 시 재추출 루프) → 구조화 JSON**으로 변환한다.

> 목적: LangGraph의 핵심(상태·조건분기·**사이클**)을 최소 코드로 보여주고,
> "왜 RAG를 프레임워크 없이 직접 짰나 / 프레임워크는 언제 쓰나"에 **써본 사람으로** 답하기 위함.

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
- **조건 분기**(`route_after_validate`): 검증 실패 + 재시도 여유 → `extract`로 되돌아감, 아니면 `finalize`
- **구조화 출력**: `llm.with_structured_output(Invoice)` — Pydantic 스키마로 강제
- **백엔드 독립**: `base_url`만 바꾸면 vLLM/Ollama/상용 API 무엇이든

> E사(애자일소다 AI Agent) 직무 = "비정형 문서 → 정보 추출·검증·자동화 워크플로"와 1:1.

---

## 3. 실행

### A) 로컬 LLM 띄우기 (둘 중 하나)

**vLLM (graphrag 스택과 일관, 처리량 높음):**
```bash
pip install vllm
vllm serve Qwen/Qwen3-30B-A3B \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384 --gpu-memory-utilization 0.9
# → OpenAI 호환 엔드포인트 http://localhost:8000/v1
```
> ⚠️ 실전 함정: vLLM 최신 휠은 새 CUDA로 빌드된 torch를 끌고 온다. GPU 드라이버가
> 그보다 낮으면(예: driver CUDA 12.4) 엔진 init 단계에서
> `RuntimeError: The NVIDIA driver on your system is too old` 로 죽는다.
> → 드라이버에 맞는 torch가 들어간 vLLM 버전을 고르거나, 아래 Ollama를 쓴다.

**Ollama (마찰 0, 빠른 시작 — 이 데모 검증에 실제 사용):**
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

## 4. 실제 검증 결과

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

> **검증 중 잡은 실전 버그**: 모델이 누락 필드를 `null`이 아니라 `"not found"` *문자열*로
> 채워, 단순 falsy 검사(`if not e.get(k)`)를 통과시켜 루프가 안 돌았다. → 검증 노드에
> placeholder 문자열(`"not found"`, `"없음"` 등)을 누락으로 정규화하는 `_is_missing()`을 추가.
> (구조화 추출 모델의 흔한 환각 — *모델 출력을 믿지 말고 경계에서 검증*.)

---

## 5. 설계 의도 (면접용)

- **왜 LangGraph?** 단순 추출은 체인으로 충분하지만, **검증→재시도 루프**처럼 *상태에 따라 되돌아가는 흐름*은 그래프가 자연스럽다. 에이전트 = 사이클이 필요한 워크플로.
- **왜 RAG는 직접 짰나(graphrag) vs 여기선 LangGraph?** 검색 융합(RRF·rerank)은 세밀 제어가 필요해 커스텀이 나았고, **에이전트 워크플로는 상태·분기·관찰성을 LangGraph가 표준으로 제공**해 프레임워크가 이득. → 도구는 작업 성격에 맞춰 고른다.
- **확장 포인트**: 추출 노드에 외부 도구(DB 조회·검색) tool-calling 추가, LangSmith로 관찰성, 멀티 문서 타입별 서브그래프.

## 기술 스택
`LangGraph` · `LangChain` · `Pydantic` · `Ollama`/`vLLM` (로컬 LLM, OpenAI 호환) · `Qwen3` (검증: `qwen3:8b` on H100, 운영 권장: `Qwen3-30B-A3B`)
