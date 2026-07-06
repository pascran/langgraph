# PLAN — langgraph 실패-진단 평가 하네스 (schema-driven, 다중 도메인)

> 문서추출 에이전트를 **"어디서·왜 틀렸는지 필드별로 분해하고, 회귀를 막는 진단 시스템"**으로 업그레이드.
> 핵심 차별점: **도메인-무관 하네스**를 **영수증(baseline) + MES 품질검사 성적서(실도메인)**에 적용하고, 판정 오류를 **업무기준 위반(rule_violation)**으로 분류.

## 0. 타겟 정합 (왜 이걸 만드나)
| JD 언어 | 하네스 대응 |
|---|---|
| (Dfinite) 실패 원인을 **모델·데이터·검색·SQL·업무기준**으로 분해 | failure taxonomy: `wrong_value/hallucinated`(모델)·`missing`(주의)·`format`(스키마)·`ambiguous`(데이터)·**`rule_violation`(업무기준)** |
| (Dfinite) ERP/**MES**/SCM 도메인 | QC 성적서 도메인 (문서추출) — take-home(DB·SQL 집계)과 보완 |
| (파수) 프롬프트·모델 **A/B 테스트**로 품질 개선 | config별 A/B 러너 + 실패유형 델타 |
| (공통) **검증·평가 자동화** | 회귀 게이트 + 리포트 CI |
| 시그니처 | 결정론 채점(재현) + LLM-judge는 델타만 |

## 1. 아키텍처 (schema-driven)
**도메인 = (추출 스키마 + 필수필드 + 업무규칙 + 필드 채점기 + 골든셋).** 하네스(run/score/diagnose/report/gate)는 도메인-무관, 도메인 스펙만 갈아끼움.

```
langgraph/
  agent.py                 # ★리팩터: 하드코딩 Invoice → Domain 주입 (classify→extract→validate→retry→finalize 유지)
  domains/
    base.py                # Domain 데이터클래스(프로토콜)
    receipt.py             # Invoice 스키마 + 규칙 (기존 이식)
    qc_report.py           # QCReport 스키마 + ★판정 규칙(measured vs spec)
  eval/
    golden/
      receipt.json         # structured gold (기존 3→8, gold 필드 추가)
      qc_report.json       # ★신규 12~15건 (rule_violation·환각미끼·format·ambiguous 포함)
    configs/               # A/B 설정
      baseline.yaml  no_retry.yaml  small_model.yaml  prompt_v2.yaml
    baselines/             # 회귀 기준선
      receipt.baseline.json  qc_report.baseline.json
    run.py                 # (domain, config) → predictions
    score.py               # 결정론 필드 채점 (domain.field_scorers)
    diagnose.py            # ★실패유형 분류기 (taxonomy)
    report.py              # results.md (실패분포 헤드라인 + A/B 표 + 보정)
    gate.py                # baseline 대비 회귀 시 exit 1
    ragas_eval.py          # (유지) 2차 시맨틱 지표 — 선택
  tests/
    test_score.py  test_diagnose.py   # ★규칙 유닛테스트
  .github/workflows/eval.yml           # CI: run → gate
```

데이터 흐름:
```
golden(domain) → run.py(config별 에이전트) → predictions
  → score.py(필드별 결정론 채점) → diagnose.py(실패유형)
  → report.py(집계·실패분포·A/B) → gate.py(회귀 시 fail)
```

## 2. 도메인 스펙 (base.py)
```python
@dataclass
class Domain:
    name: str
    schema: type[BaseModel]              # 추출 스키마 (with_structured_output)
    required_fields: list[str]           # validate 노드 필수필드
    rules: list[Callable[[dict], list[str]]]   # 업무규칙 검사 → 위반 메시지
    field_scorers: dict[str, Callable]   # 필드별 (pred,gold)→correct? (정규화·fuzzy·set-F1)
    extract_instruction: str             # 도메인 튜닝 프롬프트
```
**agent.py 리팩터:** `build_app(domain)` — extract는 `domain.schema`로 구조화, validate는 `domain.required_fields` + `domain.rules` 실행. (에이전트를 하드코딩→구성가능으로 일반화 = 엔지니어링 시그널 자체)

### 2-1. receipt.py (baseline, 기존 이식)
`Invoice(vendor, date, total, items)`, required=[vendor,date,total], rules=[].

### 2-2. qc_report.py (★실도메인)
```python
class QCReport(BaseModel):
    part_no: str | None       # 품번
    lot_no: str | None        # 로트번호
    inspect_date: str | None  # 검사일 (YYYY-MM-DD)
    item: str | None          # 검사항목
    spec: str | None          # 규격/기준 (수치+단위+공차, 예 "10.0±0.2mm")
    measured: str | None      # 측정값 (수치+단위)
    judgment: str | None      # 판정(합격/불합격) ← 업무규칙 대상
    inspector: str | None
```
**★업무규칙(rules):** `spec`(범위)와 `measured`(값)로 **기대 판정**을 계산 → 에이전트 `judgment`와 불일치면 위반.
```python
def judgment_rule(e: dict) -> list[str]:
    exp = derive_judgment(e.get("measured"), e.get("spec"))  # 파싱: 값·공차범위 비교
    if exp and e.get("judgment") and normalize(e["judgment"]) != exp:
        return [f"판정 규칙 위반: measured={e['measured']} vs spec={e['spec']} → 기대 {exp}, 추출 {e['judgment']}"]
    return []
```

## 3. score.py — 결정론 필드 채점
- part_no/lot_no/inspector: 정규화 후 정확/퍼지
- inspect_date: YYYY-MM-DD 정규화 후 정확
- spec/measured: 수치+단위 파싱 후 값 비교
- judgment: 정규화(합격/불합격) 정확
- items(receipt): set P/R/**F1**
- 산출: 필드별 correct + 전체 exact-match

## 4. diagnose.py — ★실패유형 분류기 (결정론 우선)
각 오답 필드에 (pred, gold, 원문, meta)로 태깅:
```
gold=null  & pred=null/placeholder → correct(abstention)
gold=null  & pred=값               → hallucinated        (과생성)
gold=값    & pred=null/placeholder → missing             (주의·프롬프트)
gold=값    & norm(pred)==norm(gold), raw≠ → format       (스키마/구조화)
gold=값    & pred≠gold, pred∈원문   → wrong_value        (모델 추출오류)
gold=값    & pred≠gold, pred∉원문   → hallucinated(wrong)(모델 환각)
meta=ambiguous & 오답             → ambiguous           (데이터 품질)
judgment: pred ≠ rule판정(measured,spec) → ★rule_violation (업무기준)
```
경계 애매한 "pred∈원문?"만 fuzzy 부분매칭(결정론), LLM-judge 폴백은 최소.

## 5. 지표 (report.py 헤드라인)
- 필드별 정확도 · exact-match율 · item F1(receipt)
- **환각율**(gold=null인데 채움) · **abstention 정확도**(결함필드서 올바로 null=정직성)
- **누락율** · **format-error율** · **★rule_violation율**(판정 규칙 위반)
- **실패유형 분포**("실패 N건 중 format 35%·rule_violation 25%·hallucinated 20%·missing 15%·ambiguous 5%")
- 2차: RAGAS faithfulness/FC(델타만)

## 6. A/B (run.py + configs/)
`python -m eval.run --domain qc_report --config configs/prompt_v2.yaml` → predictions 분리.
`python -m eval.report --domain qc_report --compare baseline prompt_v2` → 지표·실패유형 델타 표 + 승자.
config 축: 모델(대/소)·프롬프트 변형·retry on/off·structured on/off.

## 7. 회귀 게이트 (gate.py + baselines/ + CI)
현재 run vs `baselines/{domain}.baseline.json`: 전체 하락 임계 초과 or 특정 실패유형 급증 → **exit 1**. `.github/workflows/eval.yml`에서 run→gate.

## 8. (옵션·Phase 6) confidence 보정
extract가 필드별 확신도(0~1) 뱉게(2차 구조화 호출) → "낮은 확신=실제 오답" 상관 측정(간이 ECE/빈 표) → **언제 사람에게 넘길지(HITL)** 근거. 컨설턴트/FDE 각.

## 9. 빌드 순서
- **Phase 0** — domains/(base+receipt+qc_report) + golden/(receipt 8, **qc_report 12~15**: 정상/필드누락/날짜·수치 format변형/환각미끼/공차 ambiguous/**판정오류 rule_violation** 케이스) + agent.py 리팩터. ← 골든이 자산의 8할
- **Phase 1** — score.py + test_score.py
- **Phase 2** — diagnose.py + test_diagnose.py ← ★스타
- **Phase 3** — report.py (실패분포 헤드라인)
- **Phase 4** — run.py + configs/ (A/B)
- **Phase 5** — gate.py + baselines/ + CI
- **Phase 6** — (옵션) confidence 보정
- **Phase 7** — README 재작성 + 정직성 섹션 + 결과 캡처

## 10. 정직성·재현성 가드 (산출물에 박기)
- 합성·공개 규정 스타일 데이터(**회사 실데이터·IP 금지**), 1인 라벨, 소N → 정밀 벤치 아닌 방향성.
- 결정론 채점·taxonomy = 재현 가능(클론하면 동일). LLM-judge(RAGAS)는 2차·델타만.
- `rule_violation`은 명시적 규칙(measured vs spec)으로 판정 → 감사 가능("AI가 실패라 함"이 아니라 규칙이 판정).
- seed·config·baseline 커밋.

## 11. 견적 / 데모 서사
- ~10~12 파일, 총 ~800~1000 LOC. 골든 라벨링이 손 많이 감(핵심). 반나절~하루.
- Ollama 로컬 재사용(기존 스택). 추가 인프라 0.
- **데모 한 줄:** "제 추출 에이전트는 값만 뽑는 게 아니라, 실패의 25%가 **판정 규칙 위반**이라고 짚는 리포트와 회귀를 막는 게이트를 같이 냅니다. 그건 프롬프트가 아니라 **업무규칙 적용**을 고쳐야 한다는 뜻이죠."
