# 평가 리포트 — qc_report 추출 하네스

- 도메인 `qc_report` · 골든 13건 · 모델 `demo(합성 오류주입)`
- 채점: **결정론(재현 가능)** · 실패분류: 규칙 기반 taxonomy

## 집계
- exact-match **0.6154** · field accuracy **0.9327** · abstention(정직성) **0.5**

## ★실패유형 분포 — 어디를 고쳐야 하나
- 총 실패 **7건**
  - `missing` 1건 (14.3%) → 프롬프트에서 해당 필드 추출 강조
  - `hallucinated` 1건 (14.3%) → null 허용 강제 + 근거(원문 span) 요구
  - `wrong_value` 1건 (14.3%) → few-shot·컨텍스트 개선
  - `format` 2건 (28.6%) → 출력 스키마/형식(YYYY-MM-DD 등) 지시 강화
  - `ambiguous` 1건 (14.3%) → 원문 자체가 모호 — 데이터 품질 이슈(모델 탓 아님)
  - `rule_violation` 1건 (14.3%) → ★업무규칙(판정=측정값 vs 규격) 후처리로 강제

## 필드별 정확도
| part_no | lot_no | inspect_date | item | spec | measured | judgment | inspector |
|---|---|---|---|---|---|---|---|
| 0.9231 | 0.9231 | 0.8462 | 1.0 | 0.9231 | 0.9231 | 0.9231 | 1.0 |

## 문서별
| id | category | exact | 실패 필드(유형) |
|---|---|---|---|
| qc-01 | 정상-합격 | ❌ | inspect_date(format) |
| qc-02 | 정상-불합격 | ❌ | judgment(rule_violation) |
| qc-03 | 범위표기-합격 | ✅ | - |
| qc-04 | 최소기준(이상)-불합격 | ✅ | - |
| qc-05 | 날짜포맷변형 | ❌ | inspect_date(format) |
| qc-06 | 결함-로트누락 | ❌ | lot_no(hallucinated) |
| qc-07 | 결함-검사자누락 | ✅ | - |
| qc-08 | 환각미끼-수치혼재 | ✅ | measured(wrong_value) |
| qc-09 | 판정생략-규칙유도 | ✅ | - |
| qc-10 | 공차모호 | ✅ | spec(ambiguous) |
| qc-11 | 노이즈문서 | ❌ | part_no(missing) |
| qc-12 | 최대기준(이하)-합격 | ✅ | - |
| qc-13 | 범위초과-불합격 | ✅ | - |

## 한계 (정직성)
- 합성·공개 스타일 데이터 13건, 1인 라벨 → 정밀 벤치 아닌 방향성.
- 채점·실패분류는 규칙 기반이라 재현 가능(클론하면 동일). LLM-judge(RAGAS)는 2차·델타만.
- rule_violation은 명시적 규칙(측정값 vs 규격)으로 판정 → 감사 가능.
