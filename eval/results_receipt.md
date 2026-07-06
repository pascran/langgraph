# 평가 리포트 — receipt 추출 하네스

- 도메인 `receipt` · 골든 6건 · 모델 `qwen3:8b`
- 채점: **결정론(재현 가능)** · 실패분류: 규칙 기반 taxonomy

## 집계
- exact-match **1.0** · field accuracy **0.7917** · abstention(정직성) **1.0**

## ★실패유형 분포 — 어디를 고쳐야 하나
- 총 실패 **5건**
  - `wrong_value` 5건 (100.0%) → few-shot·컨텍스트 개선

## 필드별 정확도
| vendor | date | total | items |
|---|---|---|---|
| 1.0 | 1.0 | 1.0 | 0.1667 |

## 문서별
| id | category | exact | 실패 필드(유형) |
|---|---|---|---|
| doc-01 | 정상-영수증 | ✅ | items(wrong_value) |
| doc-02 | 정상-세금계산서 | ✅ | items(wrong_value) |
| doc-03 | 결함-공급자/날짜없음 | ✅ | - |
| doc-04 | 날짜포맷변형 | ✅ | items(wrong_value) |
| doc-05 | 환각미끼-금액혼재 | ✅ | items(wrong_value) |
| doc-06 | 정상-다른형식 | ✅ | items(wrong_value) |

## 한계 (정직성)
- 합성·공개 스타일 데이터 6건, 1인 라벨 → 정밀 벤치 아닌 방향성.
- 채점·실패분류는 규칙 기반이라 재현 가능(클론하면 동일). LLM-judge(RAGAS)는 2차·델타만.
- rule_violation은 명시적 규칙(측정값 vs 규격)으로 판정 → 감사 가능.
