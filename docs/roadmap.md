# 남은 고도화 항목 (리뷰 후 backlog)

`/review`(architect + code-reviewer)로 도출된, 아직 반영하지 않은 개선안. 전부 optional·수확체감 구간이며 포트폴리오로는 완성 상태. 가치순.

## 할 만한 것 (저노력·실가치)
- **content-hash 포인트 ID**: 현재 uuid4 → 타이브레이크로 hit@1/MRR이 재실행마다 ±노이즈(0.656↔0.594). 청크 텍스트 해시 ID로 바꾸면 검색 수치 **재현성 확보**. (`chunk_fix.py`)

## 청결도 (기계적, 무중단 브리지로 이미 동작)
- 잔여 eval 스크립트(custom_ragas·critic_rescore·build_ladder·build_final) `rag.core` 완전 이관. 현재는 `agentic_rag` 하위호환 심볼(m/cli/sv) 브리지로 동작.
- 이후 브리지 심볼 제거.

## 폴리시 (저가치)
- OCR 출력: stdout 리다이렉트 캡처 → 구조화 파일(bbox+type JSON) 직접 사용 (`ocr_all.py`).
- 파싱: 정규식 → pydantic `Block` 스키마 + 검증 (`parse.py`).
- 검증: TEDS 참조표 1개(REF_SUNTAEK) → 확장, CER/TEDS/KIE를 결과 반환형 모듈로 (`validate_kie.py`,`teds.py`).
- 일부 bare `except: pass`(Qdrant delete)를 명시적 `UnexpectedResponse`로.
- `build_ladder.py` L4: 재작성 쿼리로 검색하나 CRAG 채점은 원질문으로 하는 미세 drift.

## 보류(데이터상 무가치)
- 표 colspan/rowspan 전개·표 특화 파싱: **표 병목이 OCR/파싱이 아니라 생성-추론임이 판명**(README §OCR 병목 재검증). 27B thinking으로 해소(표 F1 +55%). 재개 불필요.
- 임베딩 디스크 캐시: 성능용, 코퍼스 소규모라 우선순위 낮음.

## 결론
차별점(계층별 측정 rigor·심판 검증·정직한 음성결과·27B 병목해소)과 코드 성숙도(core 라이브러리·19 테스트·패키징)는 완결. 위 항목은 여력 있을 때 점진 반영.
