#!/usr/bin/env python3
"""1단계: 에이전트를 골든셋 문서에 돌려 예측을 RAGAS 입력 형태로 떨군다.

venv-agent(langgraph 스택)에서 실행한다. RAGAS는 의존성 충돌을 피하려 별도
venv-ragas에서 이 출력(predictions.json)을 읽어 평가한다 — 두 스택을 분리한다.

각 예측을 RAGAS SingleTurnSample 필드로 변환:
  user_input         : 추출 지시(질문)
  retrieved_contexts : [원본 문서]  ← 추출이 근거해야 할 컨텍스트
  response           : 에이전트 추출 결과를 자연어로 직렬화
  reference          : 골든 정답(자연어)
"""
import os
import json
from pathlib import Path

# 에이전트 LLM 백엔드(추출용) — agent import 전에 설정해야 함
os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
os.environ.setdefault("LLM_MODEL", "qwen3:8b")
os.environ.setdefault("LLM_API_KEY", "ollama")

from agent import build_app  # noqa: E402  (env 설정 후 import)

QUESTION = "이 문서에서 공급자(vendor), 거래일(date), 총액(total), 품목(items)을 추출하라."

EVAL_DIR = Path(__file__).resolve().parent


def to_response(e: dict) -> str:
    """추출 dict → 자연어 문장(claim 분해가 잘 되도록). null은 '명시되지 않음'으로."""
    vendor = e.get("vendor") or "문서에 명시되지 않음"
    date = e.get("date") or "문서에 명시되지 않음"
    total = e.get("total")
    total_s = f"{total}원" if total is not None else "문서에 명시되지 않음"
    items = e.get("items") or []
    items_s = ", ".join(items) if items else "없음"
    return (
        f"공급자는 {vendor}이다. 거래일은 {date}이다. "
        f"총액은 {total_s}이다. 품목은 {items_s}이다."
    )


def main() -> None:
    golden = json.loads((EVAL_DIR / "golden_extractions.json").read_text(encoding="utf-8"))
    app = build_app()
    preds = []
    for g in golden["items"]:
        result = app.invoke({"text": g["document"], "retries": 0})
        extracted = result.get("extracted", {})
        response = to_response(extracted)
        print(f"[{g['id']}] extracted={extracted}")
        preds.append({
            "id": g["id"],
            "category": g["category"],
            "user_input": QUESTION,
            "retrieved_contexts": [g["document"]],
            "response": response,
            "reference": g["reference"],
        })
    out = EVAL_DIR / "predictions.json"
    out.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out} ({len(preds)}건)")


if __name__ == "__main__":
    main()
