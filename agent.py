#!/usr/bin/env python3
"""LangGraph 문서 추출 에이전트 (도메인-주입)
====================================================
입력 문서 → [분류 → 추출 → 검증 → (실패 시 재추출 루프) → 마무리] → 구조화 JSON

LangGraph의 핵심(일반 체인과 다른 점):
  1) StateGraph  : 노드 간 '상태(State)' 공유
  2) 조건 분기   : 상태에 따라 다음 노드를 동적으로 결정
  3) 사이클(loop): 검증 실패 시 추출 노드로 되돌아가는 '순환'
  4) 백엔드 독립 : OpenAI 호환 엔드포인트면 vLLM/Ollama/API 무엇이든 (base_url만 교체)

★도메인-무관: build_app(domain)으로 스키마·필수필드·업무규칙을 주입한다.
   기본 도메인 = 영수증(receipt). 다른 도메인은 domains/ 참조.
"""
import os
import json

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from domains.base import Domain
from domains.receipt import DOMAIN as RECEIPT

# ── LLM: OpenAI 호환 엔드포인트 (vLLM/Ollama/API 무관) ──
llm = ChatOpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
    model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-30B-A3B"),
    temperature=0,
)

# ── State: 모든 노드가 읽고 갱신하는 공유 dict ──
from typing import TypedDict, Literal  # noqa: E402


class State(TypedDict):
    text: str
    doc_type: str
    extracted: dict
    issues: list[str]
    retries: int
    answer: str


# 모델이 null 대신 채워넣는 placeholder(= 사실상 '누락')
_PLACEHOLDERS = {"", "not found", "n/a", "na", "none", "null", "unknown",
                 "없음", "미상", "해당없음", "확인불가", "명시되지 않음"}


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in _PLACEHOLDERS:
        return True
    return False


def build_app(domain: Domain | None = None, max_retries: int = 2):
    """도메인을 주입해 그래프를 조립한다. domain=None이면 영수증(기본)."""
    domain = domain or RECEIPT

    def classify(state: State) -> dict:
        resp = llm.invoke(
            "다음 문서의 종류를 한 단어로만 답하라(예: 영수증, 성적서, 계약서, 기타).\n\n"
            + state["text"][:800]
        )
        return {"doc_type": resp.content.strip().split()[0] if resp.content.strip() else "기타"}

    def extract(state: State) -> dict:
        hint = ""
        if state.get("issues"):
            hint = f"\n\n[직전 문제]: {', '.join(state['issues'])} — 이 부분을 특히 주의해 다시 추출하라."
        structured = llm.with_structured_output(domain.schema)
        obj = structured.invoke(domain.extract_instruction + hint + "\n\n" + state["text"])
        return {"extracted": obj.model_dump()}

    def validate(state: State) -> dict:
        e = state["extracted"]
        issues = [f"{k} 누락" for k in domain.required_fields if _is_missing(e.get(k))]
        issues += domain.check_rules(e)                       # ★업무규칙 검사
        retries = state.get("retries", 0) + (1 if issues else 0)
        return {"issues": issues, "retries": retries}

    def finalize(state: State) -> dict:
        out = {"doc_type": state["doc_type"], **state["extracted"],
               "validation": "ok" if not state["issues"] else state["issues"]}
        return {"answer": json.dumps(out, ensure_ascii=False, indent=2)}

    def route(state: State) -> Literal["extract", "finalize"]:
        return "extract" if state["issues"] and state["retries"] < max_retries else "finalize"

    g = StateGraph(State)
    for name, fn in [("classify", classify), ("extract", extract),
                     ("validate", validate), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_edge("classify", "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", route, {"extract": "extract", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


if __name__ == "__main__":
    app = build_app()
    text = open(os.environ.get("DOC", "sample_doc.txt"), encoding="utf-8").read()
    result = app.invoke({"text": text, "retries": 0})
    print("\n=== 최종 구조화 출력 ===")
    print(result["answer"])
