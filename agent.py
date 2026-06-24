#!/usr/bin/env python3
"""
LangGraph 문서 추출 에이전트 (데모)
====================================
입력 문서 → [분류 → 추출 → 검증 → (실패 시 재추출 루프) → 마무리] → 구조화 JSON

LangGraph의 핵심(일반 체인과 다른 점)을 보여준다:
  1) StateGraph  : 노드 간에 '상태(State)'를 공유
  2) 조건 분기   : 상태에 따라 다음 노드를 동적으로 결정
  3) 사이클(loop): 검증 실패 시 추출 노드로 되돌아가는 '순환' — DAG 체인은 못 함
  4) 백엔드 독립 : OpenAI 호환 엔드포인트면 vLLM/Ollama/API 무엇이든

E사(애자일소다 AI Agent 팀) 직무 = "업무 절차/판단 기준 기반 Agent 처리 흐름 설계"와 정합.
"""
import os
import json
from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ── LLM: OpenAI 호환 엔드포인트 (vLLM/Ollama/상용 API 무관, base_url만 바꾸면 됨) ──
llm = ChatOpenAI(
    base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("LLM_API_KEY", "EMPTY"),   # 로컬 vLLM/Ollama는 키 불필요
    model=os.environ.get("LLM_MODEL", "Qwen/Qwen3-30B-A3B"),
    temperature=0,
)


# ── 추출 결과 스키마 (구조화 출력) ──
class Invoice(BaseModel):
    vendor: str | None = Field(None, description="공급자/상호")
    date: str | None = Field(None, description="날짜 (YYYY-MM-DD)")
    total: int | None = Field(None, description="총액 (원, 숫자만)")
    items: list[str] = Field(default_factory=list, description="품목 목록")


# ── 그래프 상태: 모든 노드가 읽고 갱신하는 공유 dict ──
class State(TypedDict):
    text: str          # 입력 문서 원문
    doc_type: str      # 분류 결과
    extracted: dict    # 추출된 구조화 데이터
    issues: list[str]  # 검증에서 발견한 문제
    retries: int       # 추출 재시도 횟수
    answer: str        # 최종 출력(JSON 문자열)


# ── 노드 1: 문서 종류 분류 ──
def classify(state: State) -> dict:
    resp = llm.invoke(
        "다음 문서의 종류를 한 단어로만 답하라(예: 영수증, 계약서, 기타).\n\n"
        + state["text"][:800]
    )
    doc_type = resp.content.strip().split()[0]
    print(f"[classify] doc_type={doc_type}")
    return {"doc_type": doc_type}


# ── 노드 2: 구조화 필드 추출 (LangChain structured output) ──
def extract(state: State) -> dict:
    hint = ""
    if state.get("issues"):
        # 재시도 시: 직전 문제를 프롬프트에 넣어 보강 (단순 재실행이 아님)
        hint = f"\n\n[직전 추출에서 누락/오류]: {', '.join(state['issues'])} — 이 필드를 특히 주의해 다시 추출하라."
    structured_llm = llm.with_structured_output(Invoice)
    inv: Invoice = structured_llm.invoke(
        "다음 문서에서 vendor/date/total/items를 정확히 추출하라." + hint + "\n\n" + state["text"]
    )
    print(f"[extract] {inv.model_dump()}")
    return {"extracted": inv.model_dump()}


# ── 노드 3: 검증 (규칙 기반 — 필수 필드 확인) ──
def validate(state: State) -> dict:
    e = state["extracted"]
    issues = [f"{k} 누락" for k in ("vendor", "date", "total") if not e.get(k)]
    retries = state.get("retries", 0) + (1 if issues else 0)
    print(f"[validate] issues={issues} retries={retries}")
    return {"issues": issues, "retries": retries}


# ── 노드 4: 마무리 (최종 구조화 JSON 조립) ──
def finalize(state: State) -> dict:
    out = {
        "doc_type": state["doc_type"],
        **state["extracted"],
        "validation": "ok" if not state["issues"] else state["issues"],
    }
    return {"answer": json.dumps(out, ensure_ascii=False, indent=2)}


# ── 조건 분기: 검증 실패 + 재시도 여유 있으면 추출로 되돌아감(사이클), 아니면 마무리 ──
def route_after_validate(state: State) -> Literal["extract", "finalize"]:
    if state["issues"] and state["retries"] < 2:
        return "extract"
    return "finalize"


# ── 그래프 조립 ──
def build_app():
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("extract", extract)
    g.add_node("validate", validate)
    g.add_node("finalize", finalize)

    g.add_edge(START, "classify")
    g.add_edge("classify", "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges(
        "validate", route_after_validate,
        {"extract": "extract", "finalize": "finalize"},   # ← 사이클(루프)
    )
    g.add_edge("finalize", END)
    return g.compile()


if __name__ == "__main__":
    app = build_app()
    doc_path = os.environ.get("DOC", "sample_doc.txt")
    text = open(doc_path, encoding="utf-8").read()
    result = app.invoke({"text": text, "retries": 0})
    print("\n=== 최종 구조화 출력 ===")
    print(result["answer"])
