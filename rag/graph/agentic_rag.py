"""LangGraph 에이전틱 RAG: route → retrieve → CRAG → generate(small-to-big) → Self-RAG.
생성=ThinkingCap-27B(thinking). core 라이브러리(Embedder/Retriever/LLMClient/parent) 사용."""
from typing import TypedDict
from rag.core import config
from rag.core.embedder import Embedder
from rag.core.llm import LLMClient
from rag.retrieval.retriever import Retriever
from rag.retrieval.parent import build_pages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

_emb = Embedder(config.EMBED_MODEL)
_retr = Retriever(_emb, config.QDRANT_URL, config.COLLECTION)
_gen = LLMClient(config.GEN_URL, config.GEN_MODEL, timeout=300)  # ThinkingCap-27B (:8001); thinking 생성이 길어 timeout 상향
PAGES = build_pages(config.BLOCKS, table_fmt="text")        # small-to-big 부모(정제 텍스트)


def chat(system, user, mx=256, think=False):
    return _gen.chat(system, user, max_tokens=mx, think=think)


def search(q, k=config.TOP_K):
    return [{"text": p.payload["text"], "pages": p.payload.get("pages")} for p in _retr.hybrid(q, k)]


class State(TypedDict):
    question: str; orig: str; route: str; documents: list; retrieved: list
    generation: str; tries: int; log: list; gen_ctx: str


def n_route(s):   # 2-way: 잡담(direct) / 검색(retrieve)
    r = chat("질문을 분류: 보험약관 내용을 찾아야 하면 retrieve, 단순 인사·잡담·일반상식이면 direct. 한 단어만.", s["orig"], 4).lower()
    dec = "direct" if r.startswith("d") else "retrieve"
    return {"route": dec, "log": s.get("log", []) + [f"route → {dec}"]}


def n_direct(s):
    a = chat("보험약관 상담 어시스턴트. 인사·잡담엔 짧게 응대하고 약관 관련 질문을 하도록 안내. 1~2문장.", s["orig"], 120)
    return {"generation": a, "log": s["log"] + ["direct 응답(검색 생략)"]}


def n_retrieve(s):
    d = search(s["question"], config.TOP_K)
    return {"documents": d, "retrieved": d, "log": s["log"] + [f"retrieve → {len(d)}건(작은청크)"]}


def n_grade(s):
    kept = [d for d in s["documents"]
            if chat("문서가 질문 주제에 관련 있으면 yes 아니면 no. 한 단어.", f"[질문]{s['orig']}\n[문서]{d['text'][:400]}", 4).lower().startswith("y")]
    return {"documents": kept, "log": s["log"] + [f"CRAG → 관련 {len(kept)}/{len(s['documents'])}"]}


def n_transform(s):
    nq = chat("검색이 잘 되도록 질문을 약관 전문용어로 다르게 1문장 재작성. 질문만.", s["orig"] + f" (재시도{s.get('tries', 0) + 1})", 48)
    return {"question": nq, "tries": s.get("tries", 0) + 1, "log": s["log"] + [f"쿼리재작성 → '{nq[:38]}'"]}


def n_generate(s):
    docs = s["documents"] or s.get("retrieved") or []   # CRAG 폴백: 전부 걸러도 원검색 사용(no-CRAG보다 나빠지지 않게)
    pages = sorted({p for d in docs for p in (d["pages"] or [])})
    ctx = "\n\n".join(f"[p{n}]\n{PAGES.get(n, '')[:3000]}" for n in pages) or "(문맥없음)"  # small-to-big 부모
    a = chat("약관 문맥에만 근거해 답하고 근거를 [pN]로. 계산이 필요하면 단계적으로 계산해 최종 수치를 명시. 면책(보상안함)/보상 명확히. 없으면 '약관에서 확인 불가'.",
             f"[문맥]\n{ctx}\n\n[질문]{s['orig']}", 1536, think=True)   # 생성은 thinking ON
    return {"generation": a, "gen_ctx": ctx, "log": s["log"] + [f"generate(부모 p{pages}, thinking) → {a[:40]}..."]}


def e_route(s): return s["route"]
def e_grade(s): return "generate" if s["documents"] else ("transform" if s.get("tries", 0) < 2 else "generate")
def e_selfrag(s):
    if not s["documents"]:
        return "useful"
    ctx = (s.get("gen_ctx") or "")[:4000]   # 생성과 동일 문맥(부모 페이지)으로 근거검증
    g = chat("답변이 문맥에 근거하면 yes 아니면 no. 한 단어.", f"[문맥]{ctx}\n[답변]{s['generation']}", 4).lower()
    return "regen" if (not g.startswith("y") and s.get("tries", 0) < 2) else "useful"


g = StateGraph(State)
for n, f in [("route", n_route), ("direct", n_direct), ("retrieve", n_retrieve), ("grade", n_grade), ("transform", n_transform), ("generate", n_generate)]:
    g.add_node(n, f)
g.add_edge(START, "route")
g.add_conditional_edges("route", e_route, {"direct": "direct", "retrieve": "retrieve"})
g.add_edge("direct", END)
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", e_grade, {"generate": "generate", "transform": "transform"})
g.add_edge("transform", "retrieve")
g.add_conditional_edges("generate", e_selfrag, {"regen": "transform", "useful": END})
app = g.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    for i, q in enumerate(["안녕하세요, 오늘 날씨가 참 좋네요", "비만으로 입원하면 보상되나?", "표준형 입원에서 자기부담 20%가 연간 250만원인 경우 실제 본인부담액은?"]):
        print(f"\n===== Q: {q}")
        out = app.invoke({"question": q, "orig": q, "tries": 0, "log": []}, config={"configurable": {"thread_id": f"t{i}"}})
        for l in out["log"]:
            print("  ·", l)
        print("  ▶", out["generation"][:200])
    print("\nGRAPH_DONE")
