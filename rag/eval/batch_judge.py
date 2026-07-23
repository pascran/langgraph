"""Batch 2 (판정): 저장된 8B·27B 답변을 32B 심판으로 F1 채점 → 8B vs 27B 델타. core 라이브러리 사용."""
import json
from concurrent.futures import ThreadPoolExecutor
from rag.core import config
from rag.core.llm import LLMClient
from rag.core.metrics import answer_f1

CRIT = LLMClient(config.JUDGE_URL, config.JUDGE_MODEL, timeout=90, max_retries=2)
D = json.load(open(f"{config.DATA}/ragas/stack32_answers.json", encoding="utf-8"))
A27 = {r["q"]: r["a27"] for r in json.load(open(f"{config.DATA}/ragas/answers_27b.json", encoding="utf-8")) if r.get("a27")}
sub = [it for it in D if it["q"] in A27]  # 27B 실답 있는 문항만 공정비교
print(f"공정비교 대상: {len(sub)}/{len(D)} 문항", flush=True)


def one(it):
    gt = it["gt"]
    return {"type": it.get("type"), "q": it["q"],
            "f8": answer_f1(it["L2c"][0], gt, CRIT.judge),
            "f27": answer_f1(A27[it["q"]], gt, CRIT.judge)}


with ThreadPoolExecutor(max_workers=4) as ex:
    R = list(ex.map(one, sub))
json.dump(R, open(f"{config.DATA}/ragas/proof_scores.json", "w", encoding="utf-8"), ensure_ascii=False)


def mn(k, f=lambda r: True):
    xs = [r[k] for r in R if f(r) and r[k] == r[k]]
    return round(sum(xs) / len(xs), 3) if xs else None


tab = lambda r: r["type"] == "table"
print("=== 최종증명(core): 8B vs 27B thinking (32B F1 심판) ===")
print(f"8B  F1: 전체={mn('f8')}  표={mn('f8', tab)}")
print(f"27B F1: 전체={mn('f27')}  표={mn('f27', tab)}")
print("BATCH_JUDGE_DONE")
