#!/usr/bin/env python3
"""2단계: predictions.json을 RAGAS로 평가한다 (venv-ragas에서 실행).

추출 에이전트에 맞는 두 지표(둘 다 LLM-judge, 임베딩 불필요):
  - Faithfulness        : 추출한 값들이 원본 문서에 근거하는가 (= 환각 안 했는가)
  - FactualCorrectness  : 추출 결과가 골든 정답과 사실적으로 일치하는가

judge LLM은 thinking이 없는 qwen2.5:7b-instruct (RAGAS의 구조화 파싱 안정성).
에이전트 추출은 qwen3:8b로 별도 수행됨(dump_predictions.py).
"""
import os
import json
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama

EVAL_DIR = Path(__file__).resolve().parent
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE", "qwen2.5:7b-instruct")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")


def fmt(x) -> str:
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def main() -> None:
    preds = json.loads((EVAL_DIR / "predictions.json").read_text(encoding="utf-8"))

    judge = LangchainLLMWrapper(
        ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_BASE, temperature=0)
    )

    samples = [
        SingleTurnSample(
            user_input=p["user_input"],
            retrieved_contexts=p["retrieved_contexts"],
            response=p["response"],
            reference=p["reference"],
        )
        for p in preds
    ]
    dataset = EvaluationDataset(samples=samples)

    metrics = [Faithfulness(llm=judge), FactualCorrectness(llm=judge)]
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge)
    print("\n=== RAGAS 집계 ===")
    print(result)

    df = result.to_pandas()
    cols = [c for c in ("faithfulness", "factual_correctness(mode=f1)",
                        "factual_correctness") if c in df.columns]

    # 마크다운 리포트
    lines = ["# RAGAS 평가 — LangGraph 문서 추출 에이전트\n"]
    lines.append(f"- judge: `{JUDGE_MODEL}` (Ollama) · 추출 에이전트: `qwen3:8b`")
    lines.append(f"- 골든셋: {len(preds)}건 (정상 영수증/세금계산서 + 결함 문서)")
    lines.append("- 지표: **Faithfulness**(추출이 원본 문서에 근거하는가), "
                 "**FactualCorrectness**(골든과 사실 일치)\n")
    lines.append("## 집계 (평균)\n")
    for c in cols:
        lines.append(f"- **{c}**: {fmt(df[c].mean())}")
    lines.append("\n## 문서별\n")
    header = "| id | category | " + " | ".join(cols) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(cols)) + "|"
    lines.append(header)
    lines.append(sep)
    for p, (_, row) in zip(preds, df.iterrows()):
        vals = " | ".join(fmt(row[c]) for c in cols)
        lines.append(f"| {p['id']} | {p['category']} | {vals} |")
    lines.append("\n## 한계 (정직성)\n")
    lines.append("- N=3 합성셋, 작성자 1인 라벨 → 정밀 벤치가 아니라 방향성 sanity check.")
    lines.append("- judge LLM(qwen2.5:7b) 기반이라 절대 점수는 judge 능력에 의존.")
    lines.append("- 추출 에이전트는 RAG가 아니라 단일 문서 추출이므로, "
                 "context_precision/recall(랭킹 지표)은 적용하지 않고 "
                 "근거성(faithfulness)·정답일치(factual correctness)만 측정한다.")

    out = EVAL_DIR / "ragas_results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
