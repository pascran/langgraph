#!/usr/bin/env python3
"""results.md 생성: 지표 + ★실패유형 분포(헤드라인) + 문서별 표 + 정직성.

  python -m eval.report --domain qc_report --pred eval/predictions_qc_report.json --model qwen3-30b-a3b
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S      # noqa: E402
import diagnose as D   # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

HINT = {
    "missing": "프롬프트에서 해당 필드 추출 강조",
    "hallucinated": "null 허용 강제 + 근거(원문 span) 요구",
    "wrong_value": "few-shot·컨텍스트 개선",
    "format": "출력 스키마/형식(YYYY-MM-DD 등) 지시 강화",
    "ambiguous": "원문 자체가 모호 — 데이터 품질 이슈(모델 탓 아님)",
    "rule_violation": "★업무규칙(판정=측정값 vs 규격) 후처리로 강제",
}


def _failed_fields(row, findings, doc_id):
    tags = {f["field"]: f["tag"] for f in findings if f["id"] == doc_id}
    return ", ".join(f"{k}({v})" for k, v in tags.items()) or "-"


def build_report(golden, predictions, model="?"):
    sc = S.score_all(golden, predictions)
    dg = D.diagnose_all(golden, predictions)
    m, total = sc["metrics"], dg["total_failures"]
    L = [f"# 평가 리포트 — {golden.get('domain')} 추출 하네스", ""]
    L.append(f"- 도메인 `{golden.get('domain')}` · 골든 {len(golden['items'])}건 · 모델 `{model}`")
    L.append(f"- 채점: **결정론(재현 가능)** · 실패분류: 규칙 기반 taxonomy\n")
    L.append("## 집계")
    L.append(f"- exact-match **{m['exact_match']}** · field accuracy **{m['field_accuracy']}** "
             f"· abstention(정직성) **{m['abstention_accuracy']}**\n")
    L.append("## ★실패유형 분포 — 어디를 고쳐야 하나")
    if total == 0:
        L.append("- 실패 없음\n")
    else:
        L.append(f"- 총 실패 **{total}건**")
        for t in D.TAXONOMY:
            c = dg["distribution"][t]
            if c:
                L.append(f"  - `{t}` {c}건 ({dg['pct'][t]}%) → {HINT[t]}")
        L.append("")
    L.append("## 필드별 정확도")
    L.append("| " + " | ".join(m["per_field"]) + " |")
    L.append("|" + "|".join(["---"] * len(m["per_field"])) + "|")
    L.append("| " + " | ".join(str(v) for v in m["per_field"].values()) + " |\n")
    L.append("## 문서별")
    L.append("| id | category | exact | 실패 필드(유형) |")
    L.append("|---|---|---|---|")
    for r in sc["rows"]:
        L.append(f"| {r['id']} | {r['category']} | {'✅' if r['exact'] else '❌'} "
                 f"| {_failed_fields(r, dg['findings'], r['id'])} |")
    L.append("\n## 한계 (정직성)")
    L.append(f"- 합성·공개 스타일 데이터 {len(golden['items'])}건, 1인 라벨 → 정밀 벤치 아닌 방향성.")
    L.append("- 채점·실패분류는 규칙 기반이라 재현 가능(클론하면 동일). LLM-judge(RAGAS)는 2차·델타만.")
    L.append("- rule_violation은 명시적 규칙(측정값 vs 규격)으로 판정 → 감사 가능.")
    return "\n".join(L) + "\n"


def compare_report(golden, named_preds):
    """A/B: {name: predictions} → 지표·실패유형 델타 표."""
    rows = {}
    for name, preds in named_preds.items():
        m = S.score_all(golden, preds)["metrics"]
        d = D.diagnose_all(golden, preds)
        rows[name] = {"exact": m["exact_match"], "field": m["field_accuracy"],
                      "fail": d["total_failures"], "dist": d["distribution"]}
    L = [f"# A/B 비교 — {golden.get('domain')}", "",
         "| config | exact | field_acc | 실패 | " + " | ".join(D.TAXONOMY) + " |",
         "|---|---|---|---|" + "|".join(["---"] * len(D.TAXONOMY)) + "|"]
    for name, r in rows.items():
        dist = " | ".join(str(r["dist"][t]) for t in D.TAXONOMY)
        L.append(f"| {name} | {r['exact']} | {r['field']} | {r['fail']} | {dist} |")
    best = max(rows, key=lambda n: (rows[n]["field"], -rows[n]["fail"]))
    L.append(f"\n**승자: `{best}`** (field accuracy 최고, 실패 최소).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--model", default="?")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    golden = json.loads((EVAL_DIR / "golden" / f"{a.domain}.json").read_text(encoding="utf-8"))
    preds = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    md = build_report(golden, preds, a.model)
    out = Path(a.out or EVAL_DIR / f"results_{a.domain}.md")
    out.write_text(md, encoding="utf-8")
    print(f"→ {out}")
    print(md)
