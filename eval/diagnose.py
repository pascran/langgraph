#!/usr/bin/env python3
"""★실패유형 분류기 (taxonomy). 오답 필드마다 '왜 틀렸나'를 결정론 규칙으로 태깅.

  missing        : 정답 값 있는데 비움 (주의·프롬프트)
  hallucinated   : 정답 null인데 채움 / 원문에 없는 값 (모델 과생성)
  wrong_value    : 원문엔 있으나 잘못 추출 (모델 추출오류)
  format         : 값은 맞는데 형식만 틀림 (스키마/구조화 출력)
  ambiguous      : 원문 자체가 모호 (데이터 품질 — 기대된 어려움)
  rule_violation : 판정이 업무규칙(measured vs spec)에 위배 (업무기준)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _norm import (  # noqa: E402
    is_missing, canonical_equal, value_equal, value_in_doc, norm_text, derive_judgment,
)

TAXONOMY = ["missing", "hallucinated", "wrong_value", "format", "ambiguous", "rule_violation"]


def classify_field(fspec, pred_ex, gold, meta, doc):
    ftype, values, rule = fspec["type"], fspec.get("values"), fspec.get("rule")
    p, g = pred_ex.get(fspec["_name"]), gold.get(fspec["_name"])
    if is_missing(g):
        return None if is_missing(p) else "hallucinated"
    if is_missing(p):
        return "missing"
    if canonical_equal(p, g, ftype, values):
        return None
    # 오답 & 둘 다 값 있음
    if meta == "ambiguous":
        return "ambiguous"
    if rule == "judgment_vs_spec":
        derived = derive_judgment(gold.get("measured"), gold.get("spec"))
        if derived and norm_text(p) != norm_text(derived):
            return "rule_violation"
    if ftype in ("date", "number", "measure"):
        return "format" if value_equal(p, g, ftype) else "wrong_value"
    return "wrong_value" if value_in_doc(p, doc) else "hallucinated"


def diagnose_all(golden, predictions):
    fields = {n: {**s, "_name": n} for n, s in golden["fields"].items()}
    by_id = {p["id"]: p.get("extracted", {}) for p in predictions}
    findings, dist = [], {t: 0 for t in TAXONOMY}
    for it in golden["items"]:
        ex, doc, meta = by_id.get(it["id"], {}), it["document"], it.get("gold_meta", {})
        for fname, fspec in fields.items():
            tag = classify_field(fspec, ex, it["gold"], meta.get(fname), doc)
            if tag:
                findings.append({"id": it["id"], "field": fname, "tag": tag,
                                 "pred": ex.get(fname), "gold": it["gold"].get(fname)})
                dist[tag] += 1
    total = sum(dist.values())
    pct = {t: (round(100 * dist[t] / total, 1) if total else 0.0) for t in TAXONOMY}
    return {"findings": findings, "distribution": dist, "pct": pct, "total_failures": total}


if __name__ == "__main__":
    g = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    p = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    out = diagnose_all(g, p)
    print(json.dumps({"distribution": out["distribution"], "pct": out["pct"],
                      "total_failures": out["total_failures"]}, ensure_ascii=False, indent=2))
