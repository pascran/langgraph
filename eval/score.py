#!/usr/bin/env python3
"""필드별 결정론 채점 (LLM 불필요). golden + predictions → 지표.

golden: {fields:{name:{type,...}}, required:[...], items:[{id,category,document,gold,gold_meta}]}
predictions: [{id, category, extracted:{...}}]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _norm import is_missing, canonical_equal  # noqa: E402


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def score_item(pred_ex, gold, fields):
    res = {}
    for fname, fspec in fields.items():
        p, g = pred_ex.get(fname), gold.get(fname)
        gm = is_missing(g)
        if gm:
            correct = is_missing(p)                       # 정답=null → 올바른 abstention
        elif is_missing(p):
            correct = False
        else:
            correct = canonical_equal(p, g, fspec["type"], fspec.get("values"))
        res[fname] = {"correct": correct, "pred": p, "gold": g, "gold_missing": gm}
    return res


def aggregate(rows, fields, required):
    per_field = {
        f: (sum(r["fields"][f]["correct"] for r in rows) / len(rows) if rows else 0.0)
        for f in fields
    }
    exact = sum(r["exact"] for r in rows) / len(rows) if rows else 0.0
    cells = [r["fields"][f]["correct"] for r in rows for f in fields]
    overall = sum(cells) / len(cells) if cells else 0.0
    # 정직성 지표: 결함 필드에서 올바로 null 냈나(abstention) / 채웠나(환각)
    null_cells = [(r, f) for r in rows for f in fields if r["fields"][f]["gold_missing"]]
    abst = (sum(r["fields"][f]["correct"] for r, f in null_cells) / len(null_cells)
            if null_cells else None)
    return {
        "exact_match": round(exact, 4),
        "field_accuracy": round(overall, 4),
        "abstention_accuracy": round(abst, 4) if abst is not None else None,
        "per_field": {f: round(v, 4) for f, v in per_field.items()},
    }


def score_all(golden, predictions):
    fields = golden["fields"]
    required = golden.get("required", list(fields))
    by_id = {p["id"]: p.get("extracted", {}) for p in predictions}
    rows = []
    for it in golden["items"]:
        fr = score_item(by_id.get(it["id"], {}), it["gold"], fields)
        rows.append({
            "id": it["id"], "category": it["category"], "fields": fr,
            "exact": all(fr[f]["correct"] for f in required),
        })
    return {"fields": fields, "required": required, "rows": rows,
            "metrics": aggregate(rows, fields, required)}


if __name__ == "__main__":
    result = score_all(load(sys.argv[1]), load(sys.argv[2]))
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
