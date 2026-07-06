#!/usr/bin/env python3
"""LLM·외부 의존 없는 유닛테스트.  실행:  python3 tests/test_eval.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from _norm import (is_missing, canonical_equal, value_equal,  # noqa: E402
                   derive_judgment, canon_date, parse_number, set_f1)
import score as S      # noqa: E402
import diagnose as D   # noqa: E402


def check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok: {name}")


GOLDEN = {
    "fields": {
        "part_no": {"type": "text"},
        "inspect_date": {"type": "date"},
        "measured": {"type": "measure"},
        "spec": {"type": "text"},
        "judgment": {"type": "enum", "values": ["합격", "불합격"], "rule": "judgment_vs_spec"},
        "lot_no": {"type": "text"},
    },
    "required": ["part_no", "judgment"],
    "items": [{
        "id": "t1", "category": "합성",
        "document": "품번 MX-1201 로트 L1 측정 10.05 mm 규격 10.00±0.20 mm 검사 2026-07-01",
        "gold": {"part_no": "MX-1201", "inspect_date": "2026-07-01", "measured": "10.05 mm",
                 "spec": "10.00±0.20 mm", "judgment": "합격", "lot_no": "L1"},
        "gold_meta": {},
    }],
}
BASE = dict(GOLDEN["items"][0]["gold"])


def preds(ex):
    return [{"id": "t1", "category": "합성", "extracted": ex}]


def test_norm():
    check("is_missing None", is_missing(None))
    check("is_missing placeholder", is_missing("문서에 명시되지 않음"))
    check("not missing", not is_missing("MX-1201"))
    check("canon_date dot", canon_date("2026.06.28") == "2026-06-28")
    check("date strict ne", not canonical_equal("2026.06.28", "2026-06-28", "date"))
    check("date strict eq", canonical_equal("2026-06-28", "2026-06-28", "date"))
    check("value_equal date fmt", value_equal("2026.06.28", "2026-06-28", "date"))
    check("number comma", parse_number("24,100원") == 24100)
    check("set_f1 unordered", set_f1(["a", "b"], ["b", "a"]) > 0.99)


def test_rule():
    check("판정 합격(±)", derive_judgment("10.05 mm", "10.00±0.20 mm") == "합격")
    check("판정 불합격(±)", derive_judgment("3.12 mm", "3.00±0.05 mm") == "불합격")
    check("판정 범위(~)", derive_judgment("9.9", "9.80~10.20 mm") == "합격")
    check("판정 이하", derive_judgment("0.3", "≤ 0.5 mm") == "합격")
    check("판정 이상 불합격", derive_judgment("90", "≥ 100 MPa") == "불합격")


def test_taxonomy():
    d = D.diagnose_all(GOLDEN, preds({**BASE, "inspect_date": "2026.07.01"}))
    check("format", d["distribution"]["format"] == 1 and d["total_failures"] == 1)
    d = D.diagnose_all(GOLDEN, preds({**BASE, "lot_no": None}))
    check("missing", d["distribution"]["missing"] == 1)
    d = D.diagnose_all(GOLDEN, preds({**BASE, "part_no": "MX-9999"}))
    check("hallucinated (not in doc)", d["distribution"]["hallucinated"] == 1)
    d = D.diagnose_all(GOLDEN, preds({**BASE, "judgment": "불합격"}))
    check("rule_violation", d["distribution"]["rule_violation"] == 1)
    d = D.diagnose_all(GOLDEN, preds(dict(BASE)))
    check("no failure on perfect", d["total_failures"] == 0)


def test_score():
    out = S.score_all(GOLDEN, preds(dict(BASE)))
    check("exact all correct", out["metrics"]["exact_match"] == 1.0)
    out = S.score_all(GOLDEN, preds({**BASE, "judgment": "불합격"}))
    check("exact fails on judgment", out["metrics"]["exact_match"] == 0.0)


if __name__ == "__main__":
    for t in [test_norm, test_rule, test_taxonomy, test_score]:
        print(f"[{t.__name__}]")
        t()
    print("\nALL PASSED ✅")
