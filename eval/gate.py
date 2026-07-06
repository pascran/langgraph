#!/usr/bin/env python3
"""회귀 게이트: 현재 지표 vs baseline. 하락/실패유형 급증 시 exit 1 (CI 차단).

  python -m eval.gate --domain qc_report --pred eval/predictions_qc_report.json
  python -m eval.gate --domain qc_report --pred ... --update    # baseline 갱신
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S      # noqa: E402
import diagnose as D   # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TOL = 0.02


def current(golden, preds):
    m = S.score_all(golden, preds)["metrics"]
    d = D.diagnose_all(golden, preds)
    return {"exact_match": m["exact_match"], "field_accuracy": m["field_accuracy"],
            "distribution": d["distribution"]}


def check(cur, base):
    fails = []
    for k in ("field_accuracy", "exact_match"):
        if cur[k] < base[k] - TOL:
            fails.append(f"{k} 하락 {base[k]}→{cur[k]}")
    for t, c in cur["distribution"].items():
        b = base["distribution"].get(t, 0)
        if c > b:
            fails.append(f"{t} 실패 급증 {b}→{c}")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    golden = json.loads((EVAL_DIR / "golden" / f"{a.domain}.json").read_text(encoding="utf-8"))
    preds = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    cur = current(golden, preds)
    bpath = EVAL_DIR / "baselines" / f"{a.domain}.baseline.json"

    if a.update or not bpath.exists():
        bpath.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"baseline {'갱신' if a.update else '생성'} → {bpath}")
        sys.exit(0)

    base = json.loads(bpath.read_text(encoding="utf-8"))
    fails = check(cur, base)
    if fails:
        print("❌ REGRESSION:")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print(f"✅ OK (field_acc {cur['field_accuracy']} ≥ {base['field_accuracy']}-{TOL})")
    sys.exit(0)
