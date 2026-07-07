#!/usr/bin/env python3
"""평가 run을 MLflow에 로깅 — 추출 에이전트의 '학습'이 아니라 '평가 지표'를 추적한다.

LLM 불필요: 기존 predictions(에이전트 실행 결과)로 score/diagnose만 계산해 기록.
config별로 run을 남기면 MLflow UI에서 A/B(baseline vs no_retry vs small_model)가 나란히 비교된다.

  python -m eval.track --domain qc_report --pred eval/predictions_qc_report.json \
      --model qwen3:8b --run-name qc-baseline
  mlflow ui        # → http://localhost:5000 에서 run 비교·지표 추세
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import os               # noqa: E402
import score as S       # noqa: E402
import diagnose as D    # noqa: E402
import report as R      # noqa: E402
import mlflow           # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
# MLflow 3.x는 file store 폐기 → DB 백엔드 사용(sqlite 기본, env로 재정의 가능)
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--model", default="?")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--config", default=None, help="선택: config yaml → params로 로깅")
    ap.add_argument("--experiment", default="langgraph-extraction")
    a = ap.parse_args()

    golden = json.loads((EVAL_DIR / "golden" / f"{a.domain}.json").read_text(encoding="utf-8"))
    preds = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    m = S.score_all(golden, preds)["metrics"]
    dg = D.diagnose_all(golden, preds)

    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=a.run_name or f"{a.domain}-{a.model}"):
        # ── params: 무엇을 평가했나 ──
        mlflow.log_params({
            "domain": a.domain, "model": a.model, "n_docs": len(golden["items"]),
            "required_fields": ",".join(golden.get("required", [])),
        })
        if a.config and Path(a.config).exists():
            import yaml
            for k, v in (yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}).items():
                mlflow.log_param(f"cfg.{k}", v)

        # ── metrics: 얼마나 정확한가 + 어떤 실패인가 ──
        mlflow.log_metric("exact_match", m["exact_match"])
        mlflow.log_metric("field_accuracy", m["field_accuracy"])
        if m["abstention_accuracy"] is not None:
            mlflow.log_metric("abstention_accuracy", m["abstention_accuracy"])
        mlflow.log_metric("total_failures", dg["total_failures"])
        for tag in D.TAXONOMY:                       # 실패유형별 count
            mlflow.log_metric(f"fail.{tag}", dg["distribution"][tag])
        for f, v in m["per_field"].items():          # 필드별 정확도
            mlflow.log_metric(f"field.{f}", v)

        # ── artifacts: 리포트·예측·골든 ──
        rpt = EVAL_DIR / f"results_{a.domain}.md"
        rpt.write_text(R.build_report(golden, preds, a.model), encoding="utf-8")
        mlflow.log_artifact(str(rpt))
        mlflow.log_artifact(a.pred)
        mlflow.log_artifact(str(EVAL_DIR / "golden" / f"{a.domain}.json"))

        print(f"✅ logged run: exact={m['exact_match']} field={m['field_accuracy']} "
              f"abstention={m['abstention_accuracy']} fails={dg['total_failures']}")


if __name__ == "__main__":
    main()
