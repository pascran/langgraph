#!/usr/bin/env python3
"""(domain, config) → 에이전트를 골든셋에 돌리되, 매 문서를 Langfuse로 트레이싱.
   + 하네스의 결정론 채점(score)·실패진단(diagnose) 결과를 트레이스에 score로 첨부.

run.py와 하는 일은 같지만(문서→추출), 여기서는:
  1) 문서 1건 = 트레이스 1개. LangGraph 내부 노드(분류·추출·검증·재시도)가 span으로 중첩 기록.
  2) 추출 직후 그 문서 하나를 채점/진단해서, 트레이스에 score를 붙인다:
       exact_match(BOOLEAN) · field_accuracy(NUMERIC) · abstention(NUMERIC) · retries(NUMERIC)
       + 틀린 필드마다  fail:<field> = <taxonomy-tag>(CATEGORICAL, pred/gold 코멘트)
  → Langfuse UI에서 "어느 문서가, 어느 노드에서, 왜 틀렸나"를 call 단위로 관찰.

MLflow(track.py)는 run 단위 A/B(설정 통째 비교), Langfuse는 call 단위 추적(문서·노드별). 상보적.

전제:
  - Langfuse self-host 기동(localhost:3000) + 아래 env(없으면 로컬 데모 기본값 사용).
  - LLM 백엔드(Ollama/vLLM/API)는 config yaml로 주입.

  export LANGFUSE_HOST=http://localhost:3000
  export LANGFUSE_PUBLIC_KEY=...   LANGFUSE_SECRET_KEY=...
  python -m eval.run_traced --domain receipt --config eval/configs/o_ollama_qwen25.yaml
  python -m eval.run_traced --domain qc_report --config ... --limit 3
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

# 로컬 self-host 데모 기본값(전부 0인 headless-init 플레이스홀더 = 실제 비밀 아님).
# 실제 배포에서는 env로 덮어쓴다.
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:3000")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-0000000000000000000000000000")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-0000000000000000000000000000")


def score_one(golden, it, ex):
    """문서 1건을 채점·진단해서 (지표, row, findings) 반환."""
    from score import score_all
    from diagnose import diagnose_all
    one_g = {
        "fields": golden["fields"],
        "required": golden.get("required", list(golden["fields"])),
        "items": [it],
    }
    one_p = [{"id": it["id"], "category": it["category"], "extracted": ex}]
    sc = score_all(one_g, one_p)
    dg = diagnose_all(one_g, one_p)
    return sc["metrics"], sc["rows"][0], dg["findings"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="트레이싱할 문서 수 상한(CPU 추론시 유용)")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) if a.config else {}
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        if cfg.get(k):
            os.environ[k] = str(cfg[k])
    cfgname = Path(a.config).stem if a.config else "default"
    model = os.environ.get("LLM_MODEL", "?")

    # env 설정 후 import (agent가 로드 시 LLM 구성)
    dom = importlib.import_module(f"domains.{a.domain}").DOMAIN
    from agent import build_app
    app = build_app(dom, max_retries=int(cfg.get("max_retries", 2)))

    from langfuse import get_client
    from langfuse.langchain import CallbackHandler
    lf = get_client()
    if not lf.auth_check():
        print("✗ Langfuse 인증 실패 — HOST/키 확인:", os.environ.get("LANGFUSE_HOST"))
        sys.exit(1)
    handler = CallbackHandler()

    golden = json.loads((ROOT / "eval" / "golden" / f"{a.domain}.json").read_text(encoding="utf-8"))
    items = golden["items"][: a.limit] if a.limit else golden["items"]
    print(f"▶ domain={a.domain} model={model} config={cfgname} docs={len(items)} "
          f"→ Langfuse {os.environ.get('LANGFUSE_HOST')}")

    preds, first_url = [], None
    for it in items:
        did = it["id"]
        # 문서 1건 = 트레이스 1개. LangGraph 내부는 이 span 밑에 중첩된다.
        with lf.start_as_current_observation(
            as_type="chain",
            name=f"{a.domain}:{did}",
            input={"category": it["category"], "document": it["document"]},
            metadata={"domain": a.domain, "model": model, "config": cfgname, "doc_id": did},
        ) as span:
            res = app.invoke(
                {"text": it["document"], "retries": 0},
                config={"callbacks": [handler],
                        "metadata": {"doc_id": did, "domain": a.domain}},
            )
            ex = res.get("extracted", {})
            retries = int(res.get("retries", 0))
            preds.append({"id": did, "category": it["category"], "extracted": ex})

            m, row, findings = score_one(golden, it, ex)
            passed = bool(row["exact"])

            span.update(output=ex)  # 루트 observation 출력 → 트레이스 출력으로 상속
            # ── 하네스 채점 결과를 트레이스 score로 첨부 ──
            #    (pass/fail 필터링은 exact_match score로 대체 — v4엔 trace tags API 부재)
            lf.score_current_trace(name="exact_match", value=1.0 if passed else 0.0,
                                   data_type="BOOLEAN")
            lf.score_current_trace(name="field_accuracy", value=m["field_accuracy"],
                                   data_type="NUMERIC")
            if m["abstention_accuracy"] is not None:
                lf.score_current_trace(name="abstention_accuracy",
                                       value=m["abstention_accuracy"], data_type="NUMERIC")
            lf.score_current_trace(name="retries", value=float(retries), data_type="NUMERIC")
            for f in findings:
                lf.score_current_trace(
                    name=f"fail:{f['field']}", value=f["tag"], data_type="CATEGORICAL",
                    comment=f"pred={f['pred']!r} gold={f['gold']!r}",
                )
            first_url = first_url or lf.get_trace_url()

        flag = "✓" if passed else "✗"
        tags = ",".join(f"{f['field']}={f['tag']}" for f in findings) or "-"
        print(f"  {flag} [{did}] field_acc={m['field_accuracy']:.2f} retries={retries} fails: {tags}")

    lf.flush()

    # 전체 집계(참고용) — MLflow track.py와 동일 지표
    from score import score_all
    from diagnose import diagnose_all
    gm = score_all(golden if not a.limit else {**golden, "items": items}, preds)["metrics"]
    dd = diagnose_all(golden if not a.limit else {**golden, "items": items}, preds)
    print(f"\n집계: exact={gm['exact_match']} field_acc={gm['field_accuracy']} "
          f"abstention={gm['abstention_accuracy']} 실패분포={dd['distribution']}")
    if first_url:
        print(f"\nLangfuse UI: {os.environ.get('LANGFUSE_HOST')}  (첫 트레이스: {first_url})")


if __name__ == "__main__":
    main()
