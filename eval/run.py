#!/usr/bin/env python3
"""(domain, config) → 에이전트를 골든셋에 돌려 predictions 생성. ★LLM 필요.

  python -m eval.run --domain qc_report --config eval/configs/baseline.yaml
  python -m eval.run --domain receipt   --config eval/configs/no_retry.yaml
config(yaml): LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, max_retries
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) if a.config else {}
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        if cfg.get(k):
            os.environ[k] = str(cfg[k])

    # env 설정 후 import (agent가 모듈 로드 시 LLM 구성)
    dom = importlib.import_module(f"domains.{a.domain}").DOMAIN
    from agent import build_app
    app = build_app(dom, max_retries=int(cfg.get("max_retries", 2)))

    golden = json.loads((ROOT / "eval" / "golden" / f"{a.domain}.json").read_text(encoding="utf-8"))
    preds = []
    for it in golden["items"]:
        res = app.invoke({"text": it["document"], "retries": 0})
        ex = res.get("extracted", {})
        print(f"[{it['id']}] {ex}")
        preds.append({"id": it["id"], "category": it["category"], "extracted": ex})

    out = Path(a.out or ROOT / "eval" / f"predictions_{a.domain}.json")
    out.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out} ({len(preds)}건)  model={os.environ.get('LLM_MODEL')}")


if __name__ == "__main__":
    main()
