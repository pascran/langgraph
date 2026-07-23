"""small-to-big 부모 페이지 빌더 — 8곳에 복붙되던 PMD/PAGES 구성 단일화.

파싱블록(blocks_kie.jsonl)에서 페이지별 정제 텍스트를 만든다. 표 표현은 md/html/text 선택.
"""
import json
from collections import defaultdict
from rag.core.tables import html2md, strip_html
from rag.core import config


def build_pages(blocks_path=config.BLOCKS, table_fmt="text"):
    """table_fmt: 'text'(태그제거) | 'md'(markdown표) | 'html'(원형)."""
    bl = [json.loads(l) for l in open(blocks_path, encoding="utf-8")]
    pg = defaultdict(list)
    for b in sorted(bl, key=lambda b: (b["page"], b["order"])):
        if b["type"] in ("header", "page_number"):
            continue
        if b["type"] == "table":
            if table_fmt == "md":
                t = "[표]\n" + html2md(b["content"])
            elif table_fmt == "html":
                t = "[표]\n" + b["content"].strip()
            else:
                t = strip_html(b["content"])
            pg[b["page"]].append(t)
        else:
            pg[b["page"]].append(strip_html(b["content"]))
    return {p: "\n".join(v) for p, v in pg.items()}


def parent_context(pages, page_map, max_chars=3000):
    """검색된 페이지 목록 → '[pN]\\n본문' 컨텍스트 리스트."""
    return [f"[p{n}]\n" + page_map.get(n, "")[:max_chars] for n in pages] or ["(없음)"]
