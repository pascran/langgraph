"""코사인급락 청킹의 순수 로직 — chunk_fix.py에서 추출(동작 동일). 임베딩은 주입받아 테스트 가능.

split_sentences: 블록 → (문장, 블록) 스트림(표는 ('__T__', 블록)).
build_chunks: 문장 스트림 + 문장 dense벡터 → 청크 리스트. 표는 원형 1청크, 짧은 조각은 앞 청크에 병합.
"""
import re
from rag.core import config
from rag.core.tables import strip_html


def split_sentences(blocks, sent_min=config.SENT_MIN):
    blocks = [b for b in blocks if b["type"] not in ("header", "page_number")]
    blocks.sort(key=lambda b: (b["page"], b["order"]))
    sents = []
    for b in blocks:
        if b["type"] == "table":
            sents.append(("__T__", b))
            continue
        for s in re.split(r"(?<=[.다])\s+", strip_html(b["content"])):
            s = s.strip()
            if len(s) >= sent_min:
                sents.append((s, b))
    return sents


def build_chunks(sents, dense_vecs, pctl=config.CHUNK_PCTL, min_split=config.CHUNK_MIN_SPLIT,
                 max_len=config.CHUNK_MAX, merge_min=config.CHUNK_MERGE_MIN):
    import numpy as np
    E = dense_vecs
    d = [1 - float(np.dot(E[i], E[i + 1])) for i in range(len(E) - 1)]
    thr = float(np.percentile(d, pctl)) if d else 1.0
    chunks, cur, ei, prev = [], [], 0, None

    def flush():
        nonlocal cur
        if cur:
            txt = " ".join(x[0] for x in cur).strip()
            pages = sorted(set(x[1]["page"] for x in cur))
            if len(txt) >= merge_min:
                chunks.append({"text": txt, "meta": {**cur[-1][1].get("meta", {}), "pages": pages, "btype": "text"}})
            elif chunks and chunks[-1]["meta"].get("btype") == "text":  # 짧은 조각 앞 청크에 병합
                chunks[-1]["text"] = (chunks[-1]["text"] + " " + txt).strip()
                chunks[-1]["meta"]["pages"] = sorted(set(chunks[-1]["meta"]["pages"] + pages))
        cur = []

    for s, b in sents:
        if s == "__T__":
            flush(); prev = None
            # 표 텍스트는 chunk_fix와 동일하게 태그만 치환(양끝 strip 안 함) — 임베딩 동일성 보존
            tbl = re.sub(r"<[^>]+>", " ", b["content"])
            chunks.append({"text": tbl, "meta": {**b.get("meta", {}), "pages": [b["page"]], "btype": "table"}})
            continue
        v = E[ei]; ei += 1
        if cur and prev is not None and (1 - float(np.dot(v, prev))) > thr and sum(len(x[0]) for x in cur) > min_split:
            flush()
        cur.append((s, b)); prev = v
        if sum(len(x[0]) for x in cur) > max_len:
            flush(); prev = None
    flush()
    return chunks
