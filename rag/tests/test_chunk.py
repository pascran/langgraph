import numpy as np
from rag.core.chunk import split_sentences, build_chunks


def blk(page, order, typ, content):
    return {"page": page, "order": order, "type": typ, "content": content, "meta": {}}


def test_split_marks_tables():
    s = split_sentences([blk(1, 0, "table", "<table><tr><td>x</td></tr></table>")])
    assert s[0][0] == "__T__"


def test_split_drops_short_sentences():
    s = split_sentences([blk(1, 0, "text", "짧다. 이것은 충분히 길어서 남는 문장이다.")], sent_min=5)
    assert s and all(len(t) >= 5 for t, _ in s)


def test_split_skips_headers():
    s = split_sentences([blk(1, 0, "header", "머리글 텍스트입니다"), blk(1, 1, "page_number", "21")])
    assert s == []


def test_table_yields_own_chunk():
    long = "이 문장은 삼십자 이상으로 충분히 길게 만든 텍스트 문장입니다"
    sents = [(long, blk(1, 0, "text", "")),
             ("__T__", blk(2, 1, "table", "<table><tr><td>표내용</td></tr></table>"))]
    chunks = build_chunks(sents, np.array([[1.0, 0.0]]))
    btypes = {c["meta"]["btype"] for c in chunks}
    assert "table" in btypes and "text" in btypes
    tbl = next(c for c in chunks if c["meta"]["btype"] == "table")
    assert "표내용" in tbl["text"]


def test_short_trailing_fragment_not_lost():
    long = "이 문장은 삼십자 이상으로 충분히 길게 만든 첫 텍스트 문장입니다"
    short = "짧은꼬리"
    b = blk(1, 0, "text", "")
    chunks = build_chunks([(long, b), (short, b)], np.array([[1.0, 0.0], [1.0, 0.0]]))
    joined = " ".join(c["text"] for c in chunks)
    assert "짧은꼬리" in joined  # 짧은 조각이 병합되어 유실되지 않음
