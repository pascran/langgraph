"""표(HTML) 변환 유틸 — 여러 스크립트에 흩어져 있던 html2md/strip을 단일화."""
import re


def strip_html(h: str) -> str:
    """모든 태그 제거 후 공백 정리."""
    return re.sub(r"<[^>]+>", " ", h).strip()


def html2md(h: str) -> str:
    """HTML <table> → markdown 표. 표가 아니거나 1행 이하면 태그만 제거해 반환."""
    rows = [
        [strip_html(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S | re.I)
    ]
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return strip_html(h)
    out = []
    for i, r in enumerate(rows):
        out.append("| " + " | ".join(r) + " |")
        if i == 0:
            out.append("|" + "|".join(["---"] * len(r)) + "|")
    return "\n".join(out)
