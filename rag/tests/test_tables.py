from rag.core.tables import html2md, strip_html


def test_strip_html_removes_tags():
    assert strip_html("<p>hi <b>x</b></p>") == "hi  x"


def test_html2md_multirow():
    h = "<table><tr><td>구분</td><td>값</td></tr><tr><td>의원</td><td>1만원</td></tr></table>"
    md = html2md(h)
    assert "| 구분 | 값 |" in md
    assert "|---|---|" in md
    assert "| 의원 | 1만원 |" in md


def test_html2md_single_row_falls_back_to_text():
    assert html2md("<p>그냥 문장</p>") == "그냥 문장"


def test_html2md_strips_inner_cell_tags():
    h = "<table><tr><td><b>가</b></td><td>나</td></tr><tr><td>1</td><td>2</td></tr></table>"
    md = html2md(h)
    assert "| 가 | 나 |" in md and "<b>" not in md
