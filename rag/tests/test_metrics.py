import math
from rag.core.metrics import jparse, answer_f1, faithfulness, context_recall


def test_jparse_valid():
    assert jparse('{"tp":2,"fp":1}') == {"tp": 2, "fp": 1}


def test_jparse_single_quote_fallback():
    assert jparse("{'tp': 2}") == {"tp": 2}


def test_jparse_embedded_in_text():
    assert jparse('결과: {"total":3} 끝') == {"total": 3}


def test_jparse_garbage_and_empty():
    assert jparse("no json") is None
    assert jparse("") is None


def test_answer_f1_arithmetic():
    judge = lambda p: '{"tp":2,"fp":1,"fn":1}'  # F1 = 2/(2+0.5*2) = 2/3
    assert abs(answer_f1("a", "b", judge) - 2 / 3) < 1e-9


def test_answer_f1_perfect_and_zero():
    assert answer_f1("a", "b", lambda p: '{"tp":3,"fp":0,"fn":0}') == 1.0
    assert answer_f1("a", "b", lambda p: '{"tp":0,"fp":0,"fn":0}') == 0.0


def test_faithfulness_clamps_over_one():
    # supported > total → 1.67 → 클램프 1.0 (실제로 있었던 잠복 버그 방지)
    assert faithfulness("a", ["ctx"], lambda p: '{"supported":5,"total":3}') == 1.0


def test_faithfulness_normal_ratio():
    assert faithfulness("a", ["ctx"], lambda p: '{"supported":2,"total":4}') == 0.5


def test_faithfulness_nan_on_bad_json():
    assert math.isnan(faithfulness("a", ["ctx"], lambda p: "garbage"))


def test_context_recall_clamps():
    assert context_recall("gt", ["ctx"], lambda p: '{"attributable":4,"total":2}') == 1.0
