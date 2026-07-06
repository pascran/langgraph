"""순수 stdlib 정규화·파싱 헬퍼 (LLM·pydantic 불필요, 재현 가능)."""
import re
from difflib import SequenceMatcher

# 모델이 null 대신 채워넣는 placeholder(= 사실상 '누락')
PLACEHOLDERS = {
    "", "not found", "n/a", "na", "none", "null", "unknown", "-",
    "없음", "미상", "해당없음", "확인불가", "명시되지 않음", "문서에 명시되지 않음",
}


def is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in PLACEHOLDERS:
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def norm_text(s) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("(주)", "").replace("㈜", "")
    return s.strip()


def fuzzy(a, b) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def parse_date(s):
    """다양한 날짜 표기 → (year, month, day) 또는 None."""
    m = re.search(r"(\d{4})[.\-/년\s]+(\d{1,2})[.\-/월\s]+(\d{1,2})", str(s))
    if m:
        y, mo, d = map(int, m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return (y, mo, d)
    return None


def canon_date(s):
    t = parse_date(s)
    return f"{t[0]:04d}-{t[1]:02d}-{t[2]:02d}" if t else None


def parse_number(s):
    if isinstance(s, (int, float)):
        return float(s)
    m = re.search(r"-?\d[\d,]*\.?\d*", str(s).replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def parse_measure(s):
    """'10.05 mm' → (10.05, 'mm')."""
    if s is None:
        return (None, None)
    num = parse_number(s)
    um = re.search(r"[a-zA-Zμ%℃Ω]+", str(s))
    return (num, um.group().lower() if um else None)


def set_f1(pred, gold, thr: float = 0.8) -> float:
    pred = pred or []
    gold = gold or []
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    matched, used = 0, set()
    for g in gold:
        for i, p in enumerate(pred):
            if i in used:
                continue
            if fuzzy(p, g) >= thr:
                matched += 1
                used.add(i)
                break
    prec, rec = matched / len(pred), matched / len(gold)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def canonical_equal(pred, gold, ftype, values=None) -> bool:
    """엄격 정합(정규 형식 요구): 값+형식이 정답과 같아야 correct."""
    if ftype == "date":
        return str(pred).strip() == str(gold).strip()  # gold는 canonical YYYY-MM-DD
    if ftype == "number":
        return parse_number(pred) == parse_number(gold)
    if ftype == "measure":
        return parse_measure(pred) == parse_measure(gold)
    if ftype == "enum":
        return norm_text(pred) == norm_text(gold)
    if ftype == "list":
        return set_f1(pred, gold) >= 0.999
    return norm_text(pred) == norm_text(gold)


def value_equal(pred, gold, ftype) -> bool:
    """의미 정합(형식 무시): format vs wrong_value 구분용."""
    if ftype == "date":
        cg = canon_date(gold)
        return cg is not None and canon_date(pred) == cg
    if ftype == "number":
        return parse_number(pred) == parse_number(gold)
    if ftype == "measure":
        pv, gv = parse_measure(pred), parse_measure(gold)
        return pv[0] is not None and pv[0] == gv[0]
    return norm_text(pred) == norm_text(gold)


def value_in_doc(v, doc, thr: float = 0.85) -> bool:
    v = norm_text(v)
    if not v:
        return False
    d = norm_text(doc)
    if v in d:
        return True
    return any(fuzzy(v, tok) >= thr for tok in re.split(r"[\s,]+", d) if tok)


def derive_judgment(measured, spec):
    """측정값 vs 규격 → '합격'/'불합격' 또는 None(판정 불가). ★업무규칙."""
    mv = parse_number(measured)
    if mv is None or is_missing(spec):
        return None
    sp = str(spec)
    m = re.search(r"(-?\d[\d.]*)\s*[~∼]\s*(-?\d[\d.]*)", sp)          # a~b 범위
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return "합격" if lo <= mv <= hi else "불합격"
    m = re.search(r"(-?\d[\d.]*)\s*±\s*(\d[\d.]*)", sp)               # center±tol
    if m:
        c, t = float(m.group(1)), float(m.group(2))
        return "합격" if (c - t) <= mv <= (c + t) else "불합격"
    m = re.search(r"(?:≤|이하|max|최대)\s*[:=]?\s*(-?\d[\d.]*)", sp, re.I)
    if m or re.search(r"(-?\d[\d.]*)\s*이하", sp):
        lim = float((m or re.search(r"(-?\d[\d.]*)\s*이하", sp)).group(1))
        return "합격" if mv <= lim else "불합격"
    m = re.search(r"(?:≥|이상|min|최소)\s*[:=]?\s*(-?\d[\d.]*)", sp, re.I)
    if m or re.search(r"(-?\d[\d.]*)\s*이상", sp):
        lim = float((m or re.search(r"(-?\d[\d.]*)\s*이상", sp)).group(1))
        return "합격" if mv >= lim else "불합격"
    return None
