"""MES 품질검사 성적서 도메인. ★판정(judgment)은 측정값 vs 규격 업무규칙 대상."""
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from .base import Domain

# 판정 규칙(derive_judgment)을 eval 헬퍼에서 재사용 — 단일 진실원
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from _norm import derive_judgment  # noqa: E402


class QCReport(BaseModel):
    part_no: str | None = Field(None, description="품번")
    lot_no: str | None = Field(None, description="로트번호")
    inspect_date: str | None = Field(None, description="검사일 (YYYY-MM-DD)")
    item: str | None = Field(None, description="검사항목")
    spec: str | None = Field(None, description="규격/기준 (예: 10.00±0.20 mm, 45~55 HRC, ≤ 1.0 μm)")
    measured: str | None = Field(None, description="측정값 (수치+단위)")
    judgment: str | None = Field(None, description="판정 (합격/불합격)")
    inspector: str | None = Field(None, description="검사자")


def judgment_consistency(e: dict) -> list[str]:
    """추출된 측정값·규격으로 기대 판정을 계산해, 추출 판정과 불일치면 재추출 유도."""
    derived = derive_judgment(e.get("measured"), e.get("spec"))
    j = (e.get("judgment") or "").strip()
    if derived and j and j != derived:
        return [f"판정 불일치: measured={e.get('measured')} vs spec={e.get('spec')} → 기대 {derived}"]
    return []


DOMAIN = Domain(
    name="qc_report",
    schema=QCReport,
    required_fields=["part_no", "lot_no", "inspect_date", "judgment"],
    extract_instruction=(
        "다음 품질검사 성적서에서 part_no/lot_no/inspect_date/item/spec/measured/judgment/inspector를 "
        "추출하라. 날짜는 YYYY-MM-DD. 판정(judgment)은 측정값이 규격을 만족하면 '합격', 아니면 '불합격'. "
        "문서에 없는 필드는 추측하지 말고 비워라(null)."
    ),
    rules=[judgment_consistency],
)
