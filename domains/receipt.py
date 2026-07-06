"""영수증/세금계산서 도메인 (baseline)."""
from pydantic import BaseModel, Field

from .base import Domain


class Invoice(BaseModel):
    vendor: str | None = Field(None, description="공급자/상호")
    date: str | None = Field(None, description="날짜 (YYYY-MM-DD)")
    total: int | None = Field(None, description="총액 (원, 숫자만)")
    items: list[str] = Field(default_factory=list, description="품목 목록")


DOMAIN = Domain(
    name="receipt",
    schema=Invoice,
    required_fields=["vendor", "date", "total"],
    extract_instruction=(
        "다음 문서에서 vendor/date/total/items를 정확히 추출하라. "
        "날짜는 YYYY-MM-DD 형식으로. 문서에 없는 필드는 추측하지 말고 비워라(null)."
    ),
)
