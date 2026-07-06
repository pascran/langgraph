"""도메인 어댑터 — 하네스를 도메인-무관으로 만드는 스펙.

도메인 = (추출 스키마 + 필수필드 + 업무규칙 + 추출 지시).
agent.py는 이 Domain을 주입받아 동작하고, 스키마/규칙/골든만 갈아끼우면 새 도메인에 붙는다.
"""
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel


@dataclass
class Domain:
    name: str
    schema: type[BaseModel]                       # with_structured_output용 pydantic 스키마
    required_fields: list[str]                    # validate 노드 필수필드
    extract_instruction: str                      # 도메인 튜닝 추출 프롬프트
    rules: list[Callable[[dict], list[str]]] = field(default_factory=list)  # 업무규칙

    def check_rules(self, extracted: dict) -> list[str]:
        issues: list[str] = []
        for rule in self.rules:
            issues += rule(extracted)
        return issues
