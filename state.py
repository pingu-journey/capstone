from operator import add
from typing import Annotated, Literal, TypedDict


def merge_dict(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


class InvestState(TypedDict, total=False):
    domain: str
    run_date: str
    log: Annotated[list[str], add]
    candidates: list[dict]
    search_round: int
    evaluated: Annotated[list[str], add]
    current_startup: dict | None
    tech_summary: dict | None
    market_analysis: dict | None
    market_cache: Annotated[dict[str, dict], merge_dict]
    competitor_analysis: dict | None
    scores: dict[str, dict]
    total_score: float
    decision: Literal["투자", "보류"] | None
    evaluation_history: Annotated[list[dict], add]
    references: Annotated[list[dict], add]
    report_path: str | None
