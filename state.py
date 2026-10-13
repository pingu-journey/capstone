from operator import add
from typing import Annotated, Literal, TypedDict


def merge_dict(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


def merge_worker_results(a: dict, b: dict) -> dict:
    """Task IDs are unique per candidate; identical redelivery is idempotent."""
    merged = dict(a or {})
    for task_id, result in (b or {}).items():
        if task_id in merged and merged[task_id] != result:
            raise ValueError("충돌하는 worker 결과: task_id=" + task_id)
        merged[task_id] = result
    return dict(sorted(merged.items()))


class InvestState(TypedDict, total=False):
    # Control: routing, plan, correlation and finite quality loop.
    run_id: str
    tasks: list[dict]
    report_version: int
    quality_result: dict | None
    execution_status: Literal["running", "complete", "quality_failed"]
    # Payload: candidate-scoped deltas and external artifact pointers.
    worker_results: Annotated[dict[str, dict], merge_worker_results]
    evidence_index_ref: str
    report_draft_ref: str | None
    report_preview_ref: str | None
    research_cache: Annotated[dict[str, dict], merge_dict]
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
