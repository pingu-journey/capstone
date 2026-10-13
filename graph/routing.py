from config import MAX_DISCOVERY_ROUNDS, MAX_EVALUATIONS, MAX_REPORT_IMPROVEMENTS
from langgraph.types import Send
from schemas import TaskSpec, QualityVerdict


def _queue_route(state) -> str:
    if len(state.get("evaluated", [])) >= MAX_EVALUATIONS:
        return "report"
    if state.get("candidates"):
        return "select"
    if state.get("search_round", 0) < MAX_DISCOVERY_ROUNDS:
        return "discover"
    return "report"


def route_after_judge(state) -> str:
    if state.get("decision") == "투자":
        return "report"
    return _queue_route(state)


def route_after_discover(state) -> str:
    return _queue_route(state)


def assign_workers(state):
    if not state["tasks"]:
        return "synthesizer"
    return [Send("research_worker", {
        "task": TaskSpec.model_validate(task).model_dump(),
        "current_startup": state["current_startup"],
        "run_id": state["run_id"], "run_date": state["run_date"],
    }) for task in state["tasks"]]


def route_after_quality(state):
    verdict = QualityVerdict.model_validate(state["quality_result"])
    if verdict.report_version != state["report_version"]:
        raise ValueError("품질 판정과 보고서 버전이 일치하지 않습니다.")
    if verdict.passed:
        return "report_finalize"
    if verdict.recoverable and state["report_version"] <= MAX_REPORT_IMPROVEMENTS:
        return "report_draft"
    return "quality_failed"
