"""Orchestrator → dynamic Send workers → join → hybrid report quality loop."""

import importlib.util
import json
from collections.abc import Iterable
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agents import discovery, orchestrator, select_candidate
from config import OUTPUT_DIR, ROOT
from graph.routing import assign_workers, route_after_discover, route_after_judge, route_after_quality
from schemas import QualityCheck, QualityVerdict, TaskResult
from state import InvestState
from tools.evidence import with_evidence

_INTEGRATION_MODULES = tuple(f"agents.{name}" for name in (
    "research_worker", "synthesizer", "competitor", "investment_judge", "report_writer", "report_quality"))


def integration_ready():
    return all(importlib.util.find_spec(module) for module in _INTEGRATION_MODULES)


def _stubs(decisions=None):
    """Offline routing fixtures; paths/verdicts here are not real report validation."""
    fixture = json.loads((ROOT / "tests/fixtures/sample_state.json").read_text(encoding="utf-8"))
    pending = list(decisions or [fixture["decision"]])
    fallback = pending[-1]

    def worker(state):
        task_id = state["task"]["task_id"]
        result = TaskResult(task_id=task_id, status="insufficient", findings="stub fixture",
                            source_ids=[], evidence_ref=None, attempts=1)
        return {"worker_results": {task_id: result.model_dump()}}

    def synthesize(state):
        return {"tech_summary": fixture["tech_summary"], "market_analysis": fixture["market_analysis"],
                "log": [f"[synthesizer] stub: {len(state['worker_results'])}개 집계"]}

    def competitor(state):
        return {"competitor_analysis": fixture["competitor_analysis"], "log": ["[competitor] stub"]}

    def judge(state):
        decision = pending.pop(0) if pending else fallback
        startup = state["current_startup"]
        total = fixture["total_score"] if decision == "투자" else 60.0
        record = fixture["evaluation_history"][0] | {"startup": startup["name"], "segment": startup["segment"],
                                                   "decision": decision, "total_score": total}
        return {"scores": fixture["scores"], "total_score": total, "decision": decision,
                "evaluation_history": [record], "log": ["[judge] stub"]}

    def draft(state):
        return {"report_version": state.get("report_version", 0) + 1, "log": ["[report_draft] stub"]}

    def quality(state):
        check = QualityCheck(passed=True, reason="stub: 실제 품질 검증 아님", feedback=[])
        return {"quality_result": QualityVerdict(report_version=state["report_version"],
                groundedness=check, neutrality=check, bias=check, structure_passed=True,
                structure_reasons=[], recoverable=True).model_dump()}

    def finalize(state):
        return {"report_path": str(OUTPUT_DIR / "stub_report.pdf"), "execution_status": "complete",
                "log": ["[report_finalize] stub: PDF 생성 아님"]}

    return worker, synthesize, competitor, judge, draft, quality, finalize


def build_graph(stub: bool | None = None, stub_decisions: Iterable[str] | None = None):
    from agents import report_quality
    if stub is None:
        stub = not integration_ready()
    if stub:
        worker, synthesize, competitor, judge, draft, quality, finalize = _stubs(stub_decisions)
    else:
        from agents import competitor as competitor_module
        from agents import investment_judge, report_writer, research_worker, synthesizer
        worker, synthesize = research_worker.run, synthesizer.run
        competitor = with_evidence("competitor", competitor_module.run)
        judge = investment_judge.run
        draft, quality, finalize = report_writer.draft, report_quality.run, report_writer.finalize

    def initialize(state):
        return {"run_id": state.get("run_id") or uuid4().hex,
                "report_version": state.get("report_version", 0), "execution_status": "running"}

    graph = StateGraph(InvestState)
    for name, function in (
        ("initialize", initialize), ("discover", with_evidence("discover", discovery.run)),
        ("select", select_candidate.run), ("orchestrator", orchestrator.run),
        ("research_worker", worker), ("synthesizer", synthesize), ("competitor", competitor),
        ("judge", judge), ("report_draft", draft), ("report_quality", quality),
        ("report_finalize", finalize), ("quality_failed", report_quality.failed),
    ):
        graph.add_node(name, function)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "discover")
    graph.add_conditional_edges("discover", route_after_discover,
                               {"select": "select", "discover": "discover", "report": "report_draft"})
    graph.add_edge("select", "orchestrator")
    graph.add_conditional_edges("orchestrator", assign_workers, ["research_worker", "synthesizer"])
    graph.add_edge("research_worker", "synthesizer")
    graph.add_edge("synthesizer", "competitor")
    graph.add_edge("competitor", "judge")
    graph.add_conditional_edges("judge", route_after_judge,
                               {"report": "report_draft", "select": "select", "discover": "discover"})
    graph.add_edge("report_draft", "report_quality")
    graph.add_conditional_edges("report_quality", route_after_quality,
                               ["report_draft", "report_finalize", "quality_failed"])
    graph.add_edge("report_finalize", END)
    graph.add_edge("quality_failed", END)
    return graph.compile()
