import importlib.util
import json
from collections.abc import Iterable

from langgraph.graph import END, START, StateGraph

from agents import discovery, select_candidate
from config import OUTPUT_DIR, ROOT
from graph.routing import route_after_discover, route_after_judge
from state import InvestState


_INTEGRATION_MODULES = (
    "agents.tech_summary",
    "agents.market_eval",
    "agents.competitor",
    "agents.investment_judge",
    "agents.report_writer",
)


def integration_ready() -> bool:
    return all(importlib.util.find_spec(module) for module in _INTEGRATION_MODULES)


def _stubs(decisions: Iterable[str] | None = None):
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "sample_state.json").read_text(
            encoding="utf-8"
        )
    )
    pending_decisions = list(decisions or [fixture["decision"]])
    fallback_decision = pending_decisions[-1]

    def tech(_state):
        return {
            "tech_summary": fixture["tech_summary"],
            "references": [],
            "log": ["[tech_summary] stub"],
        }

    def market(_state):
        return {
            "market_analysis": fixture["market_analysis"],
            "references": [],
            "log": ["[market_eval] stub"],
        }

    def competitor(_state):
        return {
            "competitor_analysis": fixture["competitor_analysis"],
            "references": [],
            "log": ["[competitor] stub"],
        }

    def judge(state):
        current = state["current_startup"]
        decision = pending_decisions.pop(0) if pending_decisions else fallback_decision
        total_score = fixture["total_score"] if decision == "투자" else 60.0
        record = fixture["evaluation_history"][0] | {
            "startup": current["name"],
            "segment": current["segment"],
            "total_score": total_score,
            "decision": decision,
        }
        return {
            "scores": fixture["scores"],
            "total_score": total_score,
            "decision": decision,
            "evaluation_history": [record],
            "log": ["[judge] stub"],
        }

    def report(_state):
        return {
            "report_path": str(OUTPUT_DIR / "stub_report.pdf"),
            "log": ["[report] stub (PDF 생성은 담당 D 통합 후 실행)"],
        }

    return tech, market, competitor, judge, report


def build_graph(
    stub: bool | None = None, stub_decisions: Iterable[str] | None = None
):
    if stub is None:
        stub = not integration_ready()

    if stub:
        tech, market, competitor, judge, report = _stubs(stub_decisions)
    else:
        from agents import competitor as competitor_module
        from agents import investment_judge, market_eval, report_writer, tech_summary

        tech = tech_summary.run
        market = market_eval.run
        competitor = competitor_module.run
        judge = investment_judge.run
        report = report_writer.run

    graph = StateGraph(InvestState)
    graph.add_node("discover", discovery.run)
    graph.add_node("select", select_candidate.run)
    graph.add_node("tech_summary", tech)
    graph.add_node("market_eval", market)
    graph.add_node("competitor", competitor)
    graph.add_node("judge", judge)
    graph.add_node("report", report)
    graph.add_edge(START, "discover")
    graph.add_conditional_edges(
        "discover",
        route_after_discover,
        {"select": "select", "discover": "discover", "report": "report"},
    )
    graph.add_edge("select", "tech_summary")
    graph.add_edge("select", "market_eval")
    graph.add_edge(["tech_summary", "market_eval"], "competitor")
    graph.add_edge("competitor", "judge")
    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {"report": "report", "select": "select", "discover": "discover"},
    )
    graph.add_edge("report", END)
    return graph.compile()
