import os
from datetime import date

from config import RECURSION_LIMIT
from graph.builder import build_graph
from graph.routing import route_after_discover, route_after_judge


def test_investment_goes_to_report():
    assert route_after_judge({"decision": "투자"}) == "report"


def test_rejected_candidate_uses_remaining_queue():
    state = {
        "decision": "보류",
        "candidates": [{}],
        "evaluated": [],
        "search_round": 1,
    }
    assert route_after_judge(state) == "select"


def test_empty_queue_rediscovers_before_limit():
    state = {"candidates": [], "evaluated": [], "search_round": 2}
    assert route_after_discover(state) == "discover"


def test_discovery_limit_goes_to_report():
    state = {"candidates": [], "evaluated": [], "search_round": 3}
    assert route_after_discover(state) == "report"


def test_evaluation_limit_wins_over_queue():
    state = {
        "candidates": [{}],
        "evaluated": list(map(str, range(8))),
        "search_round": 1,
    }
    assert route_after_judge(state | {"decision": "보류"}) == "report"


def _invoke_stub(decisions, evaluated=None):
    previous = os.environ.get("SEED_ONLY")
    os.environ["SEED_ONLY"] = "1"
    try:
        return build_graph(stub=True, stub_decisions=decisions).invoke(
            {
                "domain": "Energy",
                "run_date": date.today().isoformat(),
                "candidates": [],
                "search_round": 0,
                "evaluated": evaluated or [],
                "market_cache": {},
                "evaluation_history": [],
                "references": [],
                "log": [],
            },
            config={"recursion_limit": RECURSION_LIMIT},
        )
    finally:
        if previous is None:
            os.environ.pop("SEED_ONLY", None)
        else:
            os.environ["SEED_ONLY"] = previous


def test_stub_investment_path_reaches_report():
    final = _invoke_stub(["투자"])
    assert final["decision"] == "투자"
    assert len(final["evaluation_history"]) == 1
    assert final["report_path"].endswith("stub_report.pdf")


def test_stub_rejection_uses_next_candidate():
    final = _invoke_stub(["보류", "투자"])
    assert [record["decision"] for record in final["evaluation_history"]] == [
        "보류",
        "투자",
    ]
    assert len(final["evaluated"]) == 2


def test_stub_empty_queue_runs_rediscovery_to_limit():
    final = _invoke_stub(
        ["보류"], ["해줌", "식스티헤르츠", "Amperon", "Tyba"]
    )
    assert final["search_round"] == 3
    assert sum(message.startswith("[discover]") for message in final["log"]) == 3
    assert final["report_path"].endswith("stub_report.pdf")


def test_stub_all_held_path_reaches_report():
    final = _invoke_stub(["보류"])
    assert final["decision"] == "보류"
    assert len(final["evaluation_history"]) == 5
    assert final["search_round"] == 3
    assert final["report_path"].endswith("stub_report.pdf")
