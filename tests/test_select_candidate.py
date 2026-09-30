from agents.select_candidate import run


def test_selects_first_candidate_and_resets_candidate_state():
    first = {"name": "Amperon"}
    second = {"name": "Tyba"}

    result = run({"candidates": [first, second]})

    assert result == {
        "candidates": [second],
        "current_startup": first,
        "evaluated": ["Amperon"],
        "tech_summary": None,
        "market_analysis": None,
        "competitor_analysis": None,
        "scores": {},
        "total_score": 0.0,
        "decision": None,
        "log": ["[select] Amperon 평가 시작 (남은 후보 1)"],
    }
