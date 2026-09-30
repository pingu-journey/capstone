from config import MAX_DISCOVERY_ROUNDS, MAX_EVALUATIONS


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
