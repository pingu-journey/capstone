from langgraph.types import Overwrite


def run(state):
    queue = list(state["candidates"])
    current = queue.pop(0)
    update = {
        "candidates": queue,
        "current_startup": current,
        "evaluated": [current["name"]],
        "tech_summary": None,
        "market_analysis": None,
        "competitor_analysis": None,
        "scores": {},
        "total_score": 0.0,
        "decision": None,
        "log": [f"[select] {current['name']} 평가 시작 (남은 후보 {len(queue)})"],
    }
    if state.get("run_id"):
        update.update(tasks=[], worker_results=Overwrite({}))
    return update
