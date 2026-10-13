"""Prove variable fan-out, deterministic joins, partial failures and candidate reset."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.graph import END, START, StateGraph

from agents import orchestrator, research_worker, select_candidate, synthesizer
from config import ROOT
from graph.routing import assign_workers
from schemas import TaskResult
from state import InvestState, merge_worker_results
from tools import artifacts
from tools.artifacts import read_json, run_dir, write_json
from tools.services import ConfigurationError, TransientServiceError
from tools.references import make_web_ref


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUTPUT_DIR", tmp_path)
    value = json.loads((ROOT / "tests/fixtures/sample_state.json").read_text(encoding="utf-8"))
    value.update(run_id="test_run", tasks=[], worker_results={}, research_cache={}, market_cache={})
    ref = make_web_ref({"title": "Independent test evidence", "url": "https://independent.org/test"},
                       "Amperon", "test", value["run_date"])
    value["references"] = [ref]
    value["evidence_index_ref"] = write_json(run_dir(value) / "evidence_index.json", [
        {"source_id": ref["id"], "startup": "Amperon", "text": "시장 성장과 고객 수요 근거 100 GW", "page": None, "reference": ref}])
    return value


def market_cache(state):
    ref = state["references"][0]
    market = deepcopy(state["market_analysis"])
    market.update(market_size=[{"metric": "용량", "value": "100 GW", "year": "2030", "source_id": ref["id"]}],
                  policy_risks=["정책 변화 모니터링"], sources=[ref])
    return market


def test_plan_has_seven_then_three_then_four_tasks(state):
    before = deepcopy(state)
    cold = orchestrator.run(state)
    assert len(cold["tasks"]) == 7
    assert state == before
    state["market_cache"] = {"US|demand_forecasting": market_cache(state)}
    assert [task["kind"] for task in orchestrator.run(state)["tasks"]] == ["company_tech", "industry", "team"]
    state["market_cache"]["US|demand_forecasting"]["growth"] = "공개 정보 부족"
    plan = orchestrator.run(state)
    assert [task["kind"] for task in plan["tasks"]] == ["company_tech", "industry", "team", "growth"]
    assert plan["tasks"][-1]["target_fields"] == ["growth", "demand_drivers"]
    assert list((run_dir(state) / "plans").glob("*.json"))


def test_schema_valid_cache_without_excerpts_is_not_reused(state):
    state["market_cache"] = {"US|demand_forecasting": market_cache(state)}
    state.pop("evidence_index_ref")
    assert len(orchestrator.run(state)["tasks"]) == 7


def test_zero_tasks_routes_to_synthesizer(state):
    state["market_cache"] = {"US|demand_forecasting": market_cache(state)}
    state["research_cache"] = {"Amperon": {"candidate": state["current_startup"],
        "tech_summary": state["tech_summary"], "references": state["references"]}}
    state.update(orchestrator.run(state))
    assert state["tasks"] == []
    assert assign_workers(state) == "synthesizer"
    update = synthesizer.run(state)
    assert update["market_analysis"]["market_size"][0]["value"] == "100 GW"
    assert update["tech_summary"] == state["tech_summary"]


def test_dispatch_sends_minimum_context_and_persisted_plan(state):
    state.update(orchestrator.run(state))
    sends = assign_workers(state)
    assert len(sends) == 7
    assert all(send.node == "research_worker" for send in sends)
    assert all(set(send.arg) == {"task", "current_startup", "run_id", "run_date"} for send in sends)
    assert [send.arg["task"] for send in sends] == state["tasks"]


def test_result_reducer_is_order_independent_and_idempotent():
    a, b = {"b": {"status": "success"}}, {"a": {"status": "failed"}}
    assert merge_worker_results(a, b) == merge_worker_results(b, a)
    assert merge_worker_results(a, a) == a
    with pytest.raises(ValueError, match="충돌"):
        merge_worker_results(a, {"b": {"status": "failed"}})


def test_candidate_reset_overwrites_results_and_keeps_history(state):
    state["candidates"] = [state["current_startup"]]
    state["worker_results"] = {"old": {"status": "success"}}
    graph = StateGraph(InvestState)
    graph.add_node("select", select_candidate.run)
    graph.add_edge(START, "select")
    graph.add_edge("select", END)
    result = graph.compile().invoke(state)
    assert result["worker_results"] == {} and result["tasks"] == []
    assert result["references"] == state["references"]
    assert result["evaluation_history"] == state["evaluation_history"]


@pytest.mark.parametrize("recover", [True, False])
def test_worker_retries_once_without_nested_rag_retries(state, monkeypatch, recover):
    task = orchestrator.run(state)["tasks"][0]
    calls = []

    def collect(*args):
        calls.append(args)
        if len(calls) == 1 or not recover:
            raise TransientServiceError("safe failure")
        return {"answer": "확인된 근거", "references": state["references"]}

    monkeypatch.setattr(research_worker, "collect", collect)
    update = research_worker.run({key: state[key] for key in ("run_id", "run_date", "current_startup")} | {"task": task})
    result = TaskResult.model_validate(update["worker_results"][task["task_id"]])
    assert len(calls) == 2 and result.attempts == 2
    assert result.status == ("success" if recover else "failed")
    assert set(update) == {"worker_results"}
    assert Path(result.evidence_ref).is_file()


def test_worker_auth_failure_is_not_an_evidence_gap(state, monkeypatch):
    task = orchestrator.run(state)["tasks"][0]
    def collect(*args):
        raise ConfigurationError("auth failure")
    monkeypatch.setattr(research_worker, "collect", collect)
    with pytest.raises(ConfigurationError):
        research_worker.run({key: state[key] for key in ("run_id", "run_date", "current_startup")} | {"task": task})


def test_rag_worker_disables_internal_retry(state, monkeypatch):
    task = SimpleNamespace(kind="company_tech", query="질문")
    calls = []
    def rag(*args, **kwargs):
        calls.append(kwargs)
        return {"answer": "공개 정보 부족", "sources": [], "path": "web", "rewrites": 2}
    monkeypatch.setattr(research_worker.agentic_rag, "run", rag)
    research_worker.collect(task, state["current_startup"], state["run_date"])
    assert calls[0]["call_attempts"] == 1


def test_actual_send_join_runs_once_after_all_tasks(state):
    calls = []
    def worker(value):
        tid = value["task"]["task_id"]
        return {"worker_results": {tid: {"task_id": tid}}}
    def join(value):
        calls.append(set(value["worker_results"]))
        return {}
    graph = StateGraph(InvestState)
    graph.add_node("orchestrator", orchestrator.run)
    graph.add_node("research_worker", worker)
    graph.add_node("synthesizer", join)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", assign_workers, ["research_worker", "synthesizer"])
    graph.add_edge("research_worker", "synthesizer")
    graph.add_edge("synthesizer", END)
    final = graph.compile().invoke(state)
    assert len(calls) == 1 and calls[0] == {task["task_id"] for task in final["tasks"]}


def test_join_rejects_missing_results(state):
    state.update(orchestrator.run(state))
    with pytest.raises(ValueError, match="모든 작업"):
        synthesizer.run(state)


def test_partial_failures_preserve_valid_market_cache(state):
    state["market_cache"] = {"US|demand_forecasting": market_cache(state)}
    state["market_cache"]["US|demand_forecasting"]["growth"] = "공개 정보 부족"
    state.update(orchestrator.run(state))
    for task in state["tasks"]:
        ref = write_json(run_dir(state) / "workers" / f"{task['task_id']}.json",
                         {"run_id": state["run_id"], "task": task, "payload": {}, "evidence": []})
        state["worker_results"][task["task_id"]] = TaskResult(task_id=task["task_id"], status="failed",
            findings="공개 정보 부족", source_ids=[], evidence_ref=ref, attempts=2, error_code="service_unavailable").model_dump()
    update = synthesizer.run(state)
    assert update["market_analysis"]["market_size"] == market_cache(state)["market_size"]
    assert update["market_analysis"]["customers"] == state["market_analysis"]["customers"]
    assert "TRL" in update["tech_summary"]["insufficient"]
    assert update["references"] and "market_cache" in update


@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("growth_status", ["failed", "insufficient", "success"])
@pytest.mark.parametrize("narrow_target", [True, False])
def test_market_join_updates_only_successful_missing_task_fields(state, monkeypatch, cached, growth_status, narrow_target):
    prior = market_cache(state)
    prior.update(growth="공개 정보 부족", policy_risks=["공개 정보 부족"])
    if cached:
        state["market_cache"] = {"US|demand_forecasting": prior}
    tasks = orchestrator.run(state)["tasks"]
    state["tasks"] = [task for task in tasks if task["kind"] in ("growth", "regulation")]
    if narrow_target:
        next(task for task in state["tasks"] if task["kind"] == "growth")["target_fields"] = ["growth"]
    ref = state["references"][0]
    for task in state["tasks"]:
        status = growth_status if task["kind"] == "growth" else "success"
        # Even an insufficient payload with references must not enter aggregation.
        payload = {} if status == "failed" else {"answer": "공개 정보 부족" if status == "insufficient" else "확인된 근거",
            "references": [ref], "results": [{"doc_id": task["kind"], "page": 1, "text": task["kind"] + " 근거"}]}
        path = write_json(run_dir(state) / "workers" / f"{task['task_id']}.json",
                          {"run_id": state["run_id"], "task": task, "payload": payload, "evidence": []})
        state["worker_results"][task["task_id"]] = TaskResult(task_id=task["task_id"], status=status,
            findings=payload.get("answer", "공개 정보 부족"), source_ids=[ref["id"]] if payload else [],
            evidence_ref=path, attempts=2 if status == "failed" else 1).model_dump()
    contexts = []
    def synthesize(startup, context, doc_ids):
        contexts.append(doc_ids)
        # The whole-schema LLM can populate unrelated fields from policy evidence.
        market = deepcopy(prior)
        market.update(growth="새 성장률", demand_drivers=["새 수요 요인"], policy_risks=["새 정책 근거"],
                      customers=["꾸며낸 고객"], summary="새 요약", sources=[ref])
        return {"market_analysis": market, "references": [ref]}
    monkeypatch.setattr(synthesizer.market_eval, "synthesize", synthesize)
    update = synthesizer.run(state)
    market = update["market_analysis"]
    assert contexts == [["growth", "regulation"] if growth_status == "success" else ["regulation"]]
    assert market["growth"] == ("새 성장률" if growth_status == "success" else "공개 정보 부족")
    assert market["demand_drivers"] == (prior["demand_drivers"] if cached else ["새 수요 요인"] if growth_status == "success" and not narrow_target else [])
    assert market["policy_risks"] == ["새 정책 근거"]
    assert market["customers"] == (prior["customers"] if cached else [])
    assert update["market_cache"]["US|demand_forecasting"] == market
    assert market["sources"] and all(item["startup"] == "Amperon" for item in update["references"])


def test_all_successful_market_tasks_join_and_keep_full_summary(state, monkeypatch):
    state["research_cache"] = {"Amperon": {"candidate": state["current_startup"],
        "tech_summary": state["tech_summary"], "references": state["references"]}}
    state.update(orchestrator.run(state))
    assert {task["kind"] for task in state["tasks"]} == set(orchestrator.MARKET_FIELDS)
    ref = state["references"][0]
    for task in state["tasks"]:
        payload = {"answer": "확인된 근거", "references": [ref],
                   "results": [{"doc_id": task["kind"], "page": 1, "text": task["kind"]}]}
        path = write_json(run_dir(state) / "workers" / f"{task['task_id']}.json",
                          {"run_id": state["run_id"], "task": task, "payload": payload, "evidence": []})
        state["worker_results"][task["task_id"]] = TaskResult(task_id=task["task_id"], status="success",
            findings="확인된 근거", source_ids=[ref["id"]], evidence_ref=path, attempts=1).model_dump()
    expected = market_cache(state)
    monkeypatch.setattr(synthesizer.market_eval, "synthesize", lambda *args: {"market_analysis": expected, "references": [ref]})
    update = synthesizer.run(state)
    assert update["market_analysis"] == expected
    assert update["market_cache"]["US|demand_forecasting"] == expected
    assert update["references"] == [ref]
