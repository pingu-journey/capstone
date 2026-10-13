"""Deterministic join of every planned task; preserve valid cache and partial successes."""

import logging

from agents import market_eval, orchestrator, tech_summary
from schemas import TaskResult, TaskSpec, TechSummary
from tools.artifacts import read_json, run_dir, write_json
from tools.evidence import capture, index_evidence, save_node, record
from tools.references import deduplicate_references
from tools.services import TransientServiceError
from config import MAX_CALL_ATTEMPTS

logger = logging.getLogger(__name__)


def _assemble(function, fallback):
    for _ in range(MAX_CALL_ATTEMPTS):
        try:
            return function()
        except TransientServiceError:
            continue
    return fallback()


def run(state):
    tasks = [TaskSpec.model_validate(item) for item in state["tasks"]]
    expected = {task.task_id for task in tasks}
    if set(state.get("worker_results", {})) != expected:
        raise ValueError("계획된 모든 작업의 결과가 필요합니다.")
    payloads = {}
    successful_tasks = set()
    failures = []
    for task in tasks:
        result = TaskResult.model_validate(state["worker_results"][task.task_id])
        if result.task_id != task.task_id:
            raise ValueError("계획과 결과의 task_id가 일치하지 않습니다.")
        artifact = read_json(result.evidence_ref)
        if artifact["run_id"] != state["run_id"] or artifact["task"] != task.model_dump():
            raise ValueError("다른 실행 또는 작업의 근거 파일입니다.")
        if result.status == "success":
            payloads[task.kind] = artifact["payload"]
            successful_tasks.add(task.task_id)
        else:
            failures.append(task.kind)
    startup = state["current_startup"]
    name = startup["name"]
    cached_tech = orchestrator.valid_research_cache(state)
    cached_market = orchestrator.valid_market_cache(state)
    with capture() as records:
        if cached_tech:
            tech_update = {"tech_summary": cached_tech["tech_summary"], "references": cached_tech["references"]}
        else:
            def rag_result(kind):
                payload = payloads.get(kind, {})
                return {"answer": payload.get("answer", "공개 정보 부족"),
                        "sources": payload.get("references", []), "log": []}
            tech_args = (startup, state["run_date"], rag_result("company_tech"), rag_result("industry"),
                         payloads.get("team", {}).get("results", []))

            def tech_fallback():
                summary = TechSummary.model_validate(tech_summary._fallback().model_dump())
                summary.insufficient = tech_summary._insufficient(summary)
                return {"tech_summary": summary.model_dump(), "references": []}
            if any(payloads.get(kind, {}).get("references") for kind in ("company_tech", "industry", "team")):
                tech_update = _assemble(lambda: tech_summary.synthesize(*tech_args), tech_fallback)
            else:
                tech_update = tech_fallback()

        market_results = [result for kind in orchestrator.MARKET_FIELDS for result in payloads.get(kind, {}).get("results", [])]
        if market_results:
            doc_ids = list(dict.fromkeys(item["doc_id"] for item in market_results))
            context = "\n\n".join(f"[source_id={item['doc_id']}, page={item['page']}]\n{item['text']}" for item in market_results)
            market_update = _assemble(lambda: market_eval.synthesize(startup, context, doc_ids),
                                      lambda: {"market_analysis": market_eval._fallback(startup["country"], startup["segment"], []).model_dump(), "references": []})
            market = market_update["market_analysis"]
        else:
            market = market_eval._fallback(startup["country"], startup["segment"], []).model_dump()
        # A whole-schema model response is only authorized to fill successful
        # task targets that were actually missing. Other fields retain cache or
        # explicit fallback, including when there is no cache at all.
        merged = dict(cached_market) if cached_market else market_eval._fallback(
            startup["country"], startup["segment"], []).model_dump()
        successful_fields = set()
        for task in tasks:
            if task.task_id not in successful_tasks or task.kind not in orchestrator.MARKET_FIELDS:
                continue
            for field in set(task.target_fields) & set(orchestrator.MARKET_FIELDS[task.kind]):
                successful_fields.add(field)
                if not orchestrator.has_content(merged[field]) and orchestrator.has_content(market[field]):
                    merged[field] = market[field]
        all_fields = {field for fields in orchestrator.MARKET_FIELDS.values() for field in fields}
        if not cached_market and successful_fields:
            merged["summary"] = market["summary"] if successful_fields == all_fields else "부분 조사 결과이며 미완료 관점은 공개 정보 부족이다."
        merged["sources"] = deduplicate_references(merged["sources"] + market["sources"])
        market = merged
        market["sources"] = market_eval._copy_sources(market["sources"], name)
        if cached_market:
            remapped = {ref["id"]: ref for ref in market["sources"]}
            for item in read_json(state["evidence_index_ref"]):
                if item["source_id"] in remapped:
                    record(remapped[item["source_id"]], item["text"], item.get("page"))
        market["country"] = startup["country"].upper()
        key = f"{market['country']}|{market['segment']}"
        references = deduplicate_references(tech_update["references"] + market["sources"])
        update = {
            "tech_summary": tech_update["tech_summary"], "market_analysis": market,
            "market_cache": {key: market}, "references": references,
            "research_cache": {name: {"candidate": startup, "tech_summary": tech_update["tech_summary"],
                                      "references": tech_update["references"]}},
        }
    save_node(state, "synthesizer", update, records)
    update["evidence_index_ref"] = index_evidence(state)
    write_json(run_dir(state) / "analyses" / f"candidate-{len(state.get('evaluated', []))}.json",
               {"startup": name, "tasks": state["tasks"], "worker_results": state.get("worker_results", {}),
                "tech_summary": update["tech_summary"], "market_analysis": market})
    logger.info("조사 집계 완료: run_id=%s tasks=%d gaps=%d", state["run_id"], len(tasks), len(failures))
    update["log"] = [f"[synthesizer] {name}: {len(tasks)}개 집계, 부족·실패 {failures or '없음'}"]
    return update
