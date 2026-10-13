"""One task per Send. Workers only publish task-result deltas and external evidence."""

import logging
from typing import TypedDict

from agents import tech_summary
from config import MAX_CALL_ATTEMPTS
from rag import advanced_rag, agentic_rag
from schemas import Candidate, TaskSpec, TaskResult
from tools.artifacts import run_dir, write_json
from tools.evidence import capture, record
from tools.references import make_doc_ref, make_web_ref
from tools.services import TransientServiceError, check_service_error

logger = logging.getLogger(__name__)


class WorkerState(TypedDict):
    task: dict
    current_startup: dict
    run_id: str
    run_date: str


def collect(task, startup, run_date):
    name = startup["name"]
    company = tech_summary.company_label(startup)
    entity = {key: startup.get(key) for key in ("name", "segment", "homepage")}
    if task.kind in ("company_tech", "industry"):
        result = agentic_rag.run(
            task.query, company, name, run_date,
            entity=entity if task.kind == "company_tech" else None,
            web_queries=None if task.kind == "company_tech" else [task.query, f"{tech_summary.SEGMENT_EN[startup['segment']]} AI technology readiness commercialization"],
            call_attempts=1,
        )
        return {"answer": result["answer"], "references": result["sources"], "path": result["path"], "rewrites": result["rewrites"]}
    if task.kind == "team":
        results, _ = tech_summary._team_search(company, entity)
        refs = [make_web_ref(result, name, "tech_summary", run_date) for result in results]
        return {"answer": "팀 검색 근거 확보" if results else "공개 정보 부족", "results": results, "references": refs}
    results = advanced_rag.search(task.query, startup["country"].upper(), startup["segment"], k=5)
    refs = {}
    for result in results:
        ref = make_doc_ref(result["doc_id"], name, "market_eval")
        refs[ref["id"]] = ref
        record(ref, result["text"], result["page"])
    return {"answer": "시장 검색 근거 확보" if results else "공개 정보 부족", "results": results, "references": list(refs.values())}


def run(state: WorkerState):
    task = TaskSpec.model_validate(state["task"])
    startup = Candidate.model_validate(state["current_startup"]).model_dump()
    payload, records, error_code = {}, [], None
    for attempt in range(1, MAX_CALL_ATTEMPTS + 1):
        try:
            with capture() as captured:
                payload = collect(task, startup, state["run_date"])
            records = captured
            break
        except TransientServiceError:
            error_code = "service_unavailable"
            logger.warning("조사 일시 실패: run_id=%s task_id=%s attempt=%d", state["run_id"], task.task_id, attempt)
        except Exception as error:
            try:
                check_service_error(error)
            except TransientServiceError:
                error_code = "service_unavailable"
                logger.warning("조사 일시 실패: run_id=%s task_id=%s attempt=%d", state["run_id"], task.task_id, attempt)
                continue
            raise
    status = "failed" if not payload else "success" if payload["references"] and payload["answer"] != "공개 정보 부족" else "insufficient"
    if payload:
        error_code = None
    path = write_json(run_dir(state) / "workers" / f"{task.task_id}.json",
                      {"run_id": state["run_id"], "startup": startup["name"], "task": task.model_dump(),
                       "payload": payload, "evidence": records})
    result = TaskResult(task_id=task.task_id, status=status,
                        findings=payload.get("answer", "공개 정보 부족"),
                        source_ids=[ref["id"] for ref in payload.get("references", [])],
                        evidence_ref=path, attempts=attempt, error_code=error_code)
    logger.info("조사 완료: run_id=%s task_id=%s status=%s", state["run_id"], task.task_id, status)
    return {"worker_results": {task.task_id: result.model_dump()}}
