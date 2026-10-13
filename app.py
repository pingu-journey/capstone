"""CLI: correlate plans, evidence, report revisions and LangSmith under one run ID."""

import argparse
import hashlib
import logging
import os
import platform
from importlib.metadata import version
from datetime import date
from uuid import uuid4, UUID

from langsmith import tracing_context, Client
from langchain_core.tracers.langchain import wait_for_all_tracers

from config import DATA_DIR, ROOT, RECURSION_LIMIT, LLM_MODEL, JUDGE_MODEL
from graph.builder import build_graph, integration_ready
from tools.artifacts import run_dir, write_json
from tools.web_search import set_cache_enabled

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-only", action="store_true", help="웹 발굴 대신 시드 후보 사용")
    parser.add_argument("--no-cache", action="store_true", help="웹 검색 캐시 무시")
    parser.add_argument("--stub", action="store_true", help="모의 라우팅: 실제 PDF·품질 평가 아님")
    args = parser.parse_args()
    if args.seed_only:
        os.environ["SEED_ONLY"] = "1"
    set_cache_enabled(not args.no_cache)
    if not args.stub and not integration_ready():
        raise RuntimeError("에이전트 구현 모듈이 누락되었습니다.")
    if not args.stub and not any((DATA_DIR / "vectorstore").glob("*")):
        raise RuntimeError("벡터스토어가 없습니다. 먼저 `python -m rag.ingest`를 실행하세요.")
    run_id = uuid4().hex
    state = {"run_id": run_id, "domain": "Energy", "run_date": date.today().isoformat(),
             "candidates": [], "search_round": 0, "evaluated": [], "market_cache": {},
             "research_cache": {}, "tasks": [], "worker_results": {}, "report_version": 0,
             "quality_result": None, "execution_status": "running", "evaluation_history": [],
             "references": [], "log": []}
    directory = run_dir(state)
    directory.mkdir(parents=True, exist_ok=False)
    graph = build_graph(stub=args.stub)
    (directory / "graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    tracing = not args.stub and os.getenv("LANGSMITH_TRACING", "true" if os.getenv("LANGSMITH_API_KEY") else "false").lower() == "true"
    project = os.getenv("LANGSMITH_PROJECT", "energy-orchestrator-workers")
    manifest = {"run_id": run_id, "run_date": state["run_date"], "python": platform.python_version(),
                "models": {"generator": LLM_MODEL, "judge": JUDGE_MODEL}, "stub": args.stub,
                "libraries": {name: version(name) for name in ("langgraph", "langchain-openai", "sentence-transformers", "chromadb")},
                "embedding": {"model": "BAAI/bge-m3", "backend": os.getenv("EMBEDDING_BACKEND", "local")},
                "search_cache": not args.no_cache, "seed_only": args.seed_only,
                "langsmith_project": project, "langsmith_run_id": run_id if tracing else None,
                "inputs_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in list((ROOT / "prompts").glob("*.md")) + list((DATA_DIR / "docs").glob("*/*.pdf")) + [ROOT / "config/docs.yaml", ROOT / "config/rubric.yaml", ROOT / "data/seed_candidates.json", ROOT / "requirements.txt"]},
                "search_cache_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in (ROOT / "outputs/cache/search").glob("*.json")}}
    write_json(directory / "manifest.json", manifest)
    logger.info("실행 시작: run_id=%s tracing=%s", run_id, tracing)
    try:
        with tracing_context(enabled=tracing, project_name=project, metadata={"run_id": run_id}, tags=["orchestrator-workers"]):
            final = graph.invoke(state, config={"recursion_limit": RECURSION_LIMIT, "run_id": UUID(run_id),
                                 "run_name": "energy-investment", "metadata": {"run_id": run_id}, "max_concurrency": 7})
    except Exception as error:
        write_json(directory / "failure.json", {"run_id": run_id, "error_code": type(error).__name__})
        raise
    write_json(directory / "run_log.json", final)
    if tracing:
        wait_for_all_tracers()
        try:
            client = Client()
            trace = client.read_run(UUID(run_id))
            manifest["langsmith_url"] = client.get_run_url(run=trace, project_name=project)
            manifest["trace_verified"] = True
        except Exception:
            manifest["trace_verified"] = False
        write_json(directory / "manifest.json", manifest)
    print(f"실행: {run_id}")
    if args.stub:
        print("stub 라우팅 확인 완료 (실제 보고서·품질 검증 아님)")
    elif final["execution_status"] == "complete":
        print(f"보고서: {final['report_path']}")
    else:
        print(f"품질 미달: 검토용 초안과 사유를 {directory}에 보존했습니다.")
    return 0 if final["execution_status"] == "complete" else 1


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for name in ("httpx", "httpx2", "httpcore", "openai", "fontTools.subset"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("실행 실패")
        raise SystemExit(1)
