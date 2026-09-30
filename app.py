import argparse
import json
import os
from datetime import date

from config import DATA_DIR, OUTPUT_DIR, RECURSION_LIMIT
from graph.builder import build_graph, integration_ready
from tools.web_search import set_cache_enabled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-only", action="store_true", help="웹 발굴 대신 시드 후보 사용"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="웹 검색 캐시를 무시하고 새로 검색"
    )
    parser.add_argument(
        "--stub", action="store_true", help="미통합 노드를 샘플 결과로 대체"
    )
    args = parser.parse_args()

    if args.seed_only:
        os.environ["SEED_ONLY"] = "1"
    set_cache_enabled(not args.no_cache)

    use_stub = args.stub or not integration_ready()
    vectorstore = DATA_DIR / "vectorstore"
    if not use_stub and not any(vectorstore.glob("*")):
        raise SystemExit("벡터스토어가 없습니다. 먼저 `python -m rag.ingest`를 실행하세요.")

    graph = build_graph(stub=use_stub)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "graph.mmd").write_text(
        graph.get_graph().draw_mermaid(), encoding="utf-8"
    )
    initial_state = {
        "domain": "Energy",
        "run_date": date.today().isoformat(),
        "candidates": [],
        "search_round": 0,
        "evaluated": [],
        "market_cache": {},
        "evaluation_history": [],
        "references": [],
        "log": [],
    }
    final = graph.invoke(
        initial_state, config={"recursion_limit": RECURSION_LIMIT}
    )
    run_log = {
        key: final.get(key)
        for key in ("evaluation_history", "scores", "decision", "references", "log")
    }
    (OUTPUT_DIR / "run_log.json").write_text(
        json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"보고서: {final.get('report_path')}")


if __name__ == "__main__":
    main()
