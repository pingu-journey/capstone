"""Run the real graph with isolated artifacts and record a safe node timeline.

Usage: .venv/bin/python -u docs/run_report_flow_e2e.py
Replay a saved real analysis: add --report-only path/to/run_log.json
No model/Judge mocks. Existing search cache is copied; existing outputs are untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

replay = Path(sys.argv[2]).resolve() if len(sys.argv) == 3 and sys.argv[1] == "--report-only" else None
suite = ROOT / "outputs" / "runs" / ("report-flow-e2e-" + uuid4().hex[:12])
suite.mkdir(parents=True, exist_ok=False)
original_output = config.OUTPUT_DIR
config.OUTPUT_DIR = suite / "isolated-output"
cache = original_output / "cache" / "search"
if cache.exists():
    shutil.copytree(cache, config.OUTPUT_DIR / "cache" / "search")
os.environ["LANGSMITH_TRACING"] = "false"
import app
from agents import report_writer
from report.pdf_renderer import MAX_PAGES
from schemas import QualityVerdict

start = time.monotonic()
summary = {"scope": "report_segment" if replay else "full_graph", "real_services": True, "seed_only": not bool(replay),
           "search_cache": "copied existing cache; cache misses use real Tavily",
           "started_at": datetime.now(timezone.utc).isoformat(),
           "models": {"generator": config.LLM_MODEL, "judge": config.JUDGE_MODEL},
           "suite_dir": str(suite.relative_to(ROOT)), "events": [], "assertions": [],
           "generation_requests": [],
           "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for folder in ("agents", "graph", "tools", "prompts")
                             for p in (ROOT / folder).glob("*.*") if p.suffix in (".py", ".md")}}
for name in ("schemas.py", "state.py", "config.py", "report/templates.py", "report/pdf_renderer.py"):
    summary["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
def save():
    (suite / "e2e_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

generate = report_writer._invoke
def observed_generate(model, prompt, payload):
    entry = {"operation": payload.get("operation", "chapter" if "chapter" in payload else "summary"),
             "version": (payload.get("quality_feedback") or {}).get("report_version", 0) + 1,
             "chapter": payload.get("chapter", 0), "feedback_present": bool(payload.get("quality_feedback")),
             "previous_block": payload.get("previous_block"), "violations": payload.get("violations", [])}
    summary["generation_requests"].append(entry)
    save()
    return generate(model, prompt, payload)
report_writer._invoke = observed_generate

factory = app.build_graph
class ObservedGraph:
    def __init__(self, graph): self.graph = graph
    def get_graph(self): return self.graph.get_graph()
    def invoke(self, state, config):
        final = None
        for mode, packet in self.graph.stream(state, config=config, stream_mode=["updates", "values"]):
            if mode == "values":
                final = packet
                continue
            for node, update in packet.items():
                if not isinstance(update, dict): continue
                event = {"sequence": len(summary["events"]) + 1, "node": node,
                         "elapsed_seconds": round(time.monotonic() - start, 2)}
                for key in ("report_version", "execution_status", "decision", "total_score", "search_round"):
                    if key in update: event[key] = update[key]
                if update.get("current_startup"): event["company"] = update["current_startup"]["name"]
                if "candidates" in update: event["candidates"] = [c["name"] for c in update["candidates"]]
                if "tasks" in update: event["tasks"] = [{"kind": t["kind"], "target_fields": t["target_fields"]} for t in update["tasks"]]
                if isinstance(update.get("worker_results"), dict) and update["worker_results"]:
                    event["workers"] = [{k: result.get(k) for k in ("task_id", "status", "attempts", "error_code")}
                                        for result in update["worker_results"].values()]
                if update.get("quality_result"):
                    verdict = QualityVerdict.model_validate(update["quality_result"])
                    event["quality"] = verdict.model_dump()
                    event["passed"] = verdict.passed
                for key in ("report_draft_ref", "report_preview_ref", "report_path"):
                    if update.get(key): event[key] = str(Path(update[key]).relative_to(ROOT))
                event["log"] = update.get("log", [])
                summary["events"].append(event)
                save()
                print(json.dumps({k: event[k] for k in ("sequence", "node", "elapsed_seconds", "company", "report_version", "passed", "execution_status") if k in event}, ensure_ascii=False), flush=True)
        return final
app.build_graph = lambda *args, **kwargs: ObservedGraph(factory(*args, **kwargs))
sys.argv = ["app.py", "--seed-only"]
print("E2E_SUMMARY=" + str(suite / "e2e_summary.json"), flush=True)
try:
    if replay:
        from langgraph.graph import StateGraph, START, END
        from graph.routing import route_after_quality
        from state import InvestState
        from agents import report_quality
        from tools.artifacts import write_json
        state = json.loads(replay.read_text())
        summary["replayed_analysis"] = str(replay.relative_to(ROOT))
        summary["replayed_analysis_sha256"] = hashlib.sha256(replay.read_bytes()).hexdigest()
        directory = config.OUTPUT_DIR / "runs" / uuid4().hex
        state.update(run_id=directory.name, report_version=0, quality_result=None,
                     report_draft_ref=None, report_preview_ref=None, report_path=None, execution_status="running")
        evidence = json.loads((replay.parent / "evidence_index.json").read_text())
        write_json(directory / "evidence" / "replay.json", evidence)
        state["evidence_index_ref"] = str(directory / "evidence_index.json")
        graph = StateGraph(InvestState)
        for name, function in (("report_draft", report_writer.draft), ("report_quality", report_quality.run),
                               ("report_finalize", report_writer.finalize), ("quality_failed", report_quality.failed)):
            graph.add_node(name, function)
        graph.add_edge(START, "report_draft")
        graph.add_edge("report_draft", "report_quality")
        graph.add_conditional_edges("report_quality", route_after_quality, ["report_draft", "report_finalize", "quality_failed"])
        graph.add_edge("report_finalize", END)
        graph.add_edge("quality_failed", END)
        final = ObservedGraph(graph.compile()).invoke(state, {"recursion_limit": config.RECURSION_LIMIT})
        write_json(directory / "run_log.json", final)
        code = 0 if final["execution_status"] == "complete" else 1
    else:
        code = app.main()
    summary["application_exit_code"] = code
    run_file = next(config.OUTPUT_DIR.glob("runs/*/run_log.json"))
    final = json.loads(run_file.read_text())
    summary.update(run_id=final["run_id"], final_status=final["execution_status"],
                   final_version=final.get("report_version"), report_path=final.get("report_path"))
    def check(name, result): summary["assertions"].append({"name": name, "passed": bool(result)})
    events = summary["events"]
    versions = [e["report_version"] for e in events if e["node"] == "report_draft"]
    if not replay:
        check("실제 탐색·계획·worker·집계·투자 판단 노드 실행", all(any(e["node"] == n for e in events) for n in ("discover", "orchestrator", "research_worker", "synthesizer", "judge")))
    else:
        check("기존 실제 분석·근거를 재사용한 보고서 구간 실행", summary["replayed_analysis_sha256"] == hashlib.sha256(replay.read_bytes()).hexdigest())
    check("보고서 버전 1부터 연속 증가·최대 3개", versions == list(range(1, len(versions) + 1)) and 1 <= len(versions) <= 3)
    check("보고서 생성 뒤마다 품질 평가 실행", len(versions) == sum(e["node"] == "report_quality" for e in events))
    first_draft = next(i for i, e in enumerate(events) if e["node"] == "report_draft")
    check("품질 루프 중 기업 재탐색·재조사 없음", all(e["node"] not in ("discover", "select", "research_worker", "synthesizer", "judge") for e in events[first_draft:]))
    directory = run_file.parent
    drafts = [json.loads(p.read_text()) for p in sorted(directory.glob("draft_v*.json"))]
    repairs = [item for draft in drafts for item in draft.get("repair_log", [])]
    check("대체된 본문 블록에 근거 없는 기존 주장 잔존 없음", all(item["after"]["text"] == report_writer._fallback_block().text
          for item in repairs if item["chapter"] > 0 and item["action"] == "fallback"))
    check("블록 수정 요청에 이전 블록·실패 위치 전달", all(request["previous_block"] and request["violations"]
          for request in summary["generation_requests"] if request["operation"] == "repair_block"))
    check("다음 버전에서 전체 장 재생성 없음", all(request["operation"] != "chapter"
          for request in summary["generation_requests"] if request["version"] > 1))
    preserved = True
    for prior, current in zip(drafts, drafts[1:]):
        if "report" not in prior or "report" not in current: continue
        changed = {(r["chapter"], r["block"]) for r in current.get("repair_log", [])}
        for chapter in prior["report"]["chapters"]:
            for index, block in enumerate(chapter["blocks"], 1):
                if (chapter["number"], index) not in changed and block["subtitle"] != "분석의 한계":
                    preserved &= block == current["report"]["chapters"][chapter["number"] - 1]["blocks"][index - 1]
    check("이전 보고서의 정상 블록 보존", preserved)
    if final["execution_status"] == "complete":
        from pypdf import PdfReader
        pdf = Path(final["report_path"])
        check("통과한 동일 PDF만 확정", pdf.read_bytes() == Path(final["report_preview_ref"]).read_bytes())
        check(f"최종 PDF {MAX_PAGES}쪽 이하", len(PdfReader(pdf).pages) <= MAX_PAGES)
        check("최종 평가 모든 조건 통과", QualityVerdict.model_validate(final["quality_result"]).passed)
    else:
        check("미달 종료 시 최종 PDF 없음", final["report_path"] is None and not list(directory.glob("report_v*.pdf")))
        check("미달 평가와 실패 사유 보존", bool(list(directory.glob("quality_v*.json"))))
    summary["test_passed"] = all(a["passed"] for a in summary["assertions"])
except Exception as error:
    summary.update(final_status="execution_error", error_code=type(error).__name__, test_passed=False)
    # Avoid storing raw provider exceptions, request URLs or credential-bearing text.
finally:
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["elapsed_seconds"] = round(time.monotonic() - start, 2)
    save()
    print(json.dumps({k: summary.get(k) for k in ("final_status", "final_version", "test_passed", "elapsed_seconds")}, ensure_ascii=False), flush=True)
raise SystemExit(0 if summary["test_passed"] else 1)
