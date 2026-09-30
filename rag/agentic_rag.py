"""기술 요약용 Agentic RAG 서브그래프 (명세서 5.3)."""
import re
import sys
from datetime import date
from functools import lru_cache
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from config import OUTPUT_DIR, ROOT
from tools.llm import get_llm
from tools.references import make_doc_ref, make_web_ref
from tools.web_search import search


PROMPT_DIR = ROOT / "prompts"
USED_BY = "tech_summary"
LOG_PREFIX = "[tech_summary/agentic_rag]"
INSUFFICIENT = "공개 정보 부족"

DEFAULT_WEB_QUERIES = [
    "{company} technology",
    "{company} 특허 OR patent",
    "{company} product performance",
]


class RagState(TypedDict, total=False):
    question: str
    company: str
    query: str
    docs: list[dict]
    relevant_docs: list[dict]
    rewrite_count: int
    web_results: list[dict]
    answer: str
    sources: list[dict]
    # run() 반환·Reference 생성용
    startup: str
    run_date: str
    web_queries: list[str] | None
    path: str
    log: Annotated[list[str], add]


class GenerateOutput(BaseModel):
    answer: str
    used_sources: list[int] = []


def _tech_docs_available() -> bool:
    try:
        from rag.embeddings import get_embedder  # noqa: F401
        from rag.ingest import get_collection

        return get_collection("tech_docs").count() > 0
    except Exception:
        return False


def web_search(state: RagState) -> dict:
    queries = state.get("web_queries") or [
        q.format(company=state["company"]) for q in DEFAULT_WEB_QUERIES
    ]
    seen, results = set(), []
    for query in queries:
        for result in search(query):
            if result["url"] and result["url"] not in seen:
                seen.add(result["url"])
                results.append(result)
    return {
        "web_results": results,
        "log": [f"{LOG_PREFIX} web_search 쿼리 {len(queries)}개 → 결과 {len(results)}건"],
    }


def _format_evidence(evidence: list[tuple[str, dict]]) -> str:
    lines = []
    for i, (kind, item) in enumerate(evidence, start=1):
        if kind == "doc":
            lines.append(f"[{i}] (문서 {item['doc_id']} p.{item['page']}) {item['text']}")
        else:
            lines.append(f"[{i}] (웹) {item['title']} — {item['url']}\n{item['content']}")
    return "\n\n".join(lines)


def _to_reference(kind: str, item: dict, state: RagState) -> dict:
    if kind == "doc":
        return make_doc_ref(item["doc_id"], state["startup"], USED_BY)
    return make_web_ref(item, state["startup"], USED_BY, state["run_date"])


def generate(state: RagState) -> dict:
    relevant_docs = state.get("relevant_docs") or []
    web_results = state.get("web_results") or []
    evidence = [("doc", d) for d in relevant_docs] + [("web", r) for r in web_results]
    path = "rag+web" if relevant_docs and web_results else ("rag" if relevant_docs else "web")

    if not evidence:
        answer, used = INSUFFICIENT, []
    else:
        template = (PROMPT_DIR / "agentic_rag_generate.md").read_text(encoding="utf-8")
        prompt = template.format(
            company=state["company"],
            question=state["question"],
            evidence=_format_evidence(evidence),
        )
        llm = get_llm().with_structured_output(GenerateOutput)
        output = None
        for _ in range(2):  # 검증 실패 시 1회 재시도
            try:
                output = llm.invoke(prompt)
                break
            except Exception:
                continue
        answer = output.answer if output else INSUFFICIENT
        # 본문 [n] 인용이 기준, used_sources는 보조
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        cited |= set(output.used_sources if output else [])
        used = sorted(n for n in cited if 1 <= n <= len(evidence))
        if not used:  # 비었거나 파싱 실패 → 넘긴 근거 전부
            used = list(range(1, len(evidence) + 1))

    sources, seen = [], set()
    for n in used:
        ref = _to_reference(*evidence[n - 1], state)
        if ref["id"] not in seen:
            seen.add(ref["id"])
            sources.append(ref)

    rewrites = state.get("rewrite_count", 0)
    return {
        "answer": answer,
        "sources": sources,
        "path": path,
        "log": [f"{LOG_PREFIX} generate path={path}, rewrites={rewrites}, sources={len(sources)}"],
    }


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(RagState)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate)
    g.add_edge(START, "web_search")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", END)
    return g.compile()


def run(
    question: str,
    company: str,
    startup: str,
    run_date: str | None = None,
    web_queries: list[str] | None = None,
) -> dict:
    log = [] if _tech_docs_available() else [f"{LOG_PREFIX} tech_docs unavailable → web only"]
    final = build_graph().invoke({
        "question": question,
        "company": company,
        "startup": startup,
        "run_date": run_date or date.today().isoformat(),
        "web_queries": web_queries,
        "query": question,
        "rewrite_count": 0,
        "log": log,
    })
    return {
        "answer": final["answer"],
        "sources": final["sources"],
        "path": final["path"],
        "rewrites": final.get("rewrite_count", 0),
        "log": final["log"],
    }


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    mmd_path = OUTPUT_DIR / "agentic_rag.mmd"
    mmd_path.write_text(build_graph().get_graph().draw_mermaid(), encoding="utf-8")
    print(f"서브그래프: {mmd_path}")
    if len(sys.argv) >= 3:  # python -m rag.agentic_rag "질문" 기업명
        result = run(sys.argv[1], sys.argv[2], sys.argv[2])
        print("\n".join(result["log"]))
        print(result["answer"])
