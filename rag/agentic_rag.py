"""기술 요약용 Agentic RAG 서브그래프 (명세서 5.3)."""
import re
import sys
from datetime import date
from functools import lru_cache
from operator import add
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlparse

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from config import OUTPUT_DIR, RAG_MAX_REWRITES, ROOT
from schemas import SEGMENT_KO
from tools.llm import get_judge_llm, get_llm
from tools.references import make_doc_ref, make_web_ref
from tools.web_search import search


PROMPT_DIR = ROOT / "prompts"
USED_BY = "tech_summary"
LOG_PREFIX = "[tech_summary/agentic_rag]"
INSUFFICIENT = "공개 정보 부족"
TOP_K = 5
MIN_RELEVANT = 2
MAX_CONTEXT_DOCS = 5

# 전력 산업 전용 용어 (power·energy 같은 일반 단어는 동명 기업 오탐이 많아 제외)
INDUSTRY_TERMS = (
    "grid", "utility", "utilities", "load forecasting", "demand forecasting", "renewable",
    "solar", "wind", "energy storage", "battery storage", "electricity market", "power market",
    "virtual power plant", "demand response",
    "전력망", "전력시장", "발전량", "재생에너지", "가상발전소", "수요반응",
)
# 업계·경쟁사 탐색 쿼리용 넓은 목록 (특정 기업을 찾는 쿼리가 아니라 일반 단어도 허용)
BROAD_TERMS = INDUSTRY_TERMS + (
    "energy", "electricity", "power", "forecast", "forecasting",
    "에너지", "전력", "유틸리티", "수요 예측",
)
INDUSTRY_ACRONYMS = ("VPP", "DER", "ESS")  # 대소문자 구분 ("der", "ISO 9001" 오탐 방지)


def _term_pattern(terms: tuple[str, ...]) -> re.Pattern:
    # 영문은 앞뒤에 영문자가 없을 때만 매칭 (복수형 s 허용, "ESS를"처럼 한글 조사는 허용)
    return re.compile("|".join(
        rf"(?<![a-z]){re.escape(t)}s?(?![a-z])" if t.isascii() else re.escape(t) for t in terms
    ))


_STRICT_PATTERN = _term_pattern(INDUSTRY_TERMS)
_BROAD_PATTERN = _term_pattern(BROAD_TERMS)
_ACRONYM_PATTERN = re.compile("|".join(rf"(?<![A-Za-z]){t}s?(?![A-Za-z])" for t in INDUSTRY_ACRONYMS))

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
    entity: dict | None
    use_rag: bool
    graded_ids: list[str]  # 관련·무관 상관없이 판정을 마친 청크 ID
    path: str
    log: Annotated[list[str], add]


class GradeOutput(BaseModel):
    verdicts: list[Literal["yes", "no"]]


class GenerateOutput(BaseModel):
    answer: str
    used_sources: list[int] = []


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def has_industry_term(text: str, broad: bool = False) -> bool:
    pattern = _BROAD_PATTERN if broad else _STRICT_PATTERN
    return bool(pattern.search(text.lower()) or _ACRONYM_PATTERN.search(text))


def filter_web_results(
    results: list[dict], entity: dict | None = None, require_terms: bool = True
) -> tuple[list[dict], list[dict]]:
    """웹 결과 사전 필터. (유지, 제외)를 반환한다.

    - entity + require_terms: 기술 질문 기업 쿼리. 도메인 일치 또는 (기업명 + 엄격 목록 용어 1개 이상)
    - entity + not require_terms: 팀·경쟁사 기업 쿼리. 도메인 일치 또는 기업명 포함 (동명 기업은 프롬프트로 차단)
    - entity 없음: 업계·경쟁사 탐색 쿼리. 넓은 목록 용어 1개 이상
    제목이 없거나 http(s)가 아닌 결과는 make_web_ref가 거부하므로 항상 제외한다.
    """
    home = _domain(entity.get("homepage") or "") if entity else ""
    kept, dropped = [], []
    for result in results:
        if not (result.get("title") or "").strip() or urlparse(result.get("url") or "").scheme not in ("http", "https"):
            dropped.append(result)
            continue
        text = f"{result.get('title') or ''} {result.get('content') or ''}"
        if entity:
            domain = _domain(result["url"])
            same_site = bool(home) and (domain == home or domain.endswith("." + home))
            named = entity["name"].lower() in text.lower()
            ok = same_site or (named and (not require_terms or has_industry_term(text)))
        else:
            ok = has_industry_term(text, broad=True)
        (kept if ok else dropped).append(result)
    return kept, dropped


def filter_log(prefix: str, dropped: list[dict]) -> list[str]:
    if not dropped:
        return []
    titles = ", ".join((r.get("title") or r["url"])[:40] for r in dropped)
    return [f"{prefix} web 필터 제외 {len(dropped)}건: {titles}"]


def _prompt(name: str, **values) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").format(**values)


def _get_tech_collection():
    from rag.ingest import get_collection

    return get_collection("tech_docs")


@lru_cache(maxsize=1)
def _get_embedder():
    from rag.embeddings import get_embedder

    return get_embedder()


def _tech_docs_available() -> bool:
    try:
        _get_embedder()
        return _get_tech_collection().count() > 0
    except Exception:
        return False


def retrieve(state: RagState) -> dict:
    embedding = _get_embedder().embed_query(state["query"])
    res = _get_tech_collection().query(query_embeddings=[embedding], n_results=TOP_K)
    docs = [
        {"id": cid, "text": text, "doc_id": meta["doc_id"], "page": meta["page"], "distance": dist}
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]
    return {
        "docs": docs,
        "log": [f"{LOG_PREFIX} retrieve query='{state['query']}' → {len(docs)}건"],
    }


def _grade_docs(question: str, docs: list[dict]) -> list[str] | None:
    chunks = "\n\n".join(f"[{i}] {d['text']}" for i, d in enumerate(docs, start=1))
    prompt = _prompt("agentic_rag_grade", question=question, chunks=chunks)
    llm = get_judge_llm().with_structured_output(GradeOutput)
    for _ in range(2):  # 검증 실패 시 1회 재시도
        try:
            verdicts = llm.invoke(prompt).verdicts
            if len(verdicts) == len(docs):
                return verdicts
        except Exception:
            continue
    return None


def grade(state: RagState) -> dict:
    graded = set(state.get("graded_ids") or [])
    new_docs = [d for d in state.get("docs") or [] if d["id"] not in graded]
    relevant = list(state.get("relevant_docs") or [])
    log = []
    if new_docs:
        verdicts = _grade_docs(state["question"], new_docs)
        if verdicts is None:
            verdicts = ["no"] * len(new_docs)
            log.append(f"{LOG_PREFIX} grade failed → treated as irrelevant")
        relevant += [d for d, v in zip(new_docs, verdicts) if v == "yes"]
    log.append(f"{LOG_PREFIX} grade 신규 {len(new_docs)}건 판정 → 관련 누적 {len(relevant)}건")
    return {
        "relevant_docs": relevant,
        "graded_ids": sorted(graded | {d["id"] for d in new_docs}),
        "log": log,
    }


def route_after_grade(state: RagState) -> str:
    if len(state.get("relevant_docs") or []) >= MIN_RELEVANT:
        return "generate"
    if state.get("rewrite_count", 0) < RAG_MAX_REWRITES:
        return "rewrite"
    return "web_search"


def rewrite(state: RagState) -> dict:
    prompt = _prompt("agentic_rag_rewrite", question=state["question"], query=state["query"])
    try:
        query = get_llm().invoke(prompt).content.strip() or state["query"]
    except Exception:
        query = state["query"]
    count = state.get("rewrite_count", 0) + 1
    return {
        "query": query,
        "rewrite_count": count,
        "log": [f"{LOG_PREFIX} rewrite #{count} → '{query}'"],
    }


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
    kept, dropped = filter_web_results(results, state.get("entity"))
    return {
        "web_results": kept,
        "log": [f"{LOG_PREFIX} web_search 쿼리 {len(queries)}개 → 결과 {len(kept)}건"]
        + filter_log(LOG_PREFIX, dropped),
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
    # 검색 점수 높은 순(Chroma 거리 오름차순) 최대 5개
    relevant_docs = sorted(state.get("relevant_docs") or [], key=lambda d: d["distance"])
    relevant_docs = relevant_docs[:MAX_CONTEXT_DOCS]
    web_results = state.get("web_results") or []
    evidence = [("doc", d) for d in relevant_docs] + [("web", r) for r in web_results]
    path = "rag+web" if relevant_docs and web_results else ("rag" if relevant_docs else "web")

    if not evidence:
        answer, used = INSUFFICIENT, []
    else:
        entity = state.get("entity")
        entity_info = (
            f"대상 기업: {entity['name']}, 분야: {SEGMENT_KO[entity['segment']]}, "
            f"홈페이지: {entity.get('homepage') or INSUFFICIENT}. "
            "이름이 같아도 다른 분야의 회사 자료는 사용하지 마라."
            if entity else ""
        )
        prompt = _prompt(
            "agentic_rag_generate",
            company=state["company"],
            entity_info=entity_info,
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
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate)
    g.add_conditional_edges(
        START,
        lambda s: "retrieve" if s.get("use_rag") else "web_search",
        {"retrieve": "retrieve", "web_search": "web_search"},
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", END)
    return g.compile()


def run(
    question: str,
    company: str,
    startup: str,
    run_date: str | None = None,
    web_queries: list[str] | None = None,
    entity: dict | None = None,
) -> dict:
    use_rag = _tech_docs_available()
    log = [] if use_rag else [f"{LOG_PREFIX} tech_docs unavailable → web only"]
    final = build_graph().invoke({
        "question": question,
        "company": company,
        "startup": startup,
        "run_date": run_date or date.today().isoformat(),
        "web_queries": web_queries,
        "entity": entity,
        "use_rag": use_rag,
        "query": question,
        "rewrite_count": 0,
        "relevant_docs": [],
        "graded_ids": [],
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
    mmd_path = OUTPUT_DIR / "agentic_rag.md"
    mmd_path.write_text(build_graph().get_graph().draw_mermaid(), encoding="utf-8")
    print(f"서브그래프: {mmd_path}")
    if len(sys.argv) >= 3:  # python -m rag.agentic_rag "질문" 기업명
        result = run(sys.argv[1], sys.argv[2], sys.argv[2])
        print("\n".join(result["log"]))
        print(result["answer"])
