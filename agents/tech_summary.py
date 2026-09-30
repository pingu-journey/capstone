"""기술 요약 에이전트 (명세서 6.3, Agentic RAG)."""
from urllib.parse import urlparse

from config import ROOT
from rag import agentic_rag
from schemas import SEGMENT_KO, TeamInfo, TechSummary
from tools.llm import get_llm
from tools.references import make_web_ref
from tools.web_search import search


USED_BY = "tech_summary"
INSUFFICIENT = "공개 정보 부족"
PROMPT_PATH = ROOT / "prompts" / "tech_summary.md"

SEGMENT_EN = {
    "demand_forecasting": "electricity demand forecasting",
    "generation_vpp": "renewable generation forecasting virtual power plant",
    "ess_operation": "battery energy storage operation",
    "grid_management": "power grid management",
}

# insufficient 항목명 → rubric.yaml 평가 항목 id (D가 '분석의 한계'에 사용)
INSUFFICIENT_TO_RUBRIC = {
    "팀 정보": "team",
    "TRL": "trl",
    "성능 지표": "tech_value",
    "확장성": "scalability",
    "핵심 기술": "tech_value",
    "제품": "tech_value",
    "강점·약점": "tech_value",
}


class TechSummaryDraft(TechSummary):
    used_team_sources: list[int] = []


def company_label(startup: dict) -> str:
    domain = urlparse(startup.get("homepage") or "").netloc.removeprefix("www.")
    return f"{startup['name']} ({domain})" if domain else startup["name"]


def _team_search(company: str, entity: dict) -> tuple[list[dict], list[str]]:
    seen, results = set(), []
    for query in [f"{company} founder CEO CTO", f"{company} 창업자 대표 이력"]:
        for result in search(query):
            if result["url"] and result["url"] not in seen:
                seen.add(result["url"])
                results.append(result)
    kept, dropped = agentic_rag.filter_web_results(results, entity)
    return kept, agentic_rag.filter_log("[tech_summary] 팀 검색", dropped)


def _fallback() -> TechSummaryDraft:
    return TechSummaryDraft(
        core_tech=INSUFFICIENT,
        products=[],
        trl=1,
        trl_rationale=INSUFFICIENT,
        performance=[],
        strengths=[],
        weaknesses=[],
        scalability=INSUFFICIENT,
        team=TeamInfo(founders=[], assessment=INSUFFICIENT),
    )


def _insufficient(summary: TechSummary) -> list[str]:
    """비어 있는 필드로 부족 항목을 결정 (LLM 자기보고는 쓰지 않음)."""
    def missing(text: str) -> bool:
        return not text.strip() or text.strip() == INSUFFICIENT

    checks = {
        "핵심 기술": missing(summary.core_tech),
        "제품": not summary.products,
        "TRL": missing(summary.trl_rationale),
        "성능 지표": not summary.performance,
        "강점·약점": not summary.strengths and not summary.weaknesses,
        "확장성": missing(summary.scalability),
        "팀 정보": not summary.team.founders,
    }
    return [label for label, is_missing in checks.items() if is_missing]


def run(state: dict) -> dict:
    startup = state["current_startup"]
    name, run_date = startup["name"], state["run_date"]
    company = company_label(startup)
    segment_ko = SEGMENT_KO[startup["segment"]]
    entity = {"name": name, "segment": startup["segment"], "homepage": startup.get("homepage")}

    tech = agentic_rag.run(
        f"{company}의 핵심 기술, 제품, 성능 지표는 무엇인가?", company, name, run_date,
        entity=entity,
    )
    industry_question = f"{segment_ko} 분야 AI 기술의 업계 수준과 상용화 단계(TRL 판단 근거)는?"
    industry = agentic_rag.run(
        industry_question, company, name, run_date,
        web_queries=[
            industry_question,
            f"{SEGMENT_EN[startup['segment']]} AI technology readiness commercialization",
        ],
    )
    team_results, team_log = _team_search(company, entity)

    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        company_info=(
            f"{name} / 국가 {startup['country']} / 단계 {startup['stage']} / "
            f"세그먼트 {segment_ko} / 홈페이지 {startup.get('homepage') or INSUFFICIENT}"
        ),
        tech_answer=tech["answer"],
        industry_answer=industry["answer"],
        team_evidence="\n\n".join(
            f"[{i}] {r['title']} — {r['url']}\n{r['content']}"
            for i, r in enumerate(team_results, start=1)
        ) or INSUFFICIENT,
    )
    llm = get_llm().with_structured_output(TechSummaryDraft)
    draft = None
    for _ in range(2):  # 검증 실패 시 1회 재시도
        try:
            draft = llm.invoke(prompt)
            break
        except Exception:
            continue
    failed = draft is None
    draft = draft or _fallback()

    summary = TechSummary(**draft.model_dump(exclude={"used_team_sources", "insufficient"}))
    for field in ("products", "performance", "strengths", "weaknesses"):  # ['공개 정보 부족'] → []
        setattr(summary, field, [x for x in getattr(summary, field) if x.strip() != INSUFFICIENT])
    insufficient = summary.insufficient = _insufficient(summary)

    used = [n for n in draft.used_team_sources if 1 <= n <= len(team_results)]
    if not used:  # 비었거나 실패 → 팀 검색 결과 전부
        used = list(range(1, len(team_results) + 1))
    references, seen = [], set()
    for ref in tech["sources"] + industry["sources"] + [
        make_web_ref(team_results[n - 1], name, USED_BY, run_date) for n in used
    ]:
        if ref["id"] not in seen:
            seen.add(ref["id"])
            references.append(ref)

    status = "생성 실패 → 기본값" if failed else f"TRL {summary.trl}"
    return {
        "tech_summary": summary.model_dump(),
        "references": references,
        "log": tech["log"] + industry["log"] + team_log + [
            f"[tech_summary] {name} {status}, 부족 {insufficient or '없음'}, 근거 {len(references)}건"
        ],
    }
