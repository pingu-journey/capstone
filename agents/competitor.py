"""경쟁사 비교 에이전트 (명세서 6.5, 웹서치)."""
import json
import re

from agents.tech_summary import SEGMENT_EN, company_label
from config import ROOT
from schemas import SEGMENT_KO, Competitor, CompetitorAnalysis
from tools.llm import get_llm
from tools.references import make_web_ref
from tools.web_search import search


USED_BY = "competitor"
INSUFFICIENT = "공개 정보 부족"
PROMPT_PATH = ROOT / "prompts" / "competitor.md"
MIN_COMPETITORS = 3
CITATION = re.compile(r"\[(\d+)\]")
VERIFIED_FIELDS = ("traction", "moat", "competition_risks")

# insufficient 항목명 → rubric.yaml 평가 항목 id (D가 '분석의 한계'에 사용)
INSUFFICIENT_TO_RUBRIC = {
    "Traction": "traction",
    "진입장벽": "moat",
    "경쟁사 비교": "moat",
}


class CompetitorOutput(CompetitorAnalysis):
    insufficient: list[str] = []  # schemas.py에 필드가 추가되기 전까지 여기서 확장


class CompetitorDraftItem(Competitor):
    status_source: int | None = None


class CompetitorDraft(CompetitorAnalysis):
    competitors: list[CompetitorDraftItem]
    used_sources: list[int] = []


def split_citations(item: str, n_sources: int) -> tuple[str, list[int]]:
    """'문장 [2][9]' → ('문장', [2, 9]). 범위 밖 번호는 버린다."""
    nums = [int(n) for n in CITATION.findall(item) if 1 <= int(n) <= n_sources]
    text = re.sub(r"\s{2,}", " ", CITATION.sub("", item)).strip()
    return text, nums


def verify_items(items: list[str], n_sources: int) -> tuple[list[str], set[int]]:
    """유효한 근거 번호가 있는 항목만 [n]을 지워 남기고, 사용한 번호를 함께 반환."""
    kept, used = [], set()
    for item in items:
        text, nums = split_citations(item, n_sources)
        if nums and text and text != INSUFFICIENT:
            kept.append(text)
            used.update(nums)
    return kept, used


def verify_status(status: str, source: int | None, n_sources: int) -> tuple[str, int | None]:
    """근거 번호가 없거나 범위 밖이면 status를 '공개 정보 부족'으로 바꾼다."""
    if source is not None and 1 <= source <= n_sources and status.strip():
        return status, source
    return INSUFFICIENT, None


def _insufficient(analysis: CompetitorAnalysis) -> list[str]:
    """비어 있는 필드로 부족 항목을 결정 (LLM 자기보고는 쓰지 않음)."""
    differentiation = analysis.differentiation.strip()
    checks = {
        "경쟁사 비교": len(analysis.competitors) < MIN_COMPETITORS
        or not differentiation or differentiation == INSUFFICIENT,
        "Traction": not analysis.traction,
        "진입장벽": not analysis.moat,
    }
    return [label for label, is_missing in checks.items() if is_missing]


def _fallback() -> CompetitorDraft:
    return CompetitorDraft(
        competitors=[], differentiation=INSUFFICIENT, moat=[], traction=[], competition_risks=[]
    )


def run(state: dict) -> dict:
    startup = state["current_startup"]
    name, run_date, country = startup["name"], state["run_date"], startup["country"]
    company = company_label(startup)
    segment_ko = SEGMENT_KO[startup["segment"]]

    seen, results = set(), []
    for query in [
        f"{company} competitors",
        f"{SEGMENT_EN[startup['segment']]} startups {country}",
        f"{segment_ko} 기업",
        f"{company} 고객 OR 계약 OR 파트너십 OR customers",
    ]:
        for result in search(query):
            if result["url"] and result["url"] not in seen:
                seen.add(result["url"])
                results.append(result)

    tech = state.get("tech_summary") or {}
    market = state.get("market_analysis") or {}
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        company_info=(
            f"{name} / 국가 {country} / 단계 {startup['stage']} / 세그먼트 {segment_ko} / "
            f"투자 {startup['funding']}"
        ),
        tech_summary=json.dumps(
            {k: tech.get(k) for k in ("core_tech", "products", "trl", "performance", "strengths", "weaknesses")},
            ensure_ascii=False,
        ),
        market_analysis=json.dumps(
            {k: market.get(k) for k in ("summary", "growth", "customers", "demand_drivers")},
            ensure_ascii=False,
        ),
        evidence="\n\n".join(
            f"[{i}] {r['title']} — {r['url']}\n{r['content']}" for i, r in enumerate(results, start=1)
        ) or INSUFFICIENT,
    )
    llm = get_llm().with_structured_output(CompetitorDraft)
    draft = None
    for _ in range(2):  # 검증 실패 시 1회 재시도
        try:
            draft = llm.invoke(prompt)
            break
        except Exception:
            continue
    failed = draft is None
    draft = draft or _fallback()

    n_sources = len(results)
    cited = set(draft.used_sources)
    competitors = []
    for c in draft.competitors:
        if c.name.strip().casefold() == name.strip().casefold():  # 대상 기업 자신 제외
            continue
        status, source = verify_status(c.status, c.status_source, n_sources)
        if source:
            cited.add(source)
        competitors.append(Competitor(**c.model_dump(exclude={"status", "status_source"}), status=status))
    verified = {}
    for field in VERIFIED_FIELDS:
        verified[field], nums = verify_items(getattr(draft, field), n_sources)
        cited |= nums
    analysis = CompetitorOutput(
        competitors=competitors, differentiation=draft.differentiation, **verified
    )
    analysis.insufficient = _insufficient(analysis)

    used = sorted(n for n in cited if 1 <= n <= n_sources)
    if not used:  # 비었거나 실패 → 넘긴 검색 결과 전부
        used = list(range(1, len(results) + 1))
    references = [make_web_ref(results[n - 1], name, USED_BY, run_date) for n in used]

    status = "생성 실패 → 기본값" if failed else f"경쟁사 {len(analysis.competitors)}곳"
    return {
        "competitor_analysis": analysis.model_dump(),
        "references": references,
        "log": [
            f"[competitor] {name} {status}, 부족 {analysis.insufficient or '없음'}, 근거 {len(references)}건"
        ],
    }
