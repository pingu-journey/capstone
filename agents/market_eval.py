"""Advanced RAG 근거로 시장성을 평가하는 에이전트 노드."""

from config import ROOT
from rag.advanced_rag import search
from schemas import MarketAnalysis, SEGMENT_KO
from state import InvestState
from tools.llm import get_llm
from tools.references import make_doc_ref


PROMPT_PATH = ROOT / "prompts" / "market_eval.md"
COUNTRY_KO = {
    "KR": "한국",
    "US": "미국",
    "GLOBAL": "글로벌",
}


def _questions(country: str, segment: str) -> list[str]:
    segment_ko = SEGMENT_KO[segment]
    country_ko = COUNTRY_KO.get(country, country)
    return [
        f"{segment_ko} 시장 규모와 전망 수치",
        f"{segment_ko} 시장 성장률과 성장 요인",
        f"{segment_ko} 수요처와 도입 수요",
        f"{country_ko} {segment_ko} 관련 정책·규제 환경과 리스크",
    ]


def _collect_context(country: str, segment: str) -> tuple[str, list[str]]:
    blocks = []
    doc_ids = []
    for question_number, question in enumerate(_questions(country, segment), start=1):
        results = search(question, country, segment, k=5)
        evidence = []
        for result in results:
            doc_id = result["doc_id"]
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
            evidence.append(
                f"[source_id={doc_id}, page={result['page']}]\n{result['text']}"
            )
        blocks.append(
            f"질문 {question_number}: {question}\n" + "\n\n".join(evidence)
        )
    return "\n\n---\n\n".join(blocks), doc_ids


def _validate_figures(analysis: MarketAnalysis, allowed_source_ids: set[str]) -> None:
    for figure in analysis.market_size:
        if not figure.year.strip():
            raise ValueError(f"기준 연도 누락: {figure.metric}")
        if figure.source_id not in allowed_source_ids:
            raise ValueError(
                f"허용되지 않은 source_id: {figure.source_id} "
                f"(허용: {sorted(allowed_source_ids)})"
            )


def _fallback(country: str, segment: str, sources: list[dict]) -> MarketAnalysis:
    return MarketAnalysis(
        segment=segment,
        country=country,
        market_size=[],
        growth="공개 정보 부족",
        demand_drivers=[],
        customers=[],
        policy_risks=["공개 정보 부족"],
        summary="검색 근거를 구조화된 시장 분석으로 변환하지 못했습니다.",
        sources=sources,
    )


def _copy_sources(sources: list[dict], startup: str) -> list[dict]:
    return [{**source, "startup": startup} for source in sources]


def run(state: InvestState) -> dict:
    startup = state.get("current_startup")
    if not startup:
        raise ValueError("market_eval에는 current_startup이 필요합니다.")

    name = startup["name"]
    country = startup["country"].upper()
    segment = startup["segment"]
    cache_key = f"{country}|{segment}"
    cached = (state.get("market_cache") or {}).get(cache_key)

    if cached:
        analysis = MarketAnalysis.model_validate(cached)
        references = _copy_sources(analysis.sources, name)
        analysis.sources = references
        return {
            "market_analysis": analysis.model_dump(),
            "market_cache": {cache_key: analysis.model_dump()},
            "references": references,
            "log": [f"[market_eval] {cache_key} 시장 분석 캐시 재사용"],
        }

    context, doc_ids = _collect_context(country, segment)
    references = [
        make_doc_ref(doc_id, startup=name, used_by="market_eval")
        for doc_id in doc_ids
    ]
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        startup=name,
        country=country,
        segment=segment,
        segment_ko=SEGMENT_KO[segment],
        evidence=context or "검색 근거 없음",
    )

    # MarketAnalysis에는 기본값이 있는 필드가 포함되어 있어 OpenAI의 strict
    # JSON Schema보다 function calling 방식이 호환성이 높다.
    structured_llm = get_llm().with_structured_output(
        MarketAnalysis,
        method="function_calling",
    )
    analysis = None
    last_error = None
    for attempt in range(2):
        try:
            request = prompt
            if attempt:
                request += (
                    "\n\n이전 출력이 검증에 실패했습니다. 모든 시장 수치에 제공된 "
                    "source_id와 기준 연도를 넣어 다시 작성하세요."
                )
            analysis = structured_llm.invoke(request)
            analysis.segment = segment
            analysis.country = country
            analysis.sources = references
            _validate_figures(analysis, set(doc_ids))
            break
        except Exception as error:
            last_error = error
            analysis = None

    if analysis is None:
        analysis = _fallback(country, segment, references)
        status = f"구조화 출력 실패: {type(last_error).__name__}"
    else:
        status = f"근거 문서 {len(doc_ids)}개로 시장 분석 완료"

    dumped = analysis.model_dump()
    return {
        "market_analysis": dumped,
        "market_cache": {cache_key: dumped},
        "references": references,
        "log": [f"[market_eval] {name} {status}"],
    }
