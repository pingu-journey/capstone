import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from config import DATA_DIR, MAX_CANDIDATES_PER_ROUND, OUTPUT_DIR, ROOT
from schemas import Candidate, Segment
from tools.llm import get_judge_llm, get_llm
from tools.web_search import search


QUERY_TEMPLATES = {
    "demand_forecasting": [
        "AI 전력 수요 예측 스타트업 투자 유치",
        "AI electricity demand forecasting startup raises Series",
    ],
    "generation_vpp": [
        "재생에너지 발전량 예측 VPP 스타트업 투자",
        "virtual power plant AI startup funding round",
    ],
    "ess_operation": [
        "ESS 배터리 운영 최적화 AI 스타트업 투자",
        "battery storage optimization AI startup Series A",
    ],
    "grid_management": [
        "AI 전력망 관리 스타트업 투자 유치",
        "AI grid management software startup raises",
    ],
}
PROMPT = (ROOT / "prompts" / "discovery.md").read_text(encoding="utf-8")


class CandidateBatch(BaseModel):
    candidates: list[Candidate]


class Eligibility(BaseModel):
    listed: bool
    exited: bool
    stage: Literal["Seed", "Series A", "Series B", "Series C", "Unknown"]
    eligible: bool
    priority: float = Field(ge=1, le=5)


def _invoke_with_retry(llm, schema, prompt: str):
    structured = llm.with_structured_output(schema)
    error = None
    for _ in range(2):
        try:
            return structured.invoke(prompt)
        except Exception as exc:
            error = exc
    raise RuntimeError(f"구조화 출력 검증에 실패했습니다: {error}")


def _queries(round_number: int) -> list[tuple[Segment, str]]:
    items = [
        (segment, query)
        for segment, queries in QUERY_TEMPLATES.items()
        for query in queries
    ]
    offset = (round_number * 2) % len(items)
    return items[offset:] + items[:offset]


def _seed_candidates(evaluated: set[str]) -> list[dict]:
    seeds = json.loads((DATA_DIR / "seed_candidates.json").read_text(encoding="utf-8"))
    return [
        Candidate.model_validate(item).model_dump()
        for item in seeds
        if item["name"].casefold() not in evaluated
    ][:MAX_CANDIDATES_PER_ROUND]


def _make_web_ref(result: dict, startup: str, run_date: str) -> dict:
    from tools.references import make_web_ref

    return make_web_ref(result, startup, "discovery", run_date)


def _save_round(candidates: list[dict], round_number: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"discovery_round{round_number}.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(state):
    evaluated = {name.casefold() for name in state.get("evaluated", [])}
    round_number = state.get("search_round", 0) + 1
    run_date = state["run_date"]

    if os.getenv("SEED_ONLY") == "1":
        candidates = _seed_candidates(evaluated)
        _save_round(candidates, round_number)
        return {
            "candidates": candidates,
            "search_round": round_number,
            "references": [],
            "log": [f"[discover] 시드 후보 {len(candidates)}곳 로드"],
        }

    found: list[Candidate] = []
    results_by_url: dict[str, dict] = {}
    errors: list[str] = []
    llm = None
    for segment, query in _queries(round_number - 1):
        try:
            results = search(query)
            results_by_url.update(
                (result["url"], result) for result in results if result.get("url")
            )
            llm = llm or get_llm()
            prompt = PROMPT.format(
                task=f"검색 결과에서 {segment} 세그먼트 후보를 추출하세요.",
                payload=json.dumps(results, ensure_ascii=False),
            )
            found.extend(_invoke_with_retry(llm, CandidateBatch, prompt).candidates)
        except Exception as exc:
            errors.append(f"{query}: {exc}")

    accepted: dict[str, Candidate] = {}
    judge = None
    for candidate in found:
        key = candidate.name.casefold()
        if key in evaluated or key in accepted:
            continue
        try:
            results = search(f'"{candidate.name}" IPO OR 상장 OR acquired OR 인수')
            results_by_url.update(
                (result["url"], result) for result in results if result.get("url")
            )
            judge = judge or get_judge_llm()
            prompt = PROMPT.format(
                task=(
                    "후보의 상장·Exit 여부와 투자 단계를 검증하고, "
                    "도메인 적합성·공개 정보 충분성을 1~5점으로 평가하세요. "
                    "불확실하면 eligible=false로 판정하세요."
                ),
                payload=json.dumps(
                    {"candidate": candidate.model_dump(), "results": results},
                    ensure_ascii=False,
                ),
            )
            check = _invoke_with_retry(judge, Eligibility, prompt)
        except Exception as exc:
            errors.append(f"{candidate.name}: {exc}")
            continue
        if check.eligible and not check.listed and not check.exited and check.stage != "Unknown":
            candidate.stage = check.stage
            candidate.priority = check.priority
            candidate.evidence_urls = list(
                dict.fromkeys(
                    candidate.evidence_urls
                    + [result["url"] for result in results if result.get("url")]
                )
            )
            accepted[key] = candidate

    selected = sorted(
        accepted.values(), key=lambda candidate: candidate.priority, reverse=True
    )[:MAX_CANDIDATES_PER_ROUND]
    fallback = not selected
    candidates = [candidate.model_dump() for candidate in selected]
    if fallback:
        candidates = _seed_candidates(evaluated)

    references = [
        _make_web_ref(results_by_url[url], candidate["name"], run_date)
        for candidate in candidates
        for url in candidate.get("evidence_urls", [])
        if url in results_by_url
    ]
    _save_round(candidates, round_number)
    detail = " (fallback)" if fallback else ""
    if errors:
        detail += f" (오류 {len(errors)}건)"
    return {
        "candidates": candidates,
        "search_round": round_number,
        "references": references,
        "log": [f"[discover] 후보 {len(candidates)}곳 발굴{detail}"],
    }
