"""LLM은 항목별 채점, 코드는 가중 합산·탈락 조건·최종 판정을 담당한다."""

import json
import logging
from typing import Literal

import yaml
from httpx import HTTPError
from langchain_core.exceptions import OutputParserException
from openai import APIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config import INVEST_THRESHOLD, ROOT
from schemas import Candidate, CompetitorAnalysis, EvalRecord, ItemScore, MarketAnalysis, TechSummary
from tools.llm import get_judge_llm
from tools.references import filter_references


logger = logging.getLogger(__name__)
RUBRIC_PATH = ROOT / "config" / "rubric.yaml"
PROMPT_PATH = ROOT / "prompts" / "investment_judge.md"

ItemId = Literal[
    "team", "market_size", "market_demand", "traction", "moat", "trl",
    "tech_value", "regulation", "scalability", "finance", "deal_terms",
]
ITEM_IDS = {
    "team", "market_size", "market_demand", "traction", "moat", "trl",
    "tech_value", "regulation", "scalability", "finance", "deal_terms",
}
INSUFFICIENT_ITEM_IDS = {
    "tech_summary": {
        "TRL": "trl", "팀 정보": "team", "성능 지표": "tech_value",
        "확장성": "scalability",
    },
    "competitor_analysis": {
        "Traction": "traction", "진입장벽": "moat", "경쟁사 비교": "moat",
    },
}


class RubricItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)

    id: ItemId
    name: str = Field(min_length=1)
    weight: int = Field(strict=True, gt=0, le=100)
    knockout: bool = False
    basis: str = Field(min_length=1)
    input: str = Field(min_length=1)
    high: str = Field(min_length=1)
    mid: str = Field(min_length=1)
    low: str = Field(min_length=1)


class ScoreDraft(BaseModel):
    """가중치와 최종 판정은 LLM 출력에서 받지 않는다."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)

    item_id: ItemId
    score: int = Field(strict=True, ge=0, le=10)
    rationale: str = Field(min_length=1)
    insufficient_info: bool = Field(strict=True)
    source_ids: list[str]


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    items: list[ScoreDraft] = Field(min_length=11, max_length=11)


class JudgeServiceError(RuntimeError):
    """외부 API·파싱 오류의 원문을 traceback에 노출하지 않는 호출 경계 예외."""


def _invoke_judge(messages) -> JudgeOutput:
    llm = get_judge_llm().with_structured_output(JudgeOutput)
    try:
        return JudgeOutput.model_validate(llm.invoke(messages))
    except (APIError, HTTPError, OutputParserException, ValidationError):
        raise JudgeServiceError("투자 채점 API 호출 또는 응답 검증 실패: operation=judge") from None


def _validate_rubric(items: list) -> list[dict]:
    validated = [RubricItem.model_validate(item).model_dump() for item in items]
    if len(validated) != 11 or {item["id"] for item in validated} != ITEM_IDS:
        raise ValueError("루브릭에는 중복 없이 11개 평가 항목이 있어야 합니다.")
    if sum(item["weight"] for item in validated) != 100:
        raise ValueError("루브릭 가중치 합은 100이어야 합니다.")
    if {item["id"] for item in validated if item["knockout"]} != {"team", "regulation"}:
        raise ValueError("필수 탈락 항목은 창업팀 역량과 규제·정책 위험이어야 합니다.")
    return validated


def load_rubric() -> list[dict]:
    data = yaml.safe_load(RUBRIC_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("rubric.yaml에는 threshold와 items 목록이 있어야 합니다.")
    if type(data.get("threshold")) is not int or data["threshold"] != INVEST_THRESHOLD:
        raise ValueError("루브릭 threshold와 config.INVEST_THRESHOLD가 일치해야 합니다.")
    return _validate_rubric(data["items"])


def compute(scores: dict, rubric: list) -> tuple[float, str, str | None]:
    """정수 가중합으로 경계값을 비교하며, 반환할 때만 100점 척도로 변환한다."""
    items = _validate_rubric(rubric)
    if set(scores) != ITEM_IDS:
        raise ValueError("점수에는 중복 없이 11개 평가 항목이 있어야 합니다.")
    normalized = {}
    for item in items:
        entry = scores[item["id"]]
        score = entry["score"]
        if type(score) is not int or not 0 <= score <= 10:
            raise ValueError(f"항목 점수는 0~10 정수여야 합니다: {item['id']}")
        insufficient = entry.get("insufficient_info", False)
        if type(insufficient) is not bool:
            raise ValueError(f"정보 부족 여부는 bool이어야 합니다: {item['id']}")
        normalized[item["id"]] = 5 if insufficient else score
    weighted_sum = sum(normalized[item["id"]] * item["weight"] for item in items)
    knockout = next(
        (item["name"] for item in items
         if item["knockout"] and normalized[item["id"]] <= 3), None
    )
    decision = "투자" if weighted_sum >= INVEST_THRESHOLD * 10 and knockout is None else "보류"
    return weighted_sum / 10, decision, knockout


def run(state) -> dict:
    candidate = Candidate.model_validate(state.get("current_startup"))
    analysis = {
        "current_startup": candidate.model_dump(),
        "tech_summary": TechSummary.model_validate(state.get("tech_summary")).model_dump(),
        "market_analysis": MarketAnalysis.model_validate(state.get("market_analysis")).model_dump(),
        "competitor_analysis": CompetitorAnalysis.model_validate(
            state.get("competitor_analysis")
        ).model_dump(),
    }
    rubric = load_rubric()
    # CompetitorAnalysis 검증은 insufficient를 제거하므로 원본에서 읽는다.
    forced = {
        mapping[label]
        for key, mapping in INSUFFICIENT_ITEM_IDS.items()
        for label in state[key].get("insufficient", [])
        if label in mapping
    }
    refs = filter_references(state.get("references", []), candidate.name)
    available_sources = {ref["id"] for ref in refs}
    # 이전 점수·판정·이력과 가중치는 채점 입력에서 제외한다.
    payload = {
        "rubric": [{key: item[key] for key in ("id", "name", "basis", "input", "high", "mid", "low")}
                   for item in rubric],
        "analysis": analysis,
        "references": refs,
    }
    # 외부 LLM 호출은 대기 시간이 길 수 있어 실패 위치 확인용 시작 로그도 남긴다.
    logger.info("투자 판단 시작: startup=%s", candidate.name)
    output = _invoke_judge([
        ("system", PROMPT_PATH.read_text(encoding="utf-8")),
        ("human", json.dumps(payload, ensure_ascii=False)),
    ])
    by_id = {item.item_id: item for item in output.items}
    if set(by_id) != ITEM_IDS:
        raise ValueError("LLM 채점 결과에 중복되거나 누락된 평가 항목이 있습니다.")

    scores = {}
    for item in rubric:
        draft = by_id[item["id"]]
        if set(draft.source_ids) - available_sources:
            raise ValueError(f"제공되지 않은 출처 ID가 채점 결과에 있습니다: item_id={draft.item_id}")
        insufficient = draft.insufficient_info or not draft.source_ids or draft.item_id in forced
        rationale = draft.rationale
        if insufficient:
            rationale = f"공개 정보 부족(5점 적용): {rationale}"
        if draft.source_ids:
            rationale += " [출처: " + ", ".join(dict.fromkeys(draft.source_ids)) + "]"
        scores[draft.item_id] = ItemScore(
            item_id=draft.item_id,
            score=5 if insufficient else draft.score,
            weight=item["weight"],
            rationale=rationale,
            insufficient_info=insufficient,
        ).model_dump()

    total, decision, knockout = compute(scores, rubric)
    weakest = sorted(rubric, key=lambda item: scores[item["id"]]["score"] * item["weight"])[:3]
    weakest_names = [item["name"] for item in weakest]
    reason = f"총점 {total:.1f}점 / 기준 {INVEST_THRESHOLD}점. "
    if knockout:
        reason += f"필수 탈락 조건: {knockout} 3점 이하. "
    reason += "가중 점수 하위 항목: " + ", ".join(weakest_names) + "."
    missing = [item["name"] for item in rubric if scores[item["id"]]["insufficient_info"]]
    if missing:
        reason += " 공개 정보 부족: " + ", ".join(missing) + "."
    record = EvalRecord(
        startup=candidate.name, segment=candidate.segment, total_score=total,
        decision=decision, knockout=knockout, weakest_items=weakest_names, key_reason=reason,
    )
    logger.info("투자 판단 완료: startup=%s decision=%s total_score=%.1f",
                candidate.name, decision, total)
    return {
        "scores": scores, "total_score": total, "decision": decision,
        "evaluation_history": [record.model_dump()],
        "log": [f"[judge] {candidate.name}: {decision}, {reason}"],
    }
