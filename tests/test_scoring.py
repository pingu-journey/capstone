import json
import logging
import traceback
from copy import deepcopy
from unittest.mock import Mock

import pytest
import yaml
from httpx import ConnectError, Request
from langchain_core.exceptions import OutputParserException
from openai import APIConnectionError
from pydantic import ValidationError

from agents import investment_judge as judge
from config import ROOT
from schemas import EvalRecord, ItemScore
from tools.references import make_web_ref


@pytest.fixture
def rubric():
    return judge.load_rubric()


@pytest.fixture
def state():
    sample = json.loads((ROOT / "tests/fixtures/sample_state.json").read_text(encoding="utf-8"))
    sample["references"] = [make_web_ref(
        {"title": "테스트 근거", "url": "https://example.org/evidence"},
        "Amperon", "tech_summary", "2026-09-30",
    )]
    return sample


def scores_for(rubric, value=7):
    return {item["id"]: {"score": value} for item in rubric}


@pytest.fixture
def fake_llm(monkeypatch, rubric, state):
    result = {"items": [
        {"item_id": item["id"], "score": 7, "rationale": "제공된 테스트 분석 근거",
         "insufficient_info": False, "source_ids": [state["references"][0]["id"]]}
        for item in rubric
    ]}
    structured = Mock()
    structured.invoke.return_value = result
    llm = Mock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(judge, "get_judge_llm", lambda: llm)
    return result, llm, structured


# 설계서의 11개 가중치와 필수 탈락 항목이 정확히 반영되었는지 확인한다.
def test_rubric_matches_design(rubric):
    assert {item["id"]: item["weight"] for item in rubric} == {
        "team": 25, "market_size": 12, "market_demand": 10, "traction": 10,
        "moat": 8, "trl": 7, "tech_value": 7, "regulation": 6,
        "scalability": 5, "finance": 5, "deal_terms": 5,
    }
    assert sum(item["weight"] for item in rubric) == 100
    assert {item["id"] for item in rubric if item["knockout"]} == {"team", "regulation"}


# 점수 합산이 0~100 범위에서 동작하고 70점부터 투자로 판정하는지 확인한다.
@pytest.mark.parametrize("score, total, decision", [(0, 0.0, "보류"), (7, 70.0, "투자"), (10, 100.0, "투자")])
def test_weighted_total(rubric, score, total, decision):
    actual, verdict, _ = judge.compute(scores_for(rubric, score), rubric)
    assert (actual, verdict) == (total, decision)


# 반올림이나 부동소수점 오차로 69.9점이 투자로 판정되지 않는지 확인한다.
def test_below_threshold(rubric):
    scores = scores_for(rubric)
    scores["finance"]["score"] = 8
    scores["regulation"]["score"] = 6
    assert judge.compute(scores, rubric) == (69.9, "보류", None)


# 창업팀·규제 항목은 3점에서 탈락하고 4점에서는 총점 기준으로 판정하는지 확인한다.
@pytest.mark.parametrize("item_id", ["team", "regulation"])
@pytest.mark.parametrize("score, decision", [(3, "보류"), (4, "투자")])
def test_knockout_boundary(rubric, item_id, score, decision):
    scores = scores_for(rubric, 10)
    scores[item_id]["score"] = score
    total, verdict, knockout = judge.compute(scores, rubric)
    assert total > 70
    assert verdict == decision
    expected = next(item["name"] for item in rubric if item["id"] == item_id)
    assert knockout == (expected if score == 3 else None)


# 정보 부족 항목을 5점으로 계산하고 입력 점수와 가중치를 직접 변경하지 않는지 확인한다.
def test_insufficient_info_and_code_owned_weight(rubric):
    scores = scores_for(rubric, 10)
    scores["team"] = {"score": 0, "insufficient_info": True, "weight": 999}
    before = deepcopy(scores)
    assert judge.compute(scores, rubric) == (87.5, "투자", None)
    assert scores == before


# 범위 밖 점수와 정수가 아닌 값은 조용히 보정하지 않고 거부하는지 확인한다.
@pytest.mark.parametrize("value", [-1, 11, 7.5, "7", True])
def test_invalid_score(rubric, value):
    scores = scores_for(rubric)
    scores["team"]["score"] = value
    with pytest.raises(ValueError, match="0~10 정수"):
        judge.compute(scores, rubric)


# 채점 항목이 빠지거나 정의되지 않은 항목이 추가되면 거부하는지 확인한다.
@pytest.mark.parametrize("extra", [False, True])
def test_invalid_score_ids(rubric, extra):
    scores = scores_for(rubric)
    if extra:
        scores["unknown"] = {"score": 7}
    else:
        scores.pop("team")
    with pytest.raises(ValueError, match="11개 평가 항목"):
        judge.compute(scores, rubric)


# 가중치 합·항목 ID·탈락 조건·기준점이 잘못된 루브릭을 실행 전에 거부하는지 확인한다.
@pytest.mark.parametrize("issue", ["weight", "duplicate", "knockout", "threshold", "shape"])
def test_invalid_rubric(tmp_path, monkeypatch, rubric, issue):
    data = {"threshold": 70, "items": deepcopy(rubric)}
    if issue == "weight":
        data["items"][0]["weight"] += 1
    elif issue == "duplicate":
        data["items"][1]["id"] = "team"
    elif issue == "knockout":
        data["items"][0]["knockout"] = False
    elif issue == "threshold":
        data["threshold"] = 71
    else:
        data = []
    path = tmp_path / "rubric.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(judge, "RUBRIC_PATH", path)
    with pytest.raises(ValueError):
        judge.load_rubric()


# 노드는 기존 State를 보존하며 검증된 점수·판정과 새 평가 이력 하나만 반환하는지 확인한다.
def test_run_returns_only_updates(state, fake_llm, caplog):
    state["log"] = ["[select] 기존 실행 이력"]
    before = deepcopy(state)
    with caplog.at_level(logging.INFO, logger=judge.__name__):
        update = judge.run(state)
    assert state == before
    assert set(update) == {"scores", "total_score", "decision", "evaluation_history", "log"}
    assert (update["total_score"], update["decision"]) == (70.0, "투자")
    assert len(update["evaluation_history"]) == len(update["log"]) == 1
    merged_log = state["log"] + update["log"]
    assert merged_log.count("[select] 기존 실행 이력") == 1
    assert len(merged_log) == 2
    record = EvalRecord.model_validate(update["evaluation_history"][0])
    assert record.weakest_items == ["공급·확장 가능성", "재무·투자 가능성", "투자조건"]
    for score in update["scores"].values():
        assert ItemScore.model_validate(score).score == 7
        assert state["references"][0]["id"] in score["rationale"]
    assert len(caplog.records) == 2
    assert all(r.levelno == logging.INFO for r in caplog.records)
    assert "제공된 테스트 분석 근거" not in caplog.text


# LLM 입력에서 가중치·판정 기준·이전 결과·다른 기업의 출처가 제외되는지 확인한다.
def test_llm_receives_only_scoring_context(state, fake_llm):
    _, llm, structured = fake_llm
    state["references"].append(make_web_ref(
        {"title": "다른 기업", "url": "https://example.org/other"}, "Other", "competitor", "2026-09-30"
    ))
    judge.run(state)
    llm.with_structured_output.assert_called_once_with(judge.JudgeOutput)
    messages = structured.invoke.call_args.args[0]
    payload = json.loads(messages[1][1])
    assert set(payload) == {"rubric", "analysis", "references"}
    assert all("weight" not in item and "knockout" not in item for item in payload["rubric"])
    assert set(payload["analysis"]) == {
        "current_startup", "tech_summary", "market_analysis", "competitor_analysis"
    }
    assert [ref["startup"] for ref in payload["references"]] == ["Amperon"]


# 정보 부족 표시 또는 출처 부재가 있으면 코드가 5점과 분석 한계를 기록하는지 확인한다.
@pytest.mark.parametrize("missing_source", [False, True])
def test_run_normalizes_insufficient_score(state, fake_llm, missing_source):
    result, _, _ = fake_llm
    draft = result["items"][0]
    draft["score"] = 0
    if missing_source:
        draft["source_ids"] = []
    else:
        draft["insufficient_info"] = True
    update = judge.run(state)
    assert update["scores"]["team"]["score"] == 5
    assert update["scores"]["team"]["insufficient_info"] is True
    assert update["total_score"] == 65.0
    assert update["evaluation_history"][0]["knockout"] is None
    assert "공개 정보 부족: 창업팀 역량" in update["evaluation_history"][0]["key_reason"]


# 전체 출처가 없는 샘플은 근거를 만들어내지 않고 모든 항목을 정보 부족으로 처리하는지 확인한다.
def test_run_without_sources(state, fake_llm):
    result, _, _ = fake_llm
    state["references"] = []
    for item in result["items"]:
        item["source_ids"] = []
    update = judge.run(state)
    assert (update["total_score"], update["decision"]) == (50.0, "보류")
    assert all(item["insufficient_info"] for item in update["scores"].values())


# 동일 항목이 중복되고 다른 항목이 빠진 LLM 응답을 거부하는지 확인한다.
def test_run_rejects_duplicate_items(state, fake_llm):
    result, _, _ = fake_llm
    result["items"][-1] = deepcopy(result["items"][0])
    with pytest.raises(ValueError, match="중복되거나 누락"):
        judge.run(state)


# 제공되지 않은 출처를 사용한 채점 결과를 거부하고 ERROR 로그를 중복 기록하지 않는지 확인한다.
def test_run_rejects_unknown_source(state, fake_llm, caplog):
    result, _, _ = fake_llm
    result["items"][0]["source_ids"] = ["invented-source"]
    with caplog.at_level(logging.ERROR, logger=judge.__name__):
        with pytest.raises(ValueError, match="제공되지 않은 출처"):
            judge.run(state)
    assert not caplog.records


# 구조화 출력의 점수·근거·항목 수·추가 필드를 검증하는지 확인한다.
@pytest.mark.parametrize("issue", ["score", "rationale", "length", "weight"])
def test_run_rejects_malformed_output(state, fake_llm, issue):
    result, _, _ = fake_llm
    if issue == "score":
        result["items"][0]["score"] = "7"
    elif issue == "rationale":
        result["items"][0]["rationale"] = " "
    elif issue == "length":
        result["items"].pop()
    else:
        result["items"][0]["weight"] = 25
    with pytest.raises(judge.JudgeServiceError):
        judge.run(state)


# 필수 분석 입력이 없으면 LLM을 호출하기 전에 실패하는지 확인한다.
def test_missing_analysis_fails_before_llm(state, fake_llm):
    _, llm, _ = fake_llm
    state["tech_summary"] = None
    with pytest.raises(ValidationError):
        judge.run(state)
    llm.with_structured_output.assert_not_called()


# LLM 호출 실패를 숨기지 않고 상위 실행 경계로 전달하며 State를 보존하는지 확인한다.
def test_llm_failure_propagates(state, fake_llm):
    _, _, structured = fake_llm
    structured.invoke.side_effect = RuntimeError("모의 API 실패")
    before = deepcopy(state)
    with pytest.raises(RuntimeError, match="모의 API 실패"):
        judge.run(state)
    assert state == before


# 외부 오류 원문을 안전한 예외로 바꾸고 최상위 traceback에도 토큰이 노출되지 않는지 확인한다.
@pytest.mark.parametrize("kind", ["api", "http", "parser", "validation"])
def test_external_failure_is_sanitized(state, fake_llm, caplog, kind):
    result, _, structured = fake_llm
    secret = "sensitive-token-for-test"
    request = Request("GET", "https://example.org/?token=" + secret)
    if kind == "api":
        structured.invoke.side_effect = APIConnectionError(message=secret, request=request)
    elif kind == "http":
        structured.invoke.side_effect = ConnectError(secret, request=request)
    elif kind == "parser":
        structured.invoke.side_effect = OutputParserException(secret, llm_output=secret)
    else:
        result["items"][0]["score"] = secret
    before = deepcopy(state)
    boundary_logger = logging.getLogger("test.entrypoint")
    with caplog.at_level(logging.ERROR):
        try:
            judge.run(state)
        except judge.JudgeServiceError as exc:
            rendered = "".join(traceback.format_exception(exc))
            assert secret not in rendered
            assert "operation=judge" in rendered
            boundary_logger.exception("실행 실패")
        else:
            pytest.fail("외부 오류가 예외로 전달되지 않았습니다.")
    assert state == before
    assert len(caplog.records) == 1
    assert caplog.records[0].name == "test.entrypoint"
    assert secret not in caplog.text


# 사용자 중단과 CLI 종료는 일반 API 실패로 변환하거나 로그로 기록하지 않는지 확인한다.
@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(2)])
def test_control_flow_exceptions_propagate(state, fake_llm, caplog, failure):
    _, _, structured = fake_llm
    structured.invoke.side_effect = failure
    with caplog.at_level(logging.ERROR):
        with pytest.raises(type(failure)):
            judge.run(state)
    assert not caplog.records
