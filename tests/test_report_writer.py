import json
import logging
import traceback
from copy import deepcopy
from unittest.mock import Mock

import pytest
from httpx import ConnectError, Request
from langchain_core.exceptions import OutputParserException

from agents import report_writer as writer
from agents.investment_judge import load_rubric
from config import ROOT
from report.markdown_renderer import to_markdown, write_markdown
from report.templates import ChapterDraft, HoldSummary, InvestmentSummary, blocks_length
from tools.references import make_web_ref


@pytest.fixture
def state():
    data = json.loads((ROOT / "tests/fixtures/sample_state.json").read_text(encoding="utf-8"))
    ref = make_web_ref({"title": "모의 근거", "url": "https://example.org/source"},
                       "Amperon", "tech_summary", "2026-09-30")
    data["references"] = [ref]
    data["scores"] = {item["id"]: {"item_id": item["id"], "score": 8, "weight": item["weight"],
                      "rationale": f"모의 평가 근거 [출처: {ref['id']}]", "insufficient_info": False}
                      for item in load_rubric()}
    data["total_score"] = 80.0
    data["evaluation_history"][0]["total_score"] = 80.0
    data["log"] = ["[judge] 기존 이력"]
    return data


@pytest.fixture
def llm(monkeypatch):
    calls = []

    def structured(model):
        def invoke(messages):
            payload = json.loads(messages[1][1])
            calls.append((model, payload))
            if model is ChapterDraft:
                return {"blocks": [{"subtitle": "근거 검토", "text": "제공된 자료만으로 분석했다."}]}
            if model is InvestmentSummary:
                return {"conclusion": payload["required_opening"] + " 제공된 평가 근거를 종합했다.",
                        "investment_points": ["기술 근거 확인", "수요 근거 확인", "팀 근거 확인"],
                        "risks": ["추가 실사 필요", "공개 정보 한계"]}
            return {"conclusion": payload["required_opening"], "candidate_count": payload["candidate_count"],
                    "common_reasons": payload["expected_common_reasons"]}
        return Mock(invoke=Mock(side_effect=invoke))

    fake = Mock()
    fake.with_structured_output.side_effect = structured
    monkeypatch.setattr(writer, "get_llm", lambda: fake)
    return fake, calls


def hold_state(state, count):
    data = deepcopy(state)
    data["decision"] = "보류"
    data["evaluation_history"] = [
        {"startup": f"후보{i + 1}", "segment": "demand_forecasting", "total_score": 50.0 + i,
         "decision": "보류", "knockout": None, "weakest_items": ["투자조건", "기술 성숙도"],
         "key_reason": "공개 정보 부족: 투자조건. 총점 기준 미달."}
        for i in range(count)
    ]
    return data


# 본문 네 장 다음 SUMMARY를 생성하고 입력 State와 이전 실행 이력을 보존한다.
def test_investment_order_and_state_preservation(state, llm, caplog):
    before = deepcopy(state)
    with caplog.at_level(logging.INFO, logger=writer.__name__):
        report = writer.build_report(state)
    _, calls = llm
    assert [model for model, _ in calls] == [ChapterDraft] * 4 + [InvestmentSummary]
    assert state == before
    assert report.summary.startswith("투자 추천, 총점 80.0점.")
    assert report.total_score == 80.0
    assert len(report.summary) <= 550
    assert all(blocks_length(ch.blocks) <= ch.char_budget for ch in report.chapters)
    assert len(calls[-1][1]["chapters"]) == 4
    assert len(calls[-1][1]["scores"]) == 5
    assert [r.levelno for r in caplog.records] == [logging.INFO, logging.INFO]
    assert "모의 평가 근거" not in caplog.text


# 장별 입력에는 명세에서 허용한 데이터만 전달하고 1장에서는 팀을 제외한다.
def test_chapter_inputs_are_scoped(state, llm):
    writer.build_report(state)
    calls = llm[1]
    assert set(calls[0][1]["inputs"]) == {"current_startup", "tech_summary"}
    assert "team" not in calls[0][1]["inputs"]["tech_summary"]
    assert set(calls[1][1]["inputs"]) == {"market_analysis", "competitor_analysis"}
    assert set(calls[2][1]["inputs"]) == {"team", "traction", "weaknesses", "policy_risks", "competition_risks"}
    assert set(calls[3][1]["inputs"]) == {"scores", "total_score", "decision", "insufficient_info"}


# 전체 보류 3·8곳과 후보 없음에서 마지막 후보 분석·점수를 가져오지 않는다.
@pytest.mark.parametrize("count", [0, 3, 8])
def test_hold_paths_ignore_stale_candidate(state, llm, count):
    data = hold_state(state, count)
    data["tech_summary"] = {"secret": "STALE_TECH"}
    data["total_score"] = 99.9
    data["scores"] = {"secret": "STALE_SCORE"}
    before = deepcopy(data)
    report = writer.build_report(data)
    assert report.route == "hold" and report.total_score is None
    assert f"평가 후보 {count}곳" in report.summary
    assert "99.9" not in report.summary
    assert len(report.chapters[1].tables[0].rows) == count
    payloads = json.dumps(llm[1][-1][1], ensure_ascii=False)
    assert "STALE_" not in payloads and "Amperon" not in payloads
    assert data == before
    if not count:
        assert "평가 가능한 후보를 확보하지 못함" in report.summary
    else:
        assert f"{count}/{count}곳" in report.summary
        assert len(report.chapters[3].tables[0].rows) == count


# 공통 취약 항목이 없으면 후보별 사유를 임의로 공통 위험으로 만들지 않는다.
def test_no_common_reason(state, llm):
    data = hold_state(state, 3)
    for i, record in enumerate(data["evaluation_history"]):
        record["weakest_items"] = [f"항목{i}"]
    assert "공통 취약 항목을 확인하지 못함" in writer.build_report(data).summary


# 정보 부족 항목은 LLM 응답에서 빠져도 코드가 분석의 한계에 보존한다.
def test_limitations_are_deterministic(state, llm):
    state["scores"]["deal_terms"].update(score=5, insufficient_info=True)
    state["total_score"] = 78.5
    state["tech_summary"]["insufficient"] = ["검증된 성능 수치"]
    report = writer.build_report(state)
    text = report.chapters[3].blocks[-1].text
    assert "투자조건" in text and "5점 적용" in text and "검증된 성능 수치" in text
    rows = report.chapters[3].tables[0].rows
    assert len(rows) == 12 and rows[-1][3] == "78.5"


# 출처는 기업 필터 후 중복 제거하며 공통 출처와 서지 형식을 보존한다.
def test_reference_selection(state, llm):
    common = make_web_ref({"title": "공통", "url": "https://example.org/common"}, "*", "market_eval", "2026-09-30")
    other = make_web_ref({"title": "다른 기업", "url": "https://example.org/other"}, "Other", "tech_summary", "2026-09-30")
    state["references"] += [common, other, deepcopy(state["references"][0])]
    report = writer.build_report(state)
    assert len(report.references["web"]) == 2
    markdown = to_markdown(report)
    assert "다른 기업" not in markdown and "공통" in markdown
    assert state["references"][0]["id"] in markdown


# 시장 수치를 코드 표로 만들고 값·단위·연도·출처를 빠뜨리지 않는다.
def test_market_table(state, llm):
    source_id = state["references"][0]["id"]
    state["market_analysis"]["market_size"] = [
        {"metric": "용량 전망", "value": "80~160 GW", "year": "2030", "source_id": source_id}]
    report = writer.build_report(state)
    assert report.chapters[1].tables[1].rows[0][-4:] == ["용량 전망", "80~160 GW", "2030", source_id]


# 입력 점수·판정·가중치·출처 불일치는 외부 LLM을 호출하기 전에 거부한다.
@pytest.mark.parametrize("issue", ["total", "weight", "item_id", "missing_score", "insufficient", "market_source", "rationale_source"])
def test_bad_inputs_fail_before_llm(state, llm, issue):
    if issue == "total":
        state["total_score"] = 90
    elif issue == "weight":
        state["scores"]["team"]["weight"] = 1
    elif issue == "item_id":
        state["scores"]["team"]["item_id"] = "trl"
    elif issue == "missing_score":
        state["scores"].pop("team")
    elif issue == "insufficient":
        state["scores"]["deal_terms"]["insufficient_info"] = True
    elif issue == "market_source":
        state["market_analysis"]["market_size"] = [
            {"metric": "규모", "value": "10 GW", "year": "2030", "source_id": "unknown"}]
    else:
        state["scores"]["team"]["rationale"] = "근거 [출처: unknown]"
    with pytest.raises(ValueError):
        writer.build_report(state)
    llm[0].with_structured_output.assert_not_called()


# 전체 보류 이력의 중복·투자 판정·범위 밖 점수를 거부한다.
@pytest.mark.parametrize("issue", ["duplicate", "invest", "range", "threshold"])
def test_bad_history(state, llm, issue):
    data = hold_state(state, 3)
    if issue == "duplicate":
        data["evaluation_history"][1]["startup"] = "후보1"
    elif issue == "invest":
        data["evaluation_history"][0]["decision"] = "투자"
    else:
        data["evaluation_history"][0]["total_score"] = 101 if issue == "range" else 80
    with pytest.raises(writer.ReportValidationError):
        writer.build_report(data)
    assert not llm[1]


# SUMMARY 일관성 실패는 한 번 재생성하고 같은 오류가 반복되면 실패한다.
@pytest.mark.parametrize("repair", [True, False])
@pytest.mark.parametrize("route", ["investment", "hold"])
def test_summary_retry(state, llm, monkeypatch, repair, route):
    if route == "hold":
        state = hold_state(state, 3)
    original = writer._invoke
    attempts = []

    def invoke(model, prompt, payload):
        result = original(model, prompt, payload)
        if model is not ChapterDraft:
            attempts.append(deepcopy(payload))
            if len(attempts) == 1 or not repair:
                result.conclusion = "잘못된 결론"
        return result

    monkeypatch.setattr(writer, "_invoke", invoke)
    if repair:
        writer.build_report(state)
    else:
        with pytest.raises(writer.ReportValidationError):
            writer.build_report(state)
    assert len(attempts) == 2 and "retry_reason" in attempts[1]


# 글자 수·빈 투자 포인트·출처·후보 수·공통 사유 오류를 SUMMARY 검사에서 거부한다.
@pytest.mark.parametrize("issue", ["length", "blank", "source", "count", "reason", "contradiction", "duplicate"])
def test_summary_content_validation(state, llm, monkeypatch, issue):
    if issue in ("count", "reason"):
        state = hold_state(state, 3)
    original = writer._invoke

    def invoke(model, prompt, payload):
        result = original(model, prompt, payload)
        if model is InvestmentSummary:
            if issue == "length":
                result.risks[0] = "가" * 550
            elif issue == "blank":
                result.investment_points[0] = " "
            elif issue == "source":
                result.risks[0] = "위험 [출처: invented]"
            elif issue == "contradiction":
                result.conclusion += " 최종적으로 보류한다."
            elif issue == "duplicate":
                result.investment_points[1] = result.investment_points[0]
        if model is HoldSummary:
            if issue == "count":
                result.candidate_count += 1
            else:
                result.common_reasons = ["임의로 만든 사유"]
        return result

    monkeypatch.setattr(writer, "_invoke", invoke)
    with pytest.raises(writer.ReportValidationError):
        writer.build_report(state)


# 장별 초과 분량과 알 수 없는 출처는 한 번만 재시도하며 SUMMARY까지 진행하지 않는다.
@pytest.mark.parametrize("issue", ["length", "source"])
def test_chapter_validation(state, llm, monkeypatch, issue):
    original = writer._invoke

    def invoke(model, prompt, payload):
        draft = original(model, prompt, payload)
        draft.blocks[0].text = "가" * 901 if issue == "length" else "근거 [출처: unknown]"
        return draft

    monkeypatch.setattr(writer, "_invoke", invoke)
    with pytest.raises(writer.ReportValidationError):
        writer.build_report(state)
    assert len(llm[1]) == 2


# PDF 재생성 단계에서 사용할 20% 축소 예산을 각 장에 적용한다.
@pytest.mark.parametrize("reduction, budgets", [(0, [900, 1100, 1100, 700]), (1, [720, 880, 880, 560]), (2, [576, 704, 704, 448])])
def test_reduced_budgets(state, llm, reduction, budgets):
    report = writer.build_report(state, reduction=reduction)
    assert [chapter.char_budget for chapter in report.chapters] == budgets


# 외부 HTTP·파싱·응답 검증 오류의 원문을 최상위 traceback에 노출하지 않는다.
@pytest.mark.parametrize("kind", ["http", "parser", "validation"])
def test_sanitized_errors(state, llm, kind):
    fake, _ = llm
    secret = "SECRET_RESPONSE_TOKEN"
    structured = Mock()
    if kind == "http":
        structured.invoke.side_effect = ConnectError(secret, request=Request("GET", "https://example.org/"))
    elif kind == "parser":
        structured.invoke.side_effect = OutputParserException(secret)
    else:
        structured.invoke.return_value = {"blocks": secret}
    fake.with_structured_output.side_effect = None
    fake.with_structured_output.return_value = structured
    with pytest.raises((writer.ReportServiceError, writer.ReportValidationError)) as exc:
        writer.build_report(state)
    assert secret not in "".join(traceback.format_exception(exc.value))
    assert structured.invoke.call_count == (1 if kind == "http" else 2)


# 사용자 중단과 프로세스 종료는 재시도나 일반 오류로 변환하지 않는다.
@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(2)])
def test_control_flow_propagates(state, llm, failure):
    fake, _ = llm
    fake.with_structured_output.side_effect = failure
    with pytest.raises(type(failure)):
        writer.build_report(state)
    assert fake.with_structured_output.call_count == 1


# Markdown은 동일 데이터에서 SUMMARY·본문·표·REFERENCE 순으로 저장한다.
def test_markdown_output(state, llm, tmp_path):
    report = writer.build_report(state)
    report.chapters[0].tables.append(writer.Table(title="이스케이프", columns=["값"], rows=[["가|나\n<script>"]]))
    path = write_markdown(report, tmp_path / "report_20260930_1200.md")
    text = path.read_text(encoding="utf-8")
    assert text == to_markdown(report)
    assert text.index("## SUMMARY") < text.index("## 1.") < text.index("## 4.") < text.index("## REFERENCE")
    assert "가\\|나<br>&lt;script&gt;" in text
    assert "| 합계 | 100 |" in text and "80\\.0" in text
    with pytest.raises(ValueError):
        write_markdown(report, tmp_path / "report.pdf")


# 보류 보고서는 보존된 시장 캐시를 읽고 그 수치·출처를 공용 데이터에 보존한다.
def test_hold_uses_market_cache(state, llm):
    data = hold_state(state, 8)
    market = deepcopy(state["market_analysis"])
    source_id = state["references"][0]["id"]
    market["market_size"] = [{"metric": "용량", "value": "100 GW", "year": "2030", "source_id": source_id}]
    data["market_cache"] = {"US|demand_forecasting": market}
    report = writer.build_report(data, reduction=2)
    assert llm[1][2][1]["inputs"]["market_cache"] == data["market_cache"]
    assert report.chapters[2].tables[0].rows[0][-3:] == ["100 GW", "2030", source_id]
    assert all(blocks_length(ch.blocks) <= ch.char_budget for ch in report.chapters)


# 긴 근거·URL·다수 출처는 본문 예산을 이유로 잘라내거나 삭제하지 않는다.
def test_long_evidence_and_many_references(state, llm):
    long_reason = ("검증이 필요한 긴 근거 " * 100).strip()
    state["scores"]["team"]["rationale"] = long_reason
    for index in range(20):
        state["references"].append(make_web_ref(
            {"title": f"근거{index}", "url": f"https://example.org/{index}?query=" + "a" * 500},
            "Amperon", "tech_summary", "2026-09-30"))
    report = writer.build_report(state)
    assert report.chapters[3].tables[0].rows[0][-1] == long_reason
    assert len(report.references["web"]) == 21
    assert "a" * 500 in to_markdown(report)


# 잘못된 판정과 지원 범위 밖 축소 횟수는 LLM 호출 전에 거부한다.
@pytest.mark.parametrize("issue", ["decision", "reduction"])
def test_invalid_route_or_reduction(state, llm, issue):
    if issue == "decision":
        state["decision"] = "추천"
    with pytest.raises(ValueError):
        writer.build_report(state, reduction=3 if issue == "reduction" else 0)
    assert not llm[1]


# 모의 LLM 본문 생성부터 실제 PDF·Markdown 저장까지 보고서 노드 전체를 실행한다.
def test_run_generates_real_artifacts(state, llm, monkeypatch, tmp_path):
    from pathlib import Path
    from pypdf import PdfReader

    monkeypatch.setattr(writer, "OUTPUT_DIR", tmp_path)
    before = deepcopy(state)
    result = writer.run(state)
    pdf = Path(result["report_path"])
    assert set(result) == {"report_path", "log"}
    assert state == before and len(result["log"]) == 1
    reader = PdfReader(pdf)
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert 1 <= len(reader.pages) <= 5
    assert "SUMMARY" in text and "REFERENCE" in text and "80.0점" in text
    assert "80\\.0점" in pdf.with_suffix(".md").read_text(encoding="utf-8")


# 실제 글자 수 초과 시 재생성 입력에 초과 길이와 더 작은 작성 목표를 전달한다.
def test_chapter_retry_includes_measured_length(state, llm, monkeypatch):
    original = writer._invoke
    attempts = []

    def invoke(model, prompt, payload):
        if model is ChapterDraft and payload["chapter"] == 1:
            attempts.append(deepcopy(payload))
            if len(attempts) == 1:
                return ChapterDraft(blocks=[{"subtitle": "초과", "text": "가" * 901}])
        return original(model, prompt, payload)

    monkeypatch.setattr(writer, "_invoke", invoke)
    writer.build_report(state)
    assert len(attempts) == 2
    assert attempts[1]["previous_char_count"] == 904
    assert attempts[1]["target_chars"] < attempts[0]["target_chars"] < attempts[0]["char_budget"]
