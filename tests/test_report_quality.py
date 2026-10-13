"""Fail → feedback → re-review, finite termination, semantic Judge and frozen artifacts."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader

from agents import report_quality as quality, report_writer as writer
from agents.investment_judge import load_rubric
from config import ROOT
from graph.routing import route_after_quality
from report.templates import ChapterDraft, HoldSummary, InvestmentSummary
from schemas import QualityCheck, QualityVerdict
from state import InvestState
from tools import artifacts
from tools.artifacts import read_json, run_dir, write_json
from tools.references import make_web_ref


def verdict(version=1, passed=True, feedback="수치를 원문과 일치시키세요.", recoverable=True, text="자료를 검토했다."):
    check = QualityCheck(passed=passed, reason="통과" if passed else feedback, feedback=[] if passed else [feedback])
    good = QualityCheck(passed=True, reason="통과", feedback=[])
    return QualityVerdict(report_version=version, groundedness=check, neutrality=good, bias=good,
                          structure_passed=True, structure_reasons=[], recoverable=recoverable,
                          issues=[] if passed else [{"chapter": 1, "block": 1, "kind": "groundedness", "text": text,
                                                     "reason": feedback}])


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUTPUT_DIR", tmp_path)
    state = json.loads((ROOT / "tests/fixtures/sample_state.json").read_text(encoding="utf-8"))
    state.update(run_id="quality_test", report_version=0, quality_result=None, execution_status="running")
    ref = make_web_ref({"title": "독립 근거", "url": "https://independent.org/test"}, "Amperon", "test", state["run_date"])
    state["references"] = [ref]
    state["scores"] = {item["id"]: {"item_id": item["id"], "weight": item["weight"], "score": 8,
                                   "rationale": "가상 근거", "insufficient_info": False} for item in load_rubric()}
    state["total_score"] = 80.0
    state["evaluation_history"][0]["total_score"] = 80.0
    write_json(run_dir(state) / "evidence/test.json", [{"source_id": ref["id"], "startup": "Amperon", "text": "실제 수치는 100 GW, 기준 연도 2030.", "page": 1, "reference": ref}])
    state["evidence_index_ref"] = str(run_dir(state) / "evidence_index.json")
    return state


@pytest.fixture
def generator(monkeypatch):
    calls = []
    def invoke(model, prompt, payload):
        calls.append(deepcopy(payload))
        if model is ChapterDraft:
            return ChapterDraft(blocks=[{"subtitle": "자료 검토", "text": "자료를 검토했다. 단일 출처 의존 한계가 있으며 추가 확인이 필요하다."}])
        if model is InvestmentSummary:
            return InvestmentSummary(conclusion=payload["required_opening"], investment_points=["기술 근거", "시장 근거", "팀 근거"], risks=["추가 실사 필요", "공개 정보 한계"])
        return HoldSummary(conclusion=payload["required_opening"], candidate_count=payload["candidate_count"], common_reasons=payload["expected_common_reasons"])
    monkeypatch.setattr(writer, "_invoke", invoke)
    return calls


def install_judge(monkeypatch, outputs):
    calls = []
    def invoke(messages):
        payload = json.loads(messages[1][1])
        calls.append(payload)
        result = outputs.pop(0) if len(outputs) > 1 else outputs[0]
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result(payload)
        return result
    monkeypatch.setattr(quality, "get_judge_llm", lambda: SimpleNamespace(with_structured_output=lambda model: SimpleNamespace(invoke=invoke)))
    return calls


def graph():
    g = StateGraph(InvestState)
    g.add_node("report_draft", writer.draft)
    g.add_node("report_quality", quality.run)
    g.add_node("report_finalize", writer.finalize)
    g.add_node("quality_failed", quality.failed)
    g.add_edge(START, "report_draft")
    g.add_edge("report_draft", "report_quality")
    g.add_conditional_edges("report_quality", route_after_quality, ["report_draft", "report_finalize", "quality_failed"])
    g.add_edge("report_finalize", END)
    g.add_edge("quality_failed", END)
    return g.compile()


def test_fail_feedback_improves_and_finalizes_same_pdf(state, generator, monkeypatch):
    calls = install_judge(monkeypatch, [verdict(1, False), verdict(2)])
    before = deepcopy(state)
    final = graph().invoke(state)
    assert final["execution_status"] == "complete" and final["report_version"] == 2
    assert state == before
    assert len(calls) == 2 and calls[0]["evidence"][0]["text"].startswith("실제 수치")
    assert generator[5]["quality_feedback"]["groundedness"]["feedback"] == ["수치를 원문과 일치시키세요."]
    path = Path(final["report_path"])
    assert path.read_bytes() == Path(final["report_preview_ref"]).read_bytes()
    assert len(PdfReader(path).pages) <= quality.MAX_PAGES
    assert not (run_dir(state) / "report_v1.pdf").exists()
    assert (run_dir(state) / "quality_v1.json").is_file()


def test_quality_failure_stops_after_three_revisions(state, generator, monkeypatch):
    calls = install_judge(monkeypatch, [lambda p: verdict(p["report_version"], False, text=p["report"]["chapters"][0]["blocks"][0]["text"])])
    final = graph().invoke(state)
    assert len(calls) == 3 and final["report_version"] == 3
    assert final["execution_status"] == "quality_failed" and final["report_path"] is None
    assert (run_dir(state) / "preview_v3.pdf").is_file()
    assert not list(run_dir(state).glob("report_v*.pdf"))


@pytest.mark.parametrize("count", [0, 3, 8])
def test_hold_and_no_candidate_quality_routes(state, generator, monkeypatch, count):
    state["decision"] = None if count == 0 else "보류"
    state["evaluation_history"] = [{"startup": f"후보{i}", "segment": "demand_forecasting", "total_score": 60.0,
        "decision": "보류", "knockout": None, "weakest_items": ["투자조건"], "key_reason": "총점 미달"} for i in range(count)]
    if count == 0:
        state["references"] = []
    calls = install_judge(monkeypatch, [verdict()])
    final = graph().invoke(state)
    assert final["execution_status"] == "complete"
    assert calls[0]["report"]["total_score"] is None
    assert f"평가 후보 {count}곳" in calls[0]["report"]["summary"]


@pytest.mark.parametrize("issue", ["uncited_number", "contradiction", "promotion", "bias", "missing_excerpt"])
def test_code_checks_cannot_be_overridden_by_judge(state, generator, monkeypatch, issue):
    state.update(writer.draft(state))
    artifact = read_json(state["report_draft_ref"])
    block = artifact["report"]["chapters"][0]["blocks"][0]
    source_id = state["references"][0]["id"]
    if issue == "uncited_number":
        block["text"] = "용량은 100 GW다."
    elif issue == "contradiction":
        block["text"] = f"용량은 999 GW다. [출처: {source_id}]"
    elif issue == "promotion":
        block["text"] = "최고의 기술이므로 무조건 투자해야 한다."
    elif issue == "bias":
        for ch in artifact["report"]["chapters"]:
            for item in ch["blocks"]:
                item["text"] = item["text"].replace("단일 출처", "자료")
    else:
        write_json(Path(state["evidence_index_ref"]), [])
    write_json(Path(state["report_draft_ref"]), artifact)
    install_judge(monkeypatch, [verdict()])
    result = QualityVerdict.model_validate(quality.run(state)["quality_result"])
    assert not result.passed
    assert not getattr(result, "neutrality" if issue == "promotion" else "bias" if issue == "bias" else "groundedness").passed


def test_semantic_judge_rejects_claim_despite_valid_source_id(state, generator, monkeypatch):
    state.update(writer.draft(state))
    def evaluate(payload):
        assert payload["evidence"][0]["source_id"] == state["references"][0]["id"]
        return verdict(1, False, "유효 ID지만 원문에 없는 팀 이력을 삭제하세요.")
    install_judge(monkeypatch, [evaluate])
    result = quality.run(state)
    assert not QualityVerdict.model_validate(result["quality_result"]).passed


def test_judge_service_failure_never_auto_passes(state, generator, monkeypatch):
    state.update(writer.draft(state))
    calls = install_judge(monkeypatch, [ValueError("malformed response")])
    state.update(quality.run(state))
    assert len(calls) == 2 and route_after_quality(state) == "quality_failed"
    assert not QualityVerdict.model_validate(state["quality_result"]).recoverable


def test_finalizer_rejects_modified_reviewed_pdf(state, generator, monkeypatch):
    state.update(writer.draft(state))
    install_judge(monkeypatch, [verdict()])
    state.update(quality.run(state))
    Path(state["report_preview_ref"]).write_bytes(b"tampered")
    with pytest.raises(writer.ReportValidationError, match="PDF가 변경"):
        writer.finalize(state)
    assert not list(run_dir(state).glob("report_v*.pdf"))


def test_stale_quality_version_rejected():
    with pytest.raises(ValueError, match="버전"):
        route_after_quality({"report_version": 2, "quality_result": verdict(1).model_dump()})


def test_layout_failure_is_feedback_and_preserves_review_markdown(state, generator, monkeypatch):
    from report.pdf_renderer import ReportLayoutError
    original = writer.render_pdf
    attempts = []
    def render(report, **kwargs):
        attempts.append(report)
        if len(attempts) == 1:
            raise ReportLayoutError("SUIT에 없는 글리프: U+6301")
        return original(report, **kwargs)
    monkeypatch.setattr(writer, "render_pdf", render)
    install_judge(monkeypatch, [verdict(2)])
    final = graph().invoke(state)
    assert final["execution_status"] == "complete" and final["report_version"] == 2
    assert (run_dir(state) / "preview_v1.md").is_file()
    assert "U+6301" in generator[5]["quality_feedback"]["structure_reasons"][0]


def test_generated_block_citations_are_validated_and_rendered(state, generator, monkeypatch):
    from tools.evidence import index_evidence
    state["evidence_index_ref"] = index_evidence(state)
    original = writer._invoke
    source_id = state["references"][0]["id"]
    def invoke(model, prompt, payload):
        draft = original(model, prompt, payload)
        if model is ChapterDraft:
            draft.blocks[0].source_ids = [source_id]
        return draft
    monkeypatch.setattr(writer, "_invoke", invoke)
    report = writer.build_report(state)
    assert all(f"[출처: {source_id}]" in chapter.blocks[0].text for chapter in report.chapters)


def test_full_eight_candidate_bound_fits_recursion_limit(monkeypatch, tmp_path):
    from graph.builder import build_graph
    from agents import discovery
    from config import RECURSION_LIMIT
    monkeypatch.setattr(artifacts, "OUTPUT_DIR", tmp_path)
    fixture = json.loads((ROOT / "tests/fixtures/sample_state.json").read_text(encoding="utf-8"))
    def discover(state):
        n = state.get("search_round", 0) + 1
        candidates = [{**fixture["current_startup"], "name": f"기업{(n-1)*5+i}"} for i in range(5)]
        return {"candidates": candidates, "search_round": n}
    monkeypatch.setattr(discovery, "run", discover)
    final = build_graph(stub=True, stub_decisions=["보류"]).invoke(
        {"run_id": "bound_test", "run_date": "2026-09-30", "evaluated": [], "log": [], "market_cache": {}, "references": [], "evaluation_history": []},
        config={"recursion_limit": RECURSION_LIMIT})
    assert len(final["evaluated"]) == 8 and final["search_round"] == 2


@pytest.mark.parametrize("claim, accepted", [
    ("현장 실증 근거로 TRL 7로 평가한다", True),
    ("현장 실증 근거로 TRL 8로 평가한다", False),
    ("현장 실증 근거로 TRL 7로 평가한다. 시장 용량은 7 GW다", False),
    ("시장 용량은 7 GW로 평가한다", False),
])
def test_trl_assessment_numeric_provenance_in_generation_and_quality(state, generator, monkeypatch, claim, accepted):
    from tools.evidence import index_evidence
    state["tech_summary"].update(trl=7, trl_rationale="현장 실증 근거", insufficient=[])
    state["references"][0]["used_by"] = "tech_summary"
    source_id = state["references"][0]["id"]
    write_json(run_dir(state) / "evidence/test.json", [{"source_id": source_id, "startup": "Amperon",
        "text": "현장 실증을 수행했다.", "reference": state["references"][0]}])
    state["evidence_index_ref"] = index_evidence(state)
    original = writer._invoke
    def invoke(model, prompt, payload):
        result = original(model, prompt, payload)
        if model is ChapterDraft and payload["chapter"] == 1:
            result.blocks[0].text = claim + f" [출처: {source_id}] 단일 출처 의존 한계가 있다."
        return result
    monkeypatch.setattr(writer, "_invoke", invoke)
    if accepted:
        state.update(writer.draft(state))
        assert "report" in read_json(state["report_draft_ref"])
    else:
        repair_log = []
        safe = writer.build_report(state, repair_log=repair_log)
        assert "공개 정보 부족" in safe.chapters[0].blocks[0].text
        assert any(item["issues"][0]["kind"] == "numeric" for item in repair_log)
        monkeypatch.setattr(writer, "_invoke", original)
        state.update(writer.draft(state))
        artifact = read_json(state["report_draft_ref"])
        artifact["report"]["chapters"][0]["blocks"][0]["text"] = claim + f" [출처: {source_id}]"
        write_json(Path(state["report_draft_ref"]), artifact)
    calls = install_judge(monkeypatch, [verdict()])
    result = QualityVerdict.model_validate(quality.run(state)["quality_result"])
    assert result.groundedness.passed is accepted
    if accepted:
        assert calls[0]["inputs"]["tech_summary"]["trl"] == 7


def test_trl_assessment_still_requires_citation(state, generator, monkeypatch):
    state["tech_summary"].update(trl=7, trl_rationale="현장 실증 근거")
    state.update(writer.draft(state))
    artifact = read_json(state["report_draft_ref"])
    artifact["report"]["chapters"][0]["blocks"][0]["text"] = "현장 실증 근거로 TRL 7로 평가한다."
    write_json(Path(state["report_draft_ref"]), artifact)
    install_judge(monkeypatch, [verdict()])
    assert not QualityVerdict.model_validate(quality.run(state)["quality_result"]).groundedness.passed


@pytest.mark.parametrize('amount, accepted', [('2000만 달러', True), ('2000만 원', False)])
def test_amount_equivalence_quality_gate_cannot_be_overridden(state, generator, monkeypatch, amount, accepted):
    from tools.evidence import index_evidence
    from report.templates import ReportDocument
    from report.pdf_renderer import render_pdf
    source = state['references'][0]['id']
    write_json(run_dir(state) / 'evidence/test.json', [{'source_id': source, 'startup': 'Amperon',
        'text': 'Series B funding: $20 million.', 'reference': state['references'][0]}])
    state['evidence_index_ref'] = index_evidence(state)
    state.update(writer.draft(state))
    artifact = read_json(state['report_draft_ref'])
    report = ReportDocument.model_validate(artifact['report'])
    report.chapters[0].blocks[0].text = f'유치액은 {amount}다. 단일 출처 의존 한계가 있다. [출처: {source}]'
    rendered = render_pdf(report)
    Path(state['report_preview_ref']).write_bytes(rendered.content)
    artifact.update(report=report.model_dump(), page_count=rendered.page_count, summary_bottom=rendered.summary_bottom)
    write_json(Path(state['report_draft_ref']), artifact)
    install_judge(monkeypatch, [verdict()])  # Even a permissive Judge cannot approve changed currency.
    result = QualityVerdict.model_validate(quality.run(state)['quality_result'])
    assert result.structure_passed
    assert result.groundedness.passed is accepted


@pytest.mark.parametrize('pages, accepted', [(10, True), (11, False)])
def test_assignment_page_limit_and_soft_character_targets(state, generator, monkeypatch, pages, accepted):
    from pypdf import PdfReader, PdfWriter
    state.update(writer.draft(state))
    artifact = read_json(state['report_draft_ref'])
    artifact['report']['chapters'][0]['char_budget'] = 1  # A writing target cannot reject verified content.
    pdf = PdfWriter()
    for page in PdfReader(state['report_preview_ref']).pages:
        pdf.add_page(page)
    while len(pdf.pages) < pages:
        pdf.add_blank_page(width=595, height=842)
    pdf.write(state['report_preview_ref'])
    artifact['page_count'] = pages
    write_json(Path(state['report_draft_ref']), artifact)
    install_judge(monkeypatch, [verdict()])
    result = QualityVerdict.model_validate(quality.run(state)['quality_result'])
    assert result.structure_passed is accepted
    assert result.passed is accepted
