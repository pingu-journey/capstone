"""Repair only rejected claims; failed repairs cannot reintroduce numbers into PDF."""
from copy import deepcopy

from agents import report_writer as writer
from report.templates import ChapterDraft, InvestmentSummary
from tools.artifacts import read_json, run_dir, write_json
from test_report_quality import state, generator, install_judge, graph, verdict


def test_repeated_bad_number_is_replaced_without_losing_healthy_block(state, generator, monkeypatch):
    original = writer._invoke
    source = state["references"][0]["id"]
    requests = []
    def invoke(model, prompt, payload):
        requests.append(deepcopy(payload))
        if model is ChapterDraft and payload["chapter"] == 1:
            blocks = [{"subtitle": "용량", "text": f"용량은 1.3 GW다. [출처: {source}]", "source_ids": [source]}]
            if payload.get("operation") != "repair_block":
                blocks.append({"subtitle": "확인된 내용", "text": "단일 출처 의존 한계가 있다."})
            return ChapterDraft(blocks=blocks)
        return original(model, prompt, payload)
    monkeypatch.setattr(writer, "_invoke", invoke)
    state.update(writer.draft(state))
    artifact = read_json(state["report_draft_ref"])
    assert "report" in artifact
    blocks = artifact["report"]["chapters"][0]["blocks"]
    assert "1.3" not in blocks[0]["text"] and "공개 정보 부족" in blocks[0]["text"]
    assert blocks[1]["text"] == "단일 출처 의존 한계가 있다."
    repair = next(p for p in requests if p.get("operation") == "repair_block")
    assert "1.3" in repair["previous_block"]["text"]
    assert repair["violations"][0]["numbers"] == ["1.3"]
    assert repair["violations"][0]["source_ids"] == [source]
    assert repair["violations"][0]["text"] == repair["previous_block"]["text"]
    assert repair["cited_evidence"][0]["source_id"] == source
    assert repair["cited_evidence"][0]["text"]
    assert artifact["repair_log"][0]["action"] == "fallback"
    assert artifact["repair_log"][0]["issues"][0]["chapter"] == 1
    assert artifact["repair_log"][0]["issues"][0]["block"] == 1
    summary = next(p for p in requests if "required_opening" in p)
    assert "1.3" not in str(summary["chapters"])


def test_bad_number_in_summary_cannot_escape_body_gate(state, generator, monkeypatch):
    original = writer._invoke
    source = state["references"][0]["id"]
    def invoke(model, prompt, payload):
        result = original(model, prompt, payload)
        if model is InvestmentSummary:
            result.investment_points[0] = f"용량 2.6 GW [출처: {source}]"
        return result
    monkeypatch.setattr(writer, "_invoke", invoke)
    state.update(writer.draft(state))
    artifact = read_json(state["report_draft_ref"])
    assert "2.6" not in artifact["report"]["summary"]
    assert artifact["report"]["summary"].startswith("투자 추천, 총점 80.0점.")
    assert any(item["chapter"] == 0 and item["action"] == "fallback" for item in artifact["repair_log"])


def test_quality_loop_reuses_healthy_blocks_and_rebuilds_summary(state, generator, monkeypatch):
    install_judge(monkeypatch, [verdict(1, False), verdict(2)])
    final = graph().invoke(state)
    assert final["execution_status"] == "complete"
    initial_chapters = [p for p in generator if "chapter" in p and not p.get("operation")]
    assert len(initial_chapters) == 4
    repair_calls = [p for p in generator if p.get("operation") == "repair_block"]
    assert len(repair_calls) == 1 and repair_calls[0]["chapter"] == 1
    assert repair_calls[0]["previous_block"]["text"]
    assert len([p for p in generator if "required_opening" in p]) == 2


def test_repair_cannot_replace_one_bad_number_with_another(state, generator, monkeypatch):
    original = writer._invoke
    source = state["references"][0]["id"]
    def invoke(model, prompt, payload):
        if model is ChapterDraft and payload["chapter"] == 1:
            value = "2.6" if payload.get("operation") else "1.3"
            return ChapterDraft(blocks=[{"subtitle": "용량", "text": f"용량 {value} GW [출처: {source}]"}])
        return original(model, prompt, payload)
    monkeypatch.setattr(writer, "_invoke", invoke)
    state.update(writer.draft(state))
    artifact = read_json(state["report_draft_ref"])
    report = str(artifact["report"])
    assert "1.3" not in report and "2.6" not in report
    repair = artifact["repair_log"][0]
    assert repair["issues"][0]["numbers"] == ["1.3"]
    assert repair["repair_issues"][0]["numbers"] == ["2.6"]


def test_repeated_semantic_failure_for_same_block_falls_back_on_third_version(state, generator, monkeypatch):
    install_judge(monkeypatch, [verdict(1, False), verdict(2, False), verdict(3)])
    final = graph().invoke(state)
    assert final["execution_status"] == "complete" and final["report_version"] == 3
    assert len([p for p in generator if p.get("operation") == "repair_block"]) == 1
    artifact = read_json(final["report_draft_ref"])
    assert artifact["repair_log"][0]["action"] == "fallback"
    assert "공개 정보 부족" in artifact["report"]["chapters"][0]["blocks"][0]["text"]


def test_unlocated_judge_failure_stops_without_random_rewrite(state, generator, monkeypatch):
    failed = verdict(1, False).model_copy(update={"issues": []})
    install_judge(monkeypatch, [failed])
    final = graph().invoke(state)
    assert final["report_version"] == 1 and final["execution_status"] == "quality_failed"
    assert len([p for p in generator if "chapter" in p]) == 4


def test_tampered_summary_numeric_gate_cannot_be_overridden(state, generator, monkeypatch):
    from pathlib import Path
    from agents import report_quality
    from tools.artifacts import write_json
    state.update(writer.draft(state))
    artifact = read_json(state["report_draft_ref"])
    artifact["report"]["summary"] += f" 용량 2.6 GW [출처: {state['references'][0]['id']}]"
    write_json(Path(state["report_draft_ref"]), artifact)
    install_judge(monkeypatch, [verdict()])
    result = report_quality.run(state)["quality_result"]
    assert not result["groundedness"]["passed"]
    assert any(issue["chapter"] == 0 and issue["numbers"] == ["2.6"] for issue in result["issues"])


def test_identical_numeric_value_with_a_valid_source_is_preserved(state, generator, monkeypatch):
    from tools.references import make_web_ref
    from tools.artifacts import write_json, run_dir
    from tools.evidence import index_evidence
    good = make_web_ref({"title": "검증 자료", "url": "https://verified.org/claim"}, "Amperon", "test", state["run_date"])
    state["references"].append(good)
    write_json(run_dir(state) / "evidence/good.json", [{"source_id": good["id"], "startup": "Amperon",
        "text": "용량은 1.3 GW다.", "reference": good}])
    state["evidence_index_ref"] = index_evidence(state)
    bad = state["references"][0]["id"]
    original = writer._invoke
    def invoke(model, prompt, payload):
        if model is ChapterDraft and payload["chapter"] == 1:
            ids = [bad] if payload.get("operation") else [good["id"], bad]
            return ChapterDraft(blocks=[{"subtitle": "용량", "text": f"용량은 1.3 GW다. [출처: {source}]"} for source in ids])
        return original(model, prompt, payload)
    monkeypatch.setattr(writer, "_invoke", invoke)
    state.update(writer.draft(state))
    blocks = read_json(state["report_draft_ref"])["report"]["chapters"][0]["blocks"]
    assert "1.3" in blocks[0]["text"] and good["id"] in blocks[0]["text"]
    assert "1.3" not in blocks[1]["text"] and "공개 정보 부족" in blocks[1]["text"]


def test_judge_locations_are_resolved_to_exact_original_not_model_quotation(state, generator, monkeypatch):
    from agents import report_quality
    state.update(writer.draft(state))
    failed = verdict(1, False, text="의역되거나 비어 있는 문장")
    install_judge(monkeypatch, [failed])
    result = report_quality.run(state)["quality_result"]
    original = read_json(state["report_draft_ref"])["report"]["chapters"][0]["blocks"][0]["text"]
    assert result["recoverable"] and result["issues"][0]["text"] == original


def test_invalid_judge_block_location_is_rejected(state, generator, monkeypatch):
    from schemas import ReportIssue
    failed = verdict(1, False).model_copy(update={"issues": [ReportIssue(chapter=1, block=99, kind="groundedness", reason="오류")]})
    calls = install_judge(monkeypatch, [failed])
    final = graph().invoke(state)
    assert final["report_version"] == 1 and final["execution_status"] == "quality_failed"
    assert len(calls) == 2


def test_hold_judge_context_uses_report_facts_not_last_candidate_score(state, generator, monkeypatch):
    state["decision"] = "보류"
    state["evaluation_history"][0].update(total_score=60.0, decision="보류", knockout=None)
    # Deliberately leave the prior candidate's 80-point score in State.
    calls = install_judge(monkeypatch, [verdict()])
    final = graph().invoke(state)
    assert final["execution_status"] == "complete"
    payload = calls[0]
    assert payload["inputs"]["decision"] == "보류" and payload["inputs"]["total_score"] is None
    assert "scores" not in payload["inputs"] and "tech_summary" not in payload["inputs"]
    assert payload["authoritative_facts"]["evaluation_history"][0]["total_score"] == 60.0
    assert payload["authoritative_facts"]["required_opening"] == "투자 대상 없음(전체 보류). 평가 후보 1곳."


def test_pass_with_failure_issues_is_retried_never_auto_finalized(state, generator, monkeypatch):
    contradictory = verdict().model_copy(update={"issues": verdict(1, False).issues})
    calls = install_judge(monkeypatch, [contradictory])
    final = graph().invoke(state)
    assert len(calls) == 2 and final["execution_status"] == "quality_failed"
    assert calls[1]["validation_feedback"] == "통과 항목에 실패 이슈가 있습니다."
    assert final["report_path"] is None


def test_failure_locations_are_requested_only_after_assessment_fails(state, generator, monkeypatch):
    failed = verdict(1, False).model_copy(update={"issues": []})
    calls = install_judge(monkeypatch, [failed, {"issues": [{"chapter": 1, "block": 1,
        "kind": "groundedness", "reason": "해당 주장 수정"}]}, verdict(2)])
    final = graph().invoke(state)
    assert final["execution_status"] == "complete" and len(calls) == 3
    assert calls[1]["failed_kinds"] == ["groundedness"]
    assert calls[1]["block_catalog"]


def test_translated_amount_survives_report_graph_without_repair(state, generator, monkeypatch):
    source = state["references"][0]["id"]
    write_json(run_dir(state) / "evidence/test.json", [{
        "source_id": source, "startup": "Amperon", "text": "Series B funding: $20 million.",
        "page": 1, "reference": state["references"][0]}])
    original = writer._invoke
    requests = []
    def invoke(model, prompt, payload):
        requests.append(deepcopy(payload))
        if model is ChapterDraft and payload["chapter"] == 1:
            return ChapterDraft(blocks=[{"subtitle": "투자 유치", "text":
                f"공개자료에 따르면 투자 유치액은 2000만 달러다. 단일 출처 의존 한계가 있어 추가 확인이 필요하다. [출처: {source}]",
                "source_ids": [source]}])
        return original(model, prompt, payload)
    monkeypatch.setattr(writer, "_invoke", invoke)
    install_judge(monkeypatch, [verdict(1)])
    final = graph().invoke(state)
    assert final["execution_status"] == "complete"
    assert final["quality_result"]["groundedness"]["passed"]
    assert "2000만 달러" in read_json(final["report_draft_ref"])["report"]["chapters"][0]["blocks"][0]["text"]
    assert not any(p.get("operation") == "repair_block" for p in requests)
