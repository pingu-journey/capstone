"""Hybrid evidence/structure checks and an independent LLM Judge; fail closed."""

import json
import re
from hashlib import sha256
from pathlib import Path

from httpx import HTTPError
from langchain_core.exceptions import OutputParserException
from openai import APIError
from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from agents import report_writer
from config import MAX_CALL_ATTEMPTS, ROOT
from report.markdown_renderer import to_markdown
from report.templates import ReportDocument, TEMPLATES, blocks_length, SUMMARY_BUDGET
from schemas import QualityCheck, QualityVerdict
from tools.artifacts import read_json, run_dir, write_json
from tools.llm import get_judge_llm
from tools.references import unsupported_numbers
from tools.services import ConfigurationError, TransientServiceError, check_service_error


def _check(reasons):
    return QualityCheck(passed=not reasons, reason="검사 통과" if not reasons else "; ".join(reasons), feedback=reasons)


def code_checks(state, artifact, evidence):
    report = ReportDocument.model_validate(artifact["report"])
    route, run_date, _, total, decision, _, _, tables, _, _ = report_writer._prepare(state)
    structure = []
    if artifact["run_id"] != state["run_id"] or artifact["report_version"] != state["report_version"]:
        structure.append("실행·보고서 버전 불일치")
    if report.route != route or report.run_date != run_date or report.decision != decision or report.total_score != total:
        structure.append("확정 판정·총점 불일치")
    if [chapter.title for chapter in report.chapters] != [template.title for template in TEMPLATES[report.route]]:
        structure.append("필수 목차 불일치")
    if len(report.summary) > SUMMARY_BUDGET or artifact["summary_bottom"] > 148.5 or artifact["page_count"] > 5:
        structure.append("SUMMARY 또는 PDF 분량 상한 초과")
    reader = PdfReader(state["report_preview_ref"])
    if len(reader.pages) != artifact["page_count"] or len(reader.pages) > 5:
        structure.append("실제 PDF 페이지 수 불일치")
    pdf_text = re.sub(r"\s+", "", "\n".join(page.extract_text() for page in reader.pages))
    if re.sub(r"\s+", "", report.summary) not in pdf_text:
        structure.append("평가 본문과 PDF SUMMARY 불일치")
    for chapter in report.chapters:
        if any(re.sub(r"\s+", "", block.text) not in pdf_text for block in chapter.blocks):
            structure.append("평가 본문과 실제 PDF 내용 불일치")
        if blocks_length(chapter.blocks) > chapter.char_budget:
            structure.append("장별 글자 예산 초과")
        if [table.model_dump() for table in chapter.tables] != [table.model_dump() for table in tables.get(chapter.number, [])]:
            structure.append("코드 생성 표가 확정 분석과 불일치")
    source_ids = {ref["id"] for items in report.references.values() for ref in items}
    available = {item["source_id"] for item in evidence if item.get("text")}
    grounding = []
    if source_ids - available:
        grounding.append("원문 발췌가 없는 출처를 제거하고 해당 주장을 정보 부족으로 표시하세요.")
    text = to_markdown(report).replace("\\[", "[").replace("\\]", "]")
    # IDs alone do not verify a fact; the Judge checks semantics against the excerpts.
    for chapter in report.chapters:
        for block in chapter.blocks:
            try:
                report_writer._citations(block.text, source_ids)
            except report_writer.ReportValidationError:
                grounding.append("제공되지 않은 인용 ID를 수정하세요.")
            factual_section = (report.route == "investment" and chapter.number <= 3) or (report.route == "hold" and chapter.number == 3)
            groups = re.findall(r"\[출처:\s*([^\]]+)\]", block.text)
            cited = {source.strip() for group in groups for source in group.split(",")}
            claim = re.sub(r"\[출처:[^\]]+\]", "", block.text)
            numbers = set(re.findall(r"\d+(?:\.\d+)?", claim))
            if factual_section and numbers:
                if not cited:
                    grounding.append(f"{chapter.number}장 {block.subtitle}: 인용 없는 수치에 원문 출처를 붙이거나 삭제하세요.")
                else:
                    if unsupported_numbers(block.text, evidence, **report_writer._numeric_assessment(state, report.route, chapter.number)):
                        grounding.append(f"{chapter.number}장 {block.subtitle}: 인용 원문에 없는 수치를 삭제·수정하세요.")
    neutrality = []
    if re.search(r"(?:무조건|반드시)\s*(?:투자|추천)|최고의 기술|압도적(?:인)?\s*우위|위험이 전혀 없", text):
        neutrality.append("무조건적 추천·기술 우열·홍보성 단정을 근거와 불확실성을 구분한 표현으로 수정하세요.")
    bias = []
    organizations = {ref["author_org"].lower() for items in report.references.values() for ref in items}
    if source_ids and len(organizations) == 1 and not any(word in text for word in ("단일 출처", "단일 자료", "한 출처")):
        bias.append("단일 출처 의존 한계를 명시하고 주장을 유보하세요.")
    return report, structure, _check(grounding), _check(neutrality), _check(bias)


def run(state):
    artifact = read_json(state["report_draft_ref"])
    version = state["report_version"]
    if "error_code" in artifact or "report" not in artifact:
        reason = artifact.get("error_reason", "검증 가능한 초안 생성 실패")
        check = _check([reason])
        verdict = QualityVerdict(report_version=version, groundedness=check, neutrality=check, bias=check,
                                 structure_passed=False, structure_reasons=[reason],
                                 recoverable=artifact.get("error_code") in ("ReportValidationError", "ReportTooLongError", "ReportLayoutError"))
        write_json(run_dir(state) / f"quality_v{version}.json", verdict.model_dump())
        return {"quality_result": verdict.model_dump(), "log": [f"[report_quality] version={version} 초안 생성 미달"]}
    try:
        evidence = read_json(state["evidence_index_ref"])
        if artifact["report"]["route"] == "investment":
            evidence = [item for item in evidence if item["startup"] in (state["current_startup"]["name"], "*")]
        report, structure, grounding, neutrality, bias = code_checks(state, artifact, evidence)
        refs = {ref["id"] for items in report.references.values() for ref in items}
        evidence = [item for item in evidence if item["source_id"] in refs and
                    (report.route == "hold" or item["startup"] in (state["current_startup"]["name"], "*"))]
        payload = {"report_version": version, "structure_passed": not structure, "structure_reasons": structure,
                   "report": report.model_dump(), "evidence": evidence,
                   "inputs": {key: state.get(key) for key in ("decision", "total_score", "scores", "evaluation_history", "search_round", "evaluated")}}
        if report.route == "investment":
            payload["inputs"].update(current_startup=state["current_startup"], tech_summary=state["tech_summary"])
        judge = get_judge_llm().with_structured_output(QualityVerdict)
        verdict = None
        for attempt in range(MAX_CALL_ATTEMPTS):
            try:
                output = judge.invoke([
                    ("system", (ROOT / "prompts/report_quality.md").read_text(encoding="utf-8")),
                    ("human", json.dumps(payload, ensure_ascii=False)),
                ])
                verdict = QualityVerdict.model_validate(output)
                if verdict.report_version != version:
                    raise ValueError("품질 판정 버전 불일치")
                break
            except (APIError, HTTPError, TransientServiceError, OutputParserException, ValidationError, ValueError) as error:
                try:
                    check_service_error(error)
                except TransientServiceError:
                    pass
                verdict = None
        if verdict is None:
            failed = _check(["품질 Judge 호출·응답 검증 실패. 자동 통과할 수 없습니다."])
            verdict = QualityVerdict(report_version=version, groundedness=failed, neutrality=failed, bias=failed,
                                     structure_passed=not structure, structure_reasons=structure, recoverable=False)
        else:
            # A model cannot override failed deterministic gates.
            verdict.structure_passed = not structure
            verdict.structure_reasons = structure
            verdict.recoverable = True
            for key, check in (("groundedness", grounding), ("neutrality", neutrality), ("bias", bias)):
                if not check.passed:
                    existing = getattr(verdict, key)
                    setattr(verdict, key, _check(check.feedback + existing.feedback))
        if verdict.passed:
            preview = Path(state["report_preview_ref"])
            artifact["reviewed_pdf_sha256"] = sha256(preview.read_bytes()).hexdigest()
            artifact["reviewed_report_sha256"] = sha256(json.dumps(artifact["report"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            write_json(Path(state["report_draft_ref"]), artifact)
    except (ValueError, OSError, PdfReadError, report_writer.ReportValidationError):
        failed = _check(["초안·근거·확정 결과의 구조 검사 실패"])
        verdict = QualityVerdict(report_version=version, groundedness=failed, neutrality=failed, bias=failed,
                                 structure_passed=False, structure_reasons=failed.feedback, recoverable=False)
    write_json(run_dir(state) / f"quality_v{version}.json", verdict.model_dump())
    return {"quality_result": verdict.model_dump(),
            "log": [f"[report_quality] version={version} {'통과' if verdict.passed else '미달'}"]}


def failed(state):
    return {"execution_status": "quality_failed", "report_path": None,
            "log": [f"[quality_failed] version={state['report_version']} 검토용 초안·사유 보존"]}
