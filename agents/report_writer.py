"""본문 생성, PDF 분량 검증, PDF/Markdown 저장을 담당하는 보고서 노드."""

import json
import logging
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from httpx import HTTPError
from langchain_core.exceptions import OutputParserException
from openai import APIError
from pydantic import ValidationError

from agents.investment_judge import compute, load_rubric
from config import INVEST_THRESHOLD, OUTPUT_DIR, ROOT, MAX_REPORT_IMPROVEMENTS
from report.markdown_renderer import to_markdown
from report.pdf_renderer import MAX_PAGES, ReportLayoutError, ReportTooLongError, render_pdf, validate_assets
from report.templates import (
    SUMMARY_BUDGET, TEMPLATES, Block, Chapter, ChapterDraft, HoldSummary,
    InvestmentSummary, ReportDocument, Table, blocks_length, chapter_budget,
)
from schemas import Candidate, CompetitorAnalysis, EvalRecord, ItemScore, MarketAnalysis, SEGMENT_KO, TechSummary, ReportIssue
from tools.llm import get_llm
from tools.references import filter_references, group_references, unsupported_numbers
from tools.artifacts import read_json, run_dir, write_json
from tools.evidence import index_evidence

logger = logging.getLogger(__name__)
PROMPT_DIR = ROOT / "prompts"


class ReportServiceError(RuntimeError):
    """외부 호출 오류의 원문을 노출하지 않는 호출 경계 예외."""


class ReportValidationError(ValueError):
    """응답·분량·출처·확정 결과 불일치. 오류에는 응답 원문을 넣지 않는다."""


def _invoke(model, prompt_name, payload):
    messages = [
        ("system", (PROMPT_DIR / prompt_name).read_text(encoding="utf-8")),
        ("human", json.dumps(payload, ensure_ascii=False)),
    ]
    try:
        output = get_llm().with_structured_output(model).invoke(messages)
        return model.model_validate(output)
    except (APIError, HTTPError):
        raise ReportServiceError("보고서 API 호출 실패: operation=report") from None
    except (OutputParserException, ValidationError):
        raise ReportValidationError("보고서 구조화 응답 검증 실패") from None


def _citations(text, source_ids):
    for group in re.findall(r"\[출처:\s*([^\]]+)\]", text):
        if any(source.strip() not in source_ids for source in group.split(",")):
            raise ReportValidationError("제공되지 않은 출처 ID가 보고서에 있습니다.")


def _numeric_assessment(state, route, chapter):
    # Only this chapter receives the current company's technical assessment.
    if route != "investment" or chapter != 1:
        return {}
    name = state["current_startup"]["name"]
    return {"tech_summary": state["tech_summary"], "assessment_source_ids": {
        ref["id"] for ref in state.get("references", [])
        if ref["startup"] in (name, "*") and ref["used_by"] == "tech_summary"
    }}


def _validated(model, data):
    # 공통 스키마는 다른 담당자 소유다. 오류 입력 노출은 이 경계에서 차단한다.
    try:
        return model.model_validate(data).model_dump()
    except ValidationError:
        raise ReportValidationError("보고서 입력 스키마 검증 실패") from None


def _prepare(state):
    if state.get("decision") not in (None, "투자", "보류"):
        raise ReportValidationError("보고서 판정은 투자·보류 또는 후보 없음이어야 합니다.")
    route = "investment" if state.get("decision") == "투자" else "hold"
    run_date = date.fromisoformat(state["run_date"]).isoformat()
    rubric = load_rubric()
    limitations = [f"분석 기준일은 {run_date}이며 공개 자료에 한정한 평가다."]
    if route == "investment":
        candidate = _validated(Candidate, state.get("current_startup"))
        tech = _validated(TechSummary, state.get("tech_summary"))
        market = _validated(MarketAnalysis, state.get("market_analysis"))
        competitor = _validated(CompetitorAnalysis, state.get("competitor_analysis"))
        scores = {key: _validated(ItemScore, value) for key, value in state.get("scores", {}).items()}
        total, decision, _ = compute(scores, rubric)
        if total != state.get("total_score") or decision != "투자":
            raise ReportValidationError("확정 총점·판정과 항목 점수가 일치하지 않습니다.")
        for item in rubric:
            score = scores[item["id"]]
            if (score["item_id"] != item["id"] or score["weight"] != item["weight"]
                    or (score["insufficient_info"] and score["score"] != 5)):
                raise ReportValidationError("항목 ID·가중치·정보 부족 점수가 일치하지 않습니다.")
        missing = [item["name"] for item in rubric if scores[item["id"]]["insufficient_info"]]
        if missing:
            limitations.append("공개 정보 부족으로 5점 적용: " + ", ".join(missing) + ".")
        if tech["insufficient"]:
            limitations.append("기술·팀 정보 부족: " + ", ".join(tech["insufficient"]) + ".")
        competitor_insufficient = state["competitor_analysis"].get("insufficient", [])
        if competitor_insufficient:
            limitations.append("경쟁 분석 정보 부족: " + ", ".join(competitor_insufficient) + ".")
        inputs = [
            {"current_startup": candidate, "tech_summary": {k: v for k, v in tech.items() if k != "team"}},
            {"market_analysis": market, "competitor_analysis": competitor},
            {"team": tech["team"], "traction": competitor["traction"], "weaknesses": tech["weaknesses"],
             "policy_risks": market["policy_risks"], "competition_risks": competitor["competition_risks"]},
            {"scores": scores, "total_score": total, "decision": decision, "insufficient_info": missing},
        ]
        refs = filter_references(state.get("references", []), candidate["name"])
        ranked = sorted(rubric, key=lambda item: scores[item["id"]]["score"] * item["weight"], reverse=True)
        summary_input = {
            "decision": decision, "total_score": total,
            "scores": {item["id"]: scores[item["id"]] for item in ranked[:3] + ranked[-2:]},
            "required_opening": f"투자 추천, 총점 {total:.1f}점.",
        }
        tables = {2: [_competitor_table(competitor), _market_table([market])],
                  4: [_score_table(scores, rubric, total)]}
        title = f"{candidate['name']} 투자 평가 보고서"
        markets = [market]
    else:
        history = [_validated(EvalRecord, record) for record in state.get("evaluation_history", [])]
        names = [record["startup"] for record in history]
        if len(names) != len(set(names)):
            raise ReportValidationError("평가 이력에 중복 후보가 있습니다.")
        if any(record["decision"] != "보류" or not 0 <= record["total_score"] <= 100
               or (record["total_score"] >= INVEST_THRESHOLD and not record["knockout"])
               for record in history):
            raise ReportValidationError("전체 보류 경로의 평가 이력·판정이 일치하지 않습니다.")
        cache = {key: _validated(MarketAnalysis, value) for key, value in state.get("market_cache", {}).items()}
        counts = Counter(item for record in history for item in set(record["weakest_items"]))
        common = [f"{item} ({count}/{len(history)}곳의 취약 항목)"
                  for item, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])) if count >= 2]
        if not history:
            common = ["평가 가능한 후보를 확보하지 못함"]
        elif not common:
            common = ["평가 이력에서 반복되는 공통 취약 항목을 확인하지 못함. 후보별 보류 사유는 비교표 참조."]
        limitations.append("이전 후보의 상세 분석·항목별 점수는 보존되지 않아 누적 평가 이력과 시장 캐시만 사용했다.")
        missing_records = [record["startup"] + ": " + record["key_reason"]
                           for record in history if "정보 부족" in record["key_reason"]]
        if missing_records:
            limitations.append("정보 부족으로 5점을 적용한 후보별 항목은 아래 공개 정보 한계 표에 기록했다.")
        if not cache:
            limitations.append("보존된 시장 분석이 없어 시장·규제 환경의 공개 정보가 부족하다.")
        inputs = [
            {"domain": state.get("domain", "Energy"),
             "requirements": "비상장, Seed~Series C, M&A·IPO 등 Exit 미완료",
             "evaluated": names, "search_round": state.get("search_round", 0)},
            {"evaluation_history": history},
            {"hold_reasons": [{k: record[k] for k in ("startup", "weakest_items", "key_reason")}
                              for record in history], "market_cache": cache},
            {"limitations": limitations, "run_date": run_date},
        ]
        refs = filter_references(state.get("references", []))
        summary_input = {
            "evaluation_history": history, "candidate_count": len(history),
            "expected_common_reasons": common,
            "required_opening": f"투자 대상 없음(전체 보류). 평가 후보 {len(history)}곳.",
        }
        tables = {2: [_candidate_table(history)], 3: [_market_table(list(cache.values()))]}
        if missing_records:
            tables[4] = [Table(title="후보별 공개 정보 한계", columns=["평가 이력에 기록된 한계"],
                               rows=[[reason] for reason in missing_records])]
        title, total, decision = "Energy AI 스타트업 투자 평가 결과", None, "보류"
        markets = list(cache.values())
    source_ids = {ref["id"] for ref in refs}
    for market in markets:
        for figure in market["market_size"]:
            if figure["source_id"] not in source_ids or not figure["year"].strip():
                raise ReportValidationError("시장 수치의 출처 또는 기준 연도가 없습니다.")
    if route == "investment":
        for score in scores.values():
            _citations(score["rationale"], source_ids)
    return route, run_date, title, total, decision, inputs, summary_input, tables, refs, limitations


def _competitor_table(analysis):
    return Table(title="경쟁사 비교", columns=["기업", "국가", "단계·상장 여부", "제공 기술", "대상 대비 차이"],
                 rows=[[item[key] for key in ("name", "country", "status", "offering", "vs_target")]
                       for item in analysis["competitors"]])


def _market_table(markets):
    return Table(title="시장 수치", columns=["국가", "세그먼트", "지표", "값·단위", "기준 연도", "출처 ID"],
                 rows=[[market["country"], SEGMENT_KO[market["segment"]], figure["metric"], figure["value"],
                        figure["year"], figure["source_id"]]
                       for market in markets for figure in market["market_size"]])


def _score_table(scores, rubric, total):
    rows = []
    for item in rubric:
        entry = scores[item["id"]]
        rows.append([item["name"], str(item["weight"]), str(entry["score"]),
                     f"{entry['score'] * item['weight'] / 10:.1f}", entry["rationale"]])
    rows.append(["합계", "100", "-", f"{total:.1f}", f"투자 기준 {INVEST_THRESHOLD}점 / 투자"])
    return Table(title="투자 평가 점수 (원점수 0~10, 총점 0~100)",
                 columns=["평가 항목", "가중치 (%)", "원점수", "가중 점수", "근거"], rows=rows)


def _candidate_table(history):
    return Table(title=f"후보별 평가 결과 (0~100점 · 투자 기준 {INVEST_THRESHOLD}점 · 전체 보류)",
                 columns=["기업", "총점", "판정", "취약 항목", "필수 탈락", "보류 사유"],
                 rows=[[record["startup"], f"{record['total_score']:.1f}", record["decision"],
                        ", ".join(record["weakest_items"]), record["knockout"] or "없음", record["key_reason"]]
                       for record in history])


def _summary_text(draft, payload, source_ids):
    if set(draft.source_ids) - source_ids:
        raise ReportValidationError("SUMMARY에 제공되지 않은 출처 ID가 있습니다.")
    if isinstance(draft, InvestmentSummary):
        if not draft.conclusion.startswith(payload["required_opening"]):
            raise ReportValidationError("SUMMARY 첫 문장의 판정·총점이 일치하지 않습니다.")
        if "보류" in draft.conclusion:
            raise ReportValidationError("SUMMARY 결론에 상충하는 판정이 있습니다.")
        if any(not text.strip() for text in draft.investment_points + draft.risks):
            raise ReportValidationError("SUMMARY 투자 포인트·리스크가 비어 있습니다.")
        if len(set(draft.investment_points)) != 3 or len(set(draft.risks)) != 2:
            raise ReportValidationError("SUMMARY 투자 포인트·리스크가 중복됩니다.")
        text = draft.conclusion + "\n투자 포인트: " + "; ".join(draft.investment_points)
        text += "\n리스크: " + "; ".join(draft.risks)
    else:
        if draft.conclusion != payload["required_opening"] or draft.candidate_count != payload["candidate_count"]:
            raise ReportValidationError("SUMMARY 보류 결론·후보 수가 일치하지 않습니다.")
        if draft.common_reasons != payload["expected_common_reasons"]:
            raise ReportValidationError("SUMMARY 공통 보류 사유가 평가 이력과 일치하지 않습니다.")
        text = draft.conclusion + "\n공통 보류 사유: " + "; ".join(draft.common_reasons)
    if draft.source_ids:
        text += "\n[출처: " + ", ".join(dict.fromkeys(draft.source_ids)) + "]"
    if re.search(r"(?:무조건|반드시)\s*(?:투자|추천)|최고의 기술|압도적(?:인)?\s*우위|위험이 전혀 없", text):
        raise ReportValidationError("SUMMARY 홍보성·무조건적 추천 표현을 삭제하세요.")
    _citations(text, source_ids)
    return text


def block_issues(state, route, chapter, index, block, evidence, source_ids, budget=None):
    """Locate deterministic failures without globally banning a numeric value."""
    text = block.subtitle + "\n" + block.text
    cited = {source.strip() for group in re.findall(r"\[출처:\s*([^\]]+)\]", text)
             for source in group.split(",")}
    ids = sorted(cited | set(block.source_ids))
    issues = []
    def add(kind, reason, numbers=()):
        issues.append(ReportIssue(chapter=chapter, block=index, kind=kind, text=block.text,
                                  source_ids=ids, numbers=sorted(numbers), reason=reason).model_dump())
    if (cited | set(block.source_ids)) - source_ids:
        add("citation", "제공되지 않은 인용 출처 ID를 삭제하세요.")
    factual = (route == "investment" and chapter <= 3) or (route == "hold" and chapter == 3)
    if factual:
        missing = unsupported_numbers(text, evidence, **_numeric_assessment(state, route, chapter))
        if missing:
            add("numeric", "인용 발췌에 없는 수치: " + ", ".join(sorted(missing)) +
                ". 해당 주장을 삭제하거나 공개 정보 부족으로 표시하세요.", missing)
    if re.search(r"(?:무조건|반드시)\s*(?:투자|추천)|최고의 기술|압도적(?:인)?\s*우위|위험이 전혀 없", text):
        add("neutrality", "홍보성·무조건적 추천 표현을 삭제하고 판단을 유보하세요.")
    return issues


def _fallback_block():
    return Block(subtitle="근거 확인 한계", text="공개 정보 부족으로 해당 주장을 제외했다. 자료의 한계로 추가 확인이 필요하다.")


def _repair_block(state, route, number, index, block, issues, payload, evidence, source_ids, budget, repair_log, *, quality=False, force_fallback=False):
    request = {**payload, "operation": "repair_block", "max_blocks": 1,
               "previous_block": block.model_dump(), "violations": issues,
               "cited_evidence": [item for item in evidence if item["source_id"] in
                                  {source for issue in issues for source in issue.get("source_ids", [])}],
               "char_budget": budget, "target_chars": min(payload["target_chars"], budget),
               "previous_char_count": blocks_length([block]),
               "retry_reason": "; ".join(item["reason"] for item in issues)}
    if any(item["kind"] == "length" for item in issues):
        request["target_chars"] = int(request["target_chars"] * 0.7)
    repaired = None
    after = []
    if not force_fallback:
        try:
            result = _invoke(ChapterDraft, "report_chapter.md", request)
            if len(result.blocks) != 1:
                raise ReportValidationError("수정 응답은 해당 블록 하나만 반환해야 합니다.")
            repaired = result.blocks[0]
            if repaired.source_ids and "[출처:" not in repaired.text:
                repaired.text += " [출처: " + ", ".join(dict.fromkeys(repaired.source_ids)) + "]"
            after = block_issues(state, route, number, index, repaired, evidence, source_ids, budget)
        except ReportValidationError:
            repaired = None
    if repaired is None or after:
        repaired = _fallback_block()
        action = "fallback"
    else:
        action = "repaired"
    repair_log.append({"chapter": number, "block": index, "action": action,
                       "quality_repair": quality, "issues": issues, "repair_issues": after,
                       "before": block.model_dump(), "after": repaired.model_dump()})
    return repaired


def _previous_report(state, route, title, total, decision):
    if not state.get("quality_result") or not state.get("report_draft_ref"):
        return None
    artifact = read_json(state["report_draft_ref"])
    if artifact["run_id"] != state["run_id"] or artifact["report_version"] != state["report_version"]:
        raise ReportValidationError("수정 대상의 실행·버전이 일치하지 않습니다.")
    if state["quality_result"]["report_version"] != state["report_version"]:
        raise ReportValidationError("수정 피드백 버전이 일치하지 않습니다.")
    if "report" not in artifact:
        return None
    previous = ReportDocument.model_validate(artifact["report"])
    if (previous.route != route or previous.run_date != state["run_date"] or previous.title != title
            or previous.total_score != total or previous.decision != decision):
        raise ReportValidationError("수정 대상의 경로·기준일이 일치하지 않습니다.")
    return artifact


def _summary_numbers(draft, summary_input, evidence):
    if not isinstance(draft, InvestmentSummary):
        return set()  # HoldSummary is compared exactly with frozen State history.
    external = "\n".join([draft.conclusion.removeprefix(summary_input["required_opening"]),
                           *draft.investment_points, *draft.risks])
    if draft.source_ids:
        external += " [출처: " + ", ".join(draft.source_ids) + "]"
    return unsupported_numbers(external, evidence)


def _fallback_summary(model, payload):
    if model is HoldSummary:
        return HoldSummary(conclusion=payload["required_opening"], candidate_count=payload["candidate_count"],
                           common_reasons=payload["expected_common_reasons"])
    return InvestmentSummary(conclusion=payload["required_opening"],
        investment_points=["기술 평가 근거는 본문과 표에 기록했다.", "시장 평가 근거는 본문과 표에 기록했다.", "팀 평가 근거는 본문과 표에 기록했다."],
        risks=["공개 정보가 부족한 항목은 추가 실사가 필요하다.", "원문으로 확인되지 않은 주장은 제외했다."])


def build_report(state, *, reduction: int = 0, repair_log=None) -> ReportDocument:
    """Preserve healthy blocks; repair rejected blocks once, then remove their claims."""
    repair_log = repair_log if repair_log is not None else []
    route, run_date, title, total, decision, inputs, summary_input, tables, refs, limitations = _prepare(state)
    source_ids = {ref["id"] for ref in refs}
    evidence = read_json(state["evidence_index_ref"]) if state.get("evidence_index_ref") else []
    evidence = [item for item in evidence if item["source_id"] in source_ids]
    feedback = state.get("quality_result") or {}
    previous = _previous_report(state, route, title, total, decision)
    if refs and len({ref["author_org"].lower() for ref in refs}) == 1:
        limitations.append("단일 출처 기관의 자료에 의존하며 독립 검증에 한계가 있다.")
    limitation_block = Block(subtitle="분석의 한계", text="\n".join(limitations))
    chapters = []
    logger.info("보고서 본문 생성 시작: route=%s reduction=%d", route, reduction)
    for number, (template, context) in enumerate(zip(TEMPLATES[route], inputs), 1):
        budget = chapter_budget(template, reduction)
        reserve = blocks_length([limitation_block]) + 2 if number == 4 else 0
        payload = {"route": route, "chapter": number, "title": template.title, "guidance": template.guidance,
                   "char_budget": max(120, budget - reserve), "target_chars": max(120, int((budget - reserve) * 0.7)),
                   "inputs": context, "source_ids": sorted(source_ids), "max_blocks": 2, "sentences_per_block": 2}
        payload["max_chars_per_block"] = max(60, (payload["target_chars"] - 120) // 2)
        payload.update(evidence=evidence, quality_feedback=feedback)
        if previous:
            old = previous["report"]["chapters"][number - 1]
            blocks = [Block.model_validate(item) for item in old["blocks"]
                      if not (number == 4 and item["subtitle"] == "분석의 한계")]
        else:
            for attempt in range(2):
                try:
                    blocks = _invoke(ChapterDraft, "report_chapter.md", payload).blocks
                    break
                except ReportValidationError:
                    if attempt:
                        raise
                    payload["retry_reason"] = "구조화 응답 검증 실패. 올바른 blocks를 반환하세요."
        accepted = []
        for index, block in enumerate(blocks, 1):
            if block.source_ids and "[출처:" not in block.text:
                block.text += " [출처: " + ", ".join(dict.fromkeys(block.source_ids)) + "]"
            remaining = max(120, budget - reserve - blocks_length(accepted) - (2 if accepted else 0))
            local = block_issues(state, route, number, index, block, evidence, source_ids, remaining)
            judged = [item for item in feedback.get("issues", []) if item["chapter"] == number and item["block"] == index]
            issues = local + judged
            if issues:
                repaired_before = bool(previous and any(item.get("quality_repair") and item["chapter"] == number and item["block"] == index
                                                       for item in previous.get("repair_log", [])))
                block = _repair_block(state, route, number, index, block, issues, payload, evidence, source_ids,
                                      remaining, repair_log, quality=bool(judged), force_fallback=repaired_before)
            accepted.append(block)
        chapter_blocks = accepted + ([limitation_block] if number == 4 else [])
        chapters.append(Chapter(number=number, title=template.title, char_budget=budget,
                                blocks=chapter_blocks, tables=tables.get(number, [])))
    summary_input.update(chapters=[chapter.model_dump() for chapter in chapters], source_ids=sorted(source_ids),
                         evidence=evidence, quality_feedback=feedback, target_chars=300, char_budget=SUMMARY_BUDGET)
    if previous:
        summary_input["previous_summary"] = previous["report"]["summary"]
    model = InvestmentSummary if route == "investment" else HoldSummary
    summary_issues = []
    force_summary = bool(previous and any(item["chapter"] == 0 for item in feedback.get("issues", [])) and
                         any(item["chapter"] == 0 and item.get("quality_repair") for item in previous.get("repair_log", [])))
    for attempt in range(2):
        draft = None
        missing = set()
        try:
            if force_summary:
                raise ReportValidationError("동일 SUMMARY 품질 재실패")
            draft = _invoke(model, f"report_summary_{route}.md", summary_input)
            summary = _summary_text(draft, summary_input, source_ids)
            missing = _summary_numbers(draft, summary_input, evidence)
            if missing:
                raise ReportValidationError("SUMMARY 인용 발췌에 없는 수치: " + ", ".join(sorted(missing)))
            break
        except ReportValidationError as error:
            text = draft.model_dump_json() if draft is not None else "SUMMARY 구조 검증 실패"
            issue = ReportIssue(chapter=0, block=1, kind="numeric" if missing else "structure", text=text,
                                source_ids=draft.source_ids if draft is not None else [],
                                numbers=sorted(missing), reason=str(error)).model_dump()
            summary_issues.append(issue)
            summary_input.update(previous_summary=text, retry_reason=str(error), violations=summary_issues, target_chars=220)
            if attempt or force_summary:
                draft = _fallback_summary(model, summary_input)
                summary = _summary_text(draft, summary_input, source_ids)
                repair_log.append({"chapter": 0, "block": 1, "action": "fallback", "quality_repair": bool(feedback),
                                   "issues": summary_issues, "after": draft.model_dump()})
                break
    if previous and any(item["chapter"] == 0 for item in feedback.get("issues", [])) and not any(item["chapter"] == 0 for item in repair_log):
        repair_log.append({"chapter": 0, "block": 1, "action": "repaired", "quality_repair": True,
                           "issues": [item for item in feedback["issues"] if item["chapter"] == 0],
                           "before": previous["report"]["summary"], "after": summary})
    report = ReportDocument(route=route, title=title, run_date=run_date, decision=decision, total_score=total,
                            summary=summary, chapters=chapters, references=group_references(refs))
    logger.info("보고서 본문 생성 완료: route=%s chapters=%d", route, len(chapters))
    return report


def draft(state):
    """Render the complete revision before quality review; never finalize here."""
    version = state.get("report_version", 0) + 1
    if version > MAX_REPORT_IMPROVEMENTS + 1:
        raise ReportValidationError("보고서 개선 상한 초과")
    evidence_ref = index_evidence(state)
    inputs = {**state, "evidence_index_ref": evidence_ref}
    directory = run_dir(state)
    path = directory / f"draft_v{version}.json"
    report = None
    repair_log = []
    try:
        validate_assets()
        for reduction in range(3):
            report = build_report(inputs, reduction=reduction, repair_log=repair_log)
            rendered = render_pdf(report)
            if rendered.page_count <= MAX_PAGES:
                break
        else:
            rendered = render_pdf(report, compact=True)
            if rendered.page_count > MAX_PAGES:
                raise ReportTooLongError(f"분량 축소 후에도 {MAX_PAGES}쪽 초과")
        write_json(path, {"run_id": state["run_id"], "report_version": version,
                          "page_count": rendered.page_count, "summary_bottom": rendered.summary_bottom,
                          "report": report.model_dump(), "repair_log": repair_log})
        preview = directory / f"preview_v{version}.pdf"
        preview.write_bytes(rendered.content)
        preview.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
        preview_ref = str(preview)
    except (ReportValidationError, ReportServiceError, ReportLayoutError) as error:
        artifact = {"run_id": state["run_id"], "report_version": version,
                    "error_code": type(error).__name__, "error_reason": str(error), "repair_log": repair_log}
        if report is not None:
            artifact["report"] = report.model_dump()
            (directory / f"preview_v{version}.md").write_text(to_markdown(report), encoding="utf-8")
        write_json(path, artifact)
        preview_ref = None
    return {"report_version": version, "report_draft_ref": str(path), "report_preview_ref": preview_ref,
            "evidence_index_ref": evidence_ref, "quality_result": None,
            "log": [f"[report_draft] version={version} 초안 저장"]}


def finalize(state):
    """Publish exactly the reviewed PDF/MD revision, with no post-review regeneration."""
    from hashlib import sha256
    from schemas import QualityVerdict
    verdict = QualityVerdict.model_validate(state["quality_result"])
    artifact = read_json(state["report_draft_ref"])
    if not verdict.passed or verdict.report_version != state["report_version"] or artifact["report_version"] != verdict.report_version:
        raise ReportValidationError("통과한 동일 버전만 확정할 수 있습니다.")
    preview = Path(state["report_preview_ref"])
    if artifact.get("reviewed_pdf_sha256") != sha256(preview.read_bytes()).hexdigest():
        raise ReportValidationError("품질 평가 후 PDF가 변경되었습니다.")
    if artifact.get("reviewed_report_sha256") != sha256(json.dumps(artifact["report"], sort_keys=True, ensure_ascii=False).encode()).hexdigest():
        raise ReportValidationError("품질 평가 후 본문이 변경되었습니다.")
    report = ReportDocument.model_validate(artifact["report"])
    path = run_dir(state) / f"report_v{verdict.report_version}.pdf"
    with path.open("xb") as stream:
        stream.write(preview.read_bytes())
    try:
        with path.with_suffix(".md").open("x", encoding="utf-8") as stream:
            stream.write(to_markdown(report))
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return {"report_path": str(path), "execution_status": "complete",
            "log": [f"[report_finalize] version={verdict.report_version} 품질 통과 보고서 저장"]}


def run(state) -> dict:
    """그래프 계약대로 PDF 경로와 새 로그만 반환한다. 입력 State는 보존한다."""
    validate_assets()
    stem = "report_" + datetime.now().strftime("%Y%m%d_%H%M")
    pdf_path, md_path = OUTPUT_DIR / f"{stem}.pdf", OUTPUT_DIR / f"{stem}.md"
    if pdf_path.exists() or md_path.exists():
        raise FileExistsError("같은 분에 생성된 보고서가 있습니다. 기존 파일을 보존하며 다음 분에 다시 실행하세요.")
    for reduction in range(3):
        report = build_report(state, reduction=reduction)
        rendered = render_pdf(report)
        if rendered.page_count <= MAX_PAGES:
            break
        if reduction < 2:
            logger.warning("PDF 분량 초과, 본문 예산 20%% 축소: pages=%d reduction=%d",
                           rendered.page_count, reduction + 1)
    else:
        logger.warning("PDF 분량 초과, 글자 크기 축소: pages=%d", rendered.page_count)
        rendered = render_pdf(report, compact=True)
        if rendered.page_count > MAX_PAGES:
            raise ReportTooLongError(f"본문 2회 축소·글자 크기 축소 후에도 REFERENCE 포함 {MAX_PAGES}페이지를 초과합니다.")
    markdown = to_markdown(report).encode("utf-8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, content in ((pdf_path, rendered.content), (md_path, markdown)):
            # 경쟁 실행이 먼저 만든 산출물도 덮어쓰지 않는다.
            with path.open("xb") as stream:
                created.append(path)
                stream.write(content)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    logger.info("보고서 저장 완료: report_path=%s pages=%d", pdf_path, rendered.page_count)
    return {"report_path": str(pdf_path), "log": [f"[report] {report.decision}: {pdf_path.name}, {rendered.page_count}페이지"]}
