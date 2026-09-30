"""본문 생성, PDF 분량 검증, PDF/Markdown 저장을 담당하는 보고서 노드."""

import json
import logging
import re
from collections import Counter
from datetime import date, datetime

from httpx import HTTPError
from langchain_core.exceptions import OutputParserException
from openai import APIError
from pydantic import ValidationError

from agents.investment_judge import compute, load_rubric
from config import INVEST_THRESHOLD, OUTPUT_DIR, ROOT
from report.markdown_renderer import to_markdown
from report.pdf_renderer import MAX_PAGES, ReportTooLongError, render_pdf, validate_assets
from report.templates import (
    SUMMARY_BUDGET, TEMPLATES, Block, Chapter, ChapterDraft, HoldSummary,
    InvestmentSummary, ReportDocument, Table, blocks_length, chapter_budget,
)
from schemas import Candidate, CompetitorAnalysis, EvalRecord, ItemScore, MarketAnalysis, SEGMENT_KO, TechSummary
from tools.llm import get_llm
from tools.references import filter_references, group_references

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
    if len(text) > SUMMARY_BUDGET:
        raise ReportValidationError("SUMMARY가 550자를 초과합니다.")
    _citations(text, source_ids)
    return text


def build_report(state, *, reduction: int = 0) -> ReportDocument:
    """State를 변경하지 않고 검증된 공용 보고서 데이터를 생성한다. 파일은 쓰지 않는다."""
    budgets = [chapter_budget(template, reduction) for template in TEMPLATES["investment"]]
    route, run_date, title, total, decision, inputs, summary_input, tables, refs, limitations = _prepare(state)
    source_ids = {ref["id"] for ref in refs}
    # 표/서지는 코드로 만들며 본문 글자 예산에서 제외한다. PDF 단계에서 실제 길이를 검사한다.
    limitation_block = Block(subtitle="분석의 한계", text="\n".join(limitations))
    chapters = []
    logger.info("보고서 본문 생성 시작: route=%s reduction=%d", route, reduction)
    for number, (template, context, budget) in enumerate(zip(TEMPLATES[route], inputs, budgets), 1):
        reserve = blocks_length([limitation_block]) + 2 if number == 4 else 0
        if budget <= reserve:
            raise ReportValidationError("필수 분석 한계가 장별 글자 예산을 초과합니다.")
        payload = {"route": route, "chapter": number, "title": template.title,
                   "guidance": template.guidance, "char_budget": budget - reserve,
                   "inputs": context, "source_ids": sorted(source_ids)}
        # 잘못된 구조·예산·출처는 한 번만 재생성한다. 원문 응답은 재전송하지 않는다.
        for attempt in range(2):
            try:
                draft = _invoke(ChapterDraft, "report_chapter.md", payload)
                if blocks_length(draft.blocks) > budget - reserve:
                    raise ReportValidationError("본문이 장별 글자 예산을 초과합니다.")
                _citations("\n".join(block.subtitle + "\n" + block.text for block in draft.blocks), source_ids)
                break
            except ReportValidationError as exc:
                if attempt:
                    raise
                logger.warning("보고서 본문 검증 실패, 재생성: chapter=%d", number)
                payload["retry_reason"] = str(exc)
        blocks = draft.blocks + ([limitation_block] if number == 4 else [])
        chapters.append(Chapter(number=number, title=template.title, char_budget=budget,
                                blocks=blocks, tables=tables.get(number, [])))
    summary_input["chapters"] = [chapter.model_dump() for chapter in chapters]
    summary_input["source_ids"] = sorted(source_ids)
    model = InvestmentSummary if route == "investment" else HoldSummary
    for attempt in range(2):
        try:
            draft = _invoke(model, f"report_summary_{route}.md", summary_input)
            summary = _summary_text(draft, summary_input, source_ids)
            break
        except ReportValidationError as exc:
            if attempt:
                raise
            logger.warning("SUMMARY 검증 실패, 재생성: route=%s", route)
            summary_input["retry_reason"] = str(exc)
    report = ReportDocument(route=route, title=title, run_date=run_date, decision=decision,
                            total_score=total, summary=summary, chapters=chapters,
                            references=group_references(refs))
    logger.info("보고서 본문 생성 완료: route=%s chapters=%d", route, len(chapters))
    return report


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
            raise ReportTooLongError("본문 2회 축소·글자 크기 축소 후에도 REFERENCE 포함 5페이지를 초과합니다.")
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
