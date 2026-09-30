import re
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader

from agents import report_writer as writer
from report import pdf_renderer as renderer
from report.markdown_renderer import to_markdown
from report.templates import Block, Table, blocks_length
from report_samples import sample_report
from state import InvestState


def normalized(text):
    return re.sub(r"\s+", "", text)


# 실제 PDF의 두 경로·후보 없음·긴 근거가 5쪽 이내이며 한글·서지·표를 보존한다.
@pytest.mark.parametrize("route,count,stress", [("investment", 1, False), ("hold", 3, False),
                                               ("hold", 8, False), ("hold", 0, False), ("investment", 1, True)])
def test_real_pdf_content_and_layout(route, count, stress):
    report = sample_report(route, count, stress)
    before = report.model_dump()
    result = renderer.render_pdf(report)
    reader = PdfReader(BytesIO(result.content))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) == result.page_count <= 5
    assert result.summary_bottom <= 148.5
    assert text.index("SUMMARY") < text.index("1. ") < text.index("REFERENCE")
    assert normalized(report.summary) in normalized(text)
    for chapter in report.chapters:
        for block in chapter.blocks:
            assert normalized(block.text) in normalized(text)
        for table in chapter.tables:
            for row in table.rows:
                for cell in row:
                    assert normalized(cell) in normalized(text)
    for refs in report.references.values():
        for ref in refs:
            assert normalized(ref["url"]) in normalized(text)
    for page in reader.pages:
        assert float(page.mediabox.width) == pytest.approx(595.28, abs=0.1)
        assert float(page.mediabox.height) == pytest.approx(841.89, abs=0.1)
    assert report.model_dump() == before
    assert reader.pages[0].images
    assert all(not page.images for page in reader.pages[1:])


# PDF에 한글 폰트와 Unicode 매핑을 내장해 다른 컴퓨터에서도 텍스트를 유지한다.
def test_font_embedding():
    reader = PdfReader(BytesIO(renderer.render_pdf(sample_report()).content))
    fonts = reader.pages[0]["/Resources"]["/Font"].values()
    for font in fonts:
        font = font.get_object()
        assert "/ToUnicode" in font
        descriptor = font["/DescendantFonts"][0].get_object()["/FontDescriptor"]
        assert "/FontFile2" in descriptor


# 긴 표는 행을 쪼개지 않고 새 페이지에서 머리글을 반복한다.
def test_table_header_repeats():
    report = sample_report()
    table = report.chapters[1].tables[0]
    table.rows = [deepcopy(table.rows[0]) for _ in range(30)]
    for index, row in enumerate(table.rows):
        row[0] = f"검증기업{index}"
    result = renderer.render_pdf(report)
    pages = [normalized(page.extract_text()) for page in PdfReader(BytesIO(result.content)).pages]
    containing = [page for page in pages if "검증기업" in page]
    assert len(containing) > 1
    assert all("단계·상장여부" in page for page in containing)
    assert all(len(re.findall(rf"검증기업{index}(?!\d)", "".join(pages))) == 1 for index in range(30))


# SUMMARY 절반 페이지 제한·지원하지 않는 문자·초대형 행은 잘린 PDF 대신 오류를 낸다.
@pytest.mark.parametrize("issue", ["summary", "glyph", "row"])
def test_layout_errors(issue):
    report = sample_report()
    if issue == "summary":
        report.summary = "줄\n" * 50
    elif issue == "glyph":
        report.chapters[0].blocks[0].text = "지원되지 않는 문자 \U0010ffff"
    else:
        report.chapters[1].tables = [Table(title="초대형 행", columns=["근거"], rows=[["긴 근거 " * 4000]])]
    with pytest.raises(renderer.ReportLayoutError):
        renderer.render_pdf(report)


# 과도한 출처는 삭제하지 않고 5페이지 초과로 단독 저장을 거부한다.
def test_rejects_oversized_document_without_file(tmp_path):
    report = sample_report()
    report.chapters[0].blocks = [Block(subtitle="긴 본문", text="보존할 근거 " * 4000)]
    path = tmp_path / "too-long.pdf"
    with pytest.raises(renderer.ReportTooLongError):
        renderer.write_pdf(report, path)
    assert not path.exists()


# 최소 LangGraph에서 실제 PDF를 저장하고 reducer가 기존 로그를 중복시키지 않는다.
def test_report_node_in_graph(monkeypatch, tmp_path):
    report = sample_report()
    monkeypatch.setattr(writer, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(writer, "build_report", lambda state, reduction=0: report)
    graph = StateGraph(InvestState)
    graph.add_node("report", writer.run)
    graph.add_edge(START, "report")
    graph.add_edge("report", END)
    initial = {"decision": "투자", "log": ["[judge] 기존 기록"]}
    before = deepcopy(initial)
    result = graph.compile().invoke(initial)
    path = Path(result["report_path"])
    assert re.fullmatch(r"report_\d{8}_\d{4}\.pdf", path.name)
    assert len(PdfReader(path).pages) <= 5
    assert path.with_suffix(".md").read_text(encoding="utf-8") == to_markdown(report)
    assert result["log"].count("[judge] 기존 기록") == 1
    assert len(result["log"]) == 2 and initial == before


# 본문 2회 축소 후 글자 크기를 줄이며 마지막 데이터로 PDF와 Markdown을 함께 저장한다.
@pytest.mark.parametrize("pages, reductions, compact", [([5], [0], [False]), ([6, 5], [0, 1], [False, False]),
                                                        ([6, 6, 6, 5], [0, 1, 2], [False, False, False, True])])
def test_page_retries(monkeypatch, tmp_path, pages, reductions, compact):
    reports = []

    def build(state, reduction=0):
        report = sample_report()
        report.chapters[0].blocks[0].text = f"예산 축소 횟수 {reduction}"
        reports.append(report)
        return report

    builder = Mock(side_effect=build)
    render = Mock(side_effect=[renderer.RenderedPDF(b"validated-pdf", count, 90) for count in pages])
    monkeypatch.setattr(writer, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(writer, "build_report", builder)
    monkeypatch.setattr(writer, "render_pdf", render)
    update = writer.run({"log": ["old"]})
    assert set(update) == {"report_path", "log"} and len(update["log"]) == 1
    assert [call.kwargs["reduction"] for call in builder.call_args_list] == reductions
    assert [call.kwargs.get("compact", False) for call in render.call_args_list] == compact
    assert Path(update["report_path"]).with_suffix(".md").read_text(encoding="utf-8") == to_markdown(reports[-1])


# 모든 축소가 실패하면 출처나 기존 산출물을 삭제하지 않고 저장 전에 중단한다.
def test_page_failure_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(writer, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(writer, "build_report", Mock(return_value=sample_report()))
    monkeypatch.setattr(writer, "render_pdf", Mock(return_value=renderer.RenderedPDF(b"overflow", 6, 90)))
    with pytest.raises(renderer.ReportTooLongError):
        writer.run({})
    assert not list(tmp_path.iterdir())


# 같은 분의 기존 산출물을 덮어쓰지 않고 LLM 호출 전 충돌을 알린다.
def test_existing_output_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(writer, "datetime", Mock(now=Mock(return_value=datetime(2026, 9, 30, 12, 0))))
    monkeypatch.setattr(writer, "OUTPUT_DIR", tmp_path)
    build = Mock(return_value=sample_report())
    monkeypatch.setattr(writer, "build_report", build)
    monkeypatch.setattr(writer, "render_pdf", Mock(return_value=renderer.RenderedPDF(b"first", 1, 90)))
    update = writer.run({})
    build.reset_mock()
    with pytest.raises(FileExistsError):
        writer.run({})
    build.assert_not_called()
    assert Path(update["report_path"]).read_bytes() == b"first"


# 두 번째 파일 저장이 실패하면 이번 호출이 만든 첫 파일만 정리한다.
def test_pair_write_failure_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr(writer, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(writer, "build_report", Mock(return_value=sample_report()))
    monkeypatch.setattr(writer, "render_pdf", Mock(return_value=renderer.RenderedPDF(b"pdf", 1, 90)))
    original = Path.open

    def fail_md(path, *args, **kwargs):
        if path.suffix == ".md" and args == ("xb",):
            raise OSError("모의 저장 실패")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_md)
    with pytest.raises(OSError, match="모의 저장 실패"):
        writer.run({})
    assert not list(tmp_path.iterdir())


# 최대 본문 예산과 550자 SUMMARY에서도 실제 PDF가 제한 안에 들어오는지 확인한다.
def test_full_character_budgets():
    report = sample_report()
    for chapter in report.chapters:
        gap = chapter.char_budget - blocks_length(chapter.blocks)
        chapter.blocks[0].text += (" 조판 검증을 위한 충분한 길이의 가상 분석 문장이다." * 80)[:gap]
        assert blocks_length(chapter.blocks) == chapter.char_budget
    report.summary += (" 본문과 점수의 일치 여부를 확인하는 검증 문장이다." * 30)[:550 - len(report.summary)]
    result = renderer.render_pdf(report)
    assert len(report.summary) == 550 and result.page_count <= 5
    assert result.summary_bottom < 148.5
