"""SUIT 기반 A4 조판. 본문·표·서지 데이터는 Markdown과 공유한다."""

import logging
from dataclasses import dataclass
from pathlib import Path

from fpdf import FPDF

from config import INVEST_THRESHOLD, ROOT
from report.templates import ReportDocument, SUMMARY_BUDGET, Table
from tools.references import REFERENCE_GROUPS, format_reference

logger = logging.getLogger(__name__)
ASSET_DIR = ROOT / "assets"
BACKGROUND = (245, 245, 245)
MAIN = (38, 38, 43)
INK = (26, 26, 26)
RED = (230, 43, 30)
MUTED = (98, 98, 104)
LINE = (218, 218, 221)
WHITE = (255, 255, 255)
MARGIN = 15
WIDTH = 180
MAX_PAGES = 5


class ReportLayoutError(ValueError):
    """내용을 삭제하지 않고는 조판할 수 없는 입력."""


class ReportTooLongError(ReportLayoutError):
    """분량·크기 축소 후에도 페이지 상한을 초과한 보고서."""


@dataclass(frozen=True)
class RenderedPDF:
    content: bytes
    page_count: int
    summary_bottom: float


def validate_assets():
    required = [ASSET_DIR / "fonts" / f"SUIT-{weight}.ttf" for weight in ("Regular", "SemiBold", "Bold")]
    required += [ASSET_DIR / "fonts/OFL.txt", ASSET_DIR / "skala-logo.png"]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("SUIT 정적 TTF·라이선스 또는 SKALA 로고 자산이 없습니다.")


class _ReportPDF(FPDF):
    def __init__(self, compact):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(False, margin=MARGIN)
        self.c_margin = 0
        self.body_size = 9.5 if compact else 10
        self.table_size = 8 if compact else 8.5
        self.body_line = self.body_size * 0.49
        self.table_line = self.table_size * 0.46
        for family, weight in (("SUIT", "Regular"), ("SUIT-SemiBold", "SemiBold"), ("SUIT-Bold", "Bold")):
            self.add_font(family, fname=ASSET_DIR / "fonts" / f"SUIT-{weight}.ttf")
        self.alias_nb_pages()

    def header(self):
        self.set_fill_color(*WHITE)
        self.rect(0, 0, self.w, self.h, style="F")
        self.set_xy(MARGIN, MARGIN)

    def footer(self):
        self.set_draw_color(*LINE)
        self.line(MARGIN, self.h - 12, self.w - MARGIN, self.h - 12)
        self.set_xy(MARGIN, self.h - 10)
        self.set_font("SUIT", size=8)
        self.set_text_color(*MUTED)
        self.cell(WIDTH, 4, f"ENERGY AI  /  {self.page_no()} / {{nb}}", align="R")

    def ensure(self, height):
        if height > self.h - 2 * MARGIN:
            raise ReportLayoutError("한 페이지 높이를 초과한 블록 또는 표 행입니다.")
        if self.y + height > self.h - MARGIN:
            self.add_page()

    def lines(self, text, width, size, family="SUIT"):
        self.set_font(family, size=size)
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
        missing = {ord(char) for char in text if char != "\n" and ord(char) not in self.current_font.cmap}
        if missing:
            raise ReportLayoutError("SUIT에 없는 글리프가 있습니다: " + ", ".join(f"U+{code:04X}" for code in sorted(missing)))
        return self.multi_cell(width, size * 0.5, text, wrapmode="CHAR", dry_run=True, output="LINES")

    def line_text(self, text, x, y, width, height, size, family="SUIT", color=INK):
        self.set_font(family, size=size)
        self.set_text_color(*color)
        self.set_xy(x, y)
        self.cell(width, height, text)

    def paragraph(self, text, *, size=None, color=INK, keep=False):
        size = size or self.body_size
        line_height = size * 0.49
        lines = self.lines(text, WIDTH, size)
        if keep and len(lines) * line_height <= self.h - 2 * MARGIN:
            self.ensure(len(lines) * line_height)
        for line in lines:
            self.ensure(line_height)
            y = self.y
            self.line_text(line, MARGIN, y, WIDTH, line_height, size, color=color)
            self.set_xy(MARGIN, y + line_height)
        self.set_y(self.y + 2)

    def heading(self, text, *, chapter=False, following=10):
        size, line_height = (16, 8) if chapter else (11, 6)
        family = "SUIT-Bold" if chapter else "SUIT-SemiBold"
        lines = self.lines(text, WIDTH, size, family)
        self.ensure(len(lines) * line_height + following + 5)
        if chapter:
            self.set_draw_color(*LINE)
            self.line(MARGIN, self.y + 1, self.w - MARGIN, self.y + 1)
        self.set_y(self.y + 4)
        for line in lines:
            y = self.y
            self.line_text(line, MARGIN, y, WIDTH, line_height, size, family)
            self.set_xy(MARGIN, y + line_height)
        self.set_y(self.y + 1)

    def opening(self, report):
        title_lines = self.lines(report.title, 142, 22, "SUIT-Bold")
        height = len(title_lines) * 10 + 17
        self.set_fill_color(*WHITE)
        self.rect(MARGIN, MARGIN, WIDTH, height, style="F")
        self.image(ASSET_DIR / "skala-logo.png", x=self.w - MARGIN - 28, y=MARGIN + 2, w=28)
        for index, line in enumerate(title_lines):
            self.line_text(line, MARGIN + 2, MARGIN + 2 + index * 10, 140, 10, 22, "SUIT-Bold")
        self.line_text(f"작성일 {report.run_date}   |   판정: {report.decision}", MARGIN + 2,
                       MARGIN + height - 10, WIDTH - 4, 6, 9, color=MUTED)
        self.set_y(MARGIN + height + 4)
        if len(report.summary) > SUMMARY_BUDGET:
            raise ReportLayoutError("SUMMARY가 550자를 초과합니다.")
        lines = self.lines(report.summary, WIDTH - 12, self.body_size)
        height = 17 + len(lines) * self.body_line
        bottom = self.y + height
        if bottom > self.h / 2:
            raise ReportLayoutError("SUMMARY가 첫 페이지 절반을 초과합니다.")
        self.set_fill_color(*INK)
        y = self.y
        self.rect(MARGIN, y, WIDTH, height, style="F")
        self.set_fill_color(*RED)
        self.rect(MARGIN, y, 1.2, height, style="F")
        self.line_text("SUMMARY", MARGIN + 6, y + 4, WIDTH - 12, 6, 11, "SUIT-Bold", BACKGROUND)
        for index, line in enumerate(lines):
            self.line_text(line, MARGIN + 6, y + 12 + index * self.body_line, WIDTH - 12,
                           self.body_line, self.body_size, color=BACKGROUND)
        self.set_xy(MARGIN, bottom + 4)
        return bottom

    def bars(self, values, *, title, maximum=100, threshold=None, highlight=None):
        self.heading(title, following=20)
        x, width = MARGIN + 52, WIDTH - 83
        for index, (label, value) in enumerate(values):
            label_lines = self.lines(label, 49, 9)
            height = max(11, len(label_lines) * 4.5 + 2)
            self.ensure(height + 6)
            y = self.y
            for offset, line in enumerate(label_lines):
                self.line_text(line, MARGIN, y + offset * 4.5, 49, 4.5, 9)
            self.set_fill_color(*LINE)
            self.rect(x, y + 1, width, 4, style="F")
            self.set_fill_color(*(RED if highlight == index else MAIN))
            if value:
                self.rect(x, y + 1, width * value / maximum, 4, style="F")
            if threshold is not None:
                self.set_draw_color(*INK)
                self.line(x + width * threshold / maximum, y, x + width * threshold / maximum, y + 6)
            self.line_text(f"{value:.1f} / {maximum}", x + width + 3, y, 28, 6, 9, "SUIT-SemiBold")
            self.set_xy(MARGIN, y + height)
        self.line_text(f"척도 0~{maximum}" + (f" · 투자 기준 {threshold}점" if threshold is not None else ""),
                       x, self.y, WIDTH - 52, 5, 8, color=MUTED)
        self.set_xy(MARGIN, self.y + 7)

    def table(self, table: Table):
        count = len(table.columns)
        if not count or any(len(row) != count for row in table.rows):
            raise ReportLayoutError("표의 열 수와 행 데이터가 일치하지 않습니다.")
        score_table = table.columns == ["평가 항목", "가중치 (%)", "원점수", "가중 점수", "근거"]
        if score_table:
            widths = [32, 15, 16, 18, 99]
        elif table.columns[0] == "기업" and count == 6:
            widths = [24, 15, 13, 32, 24, 72]
        elif table.columns[0] == "기업" and count == 5:
            widths = [27, 13, 25, 47, 68]
        elif count == 6:
            widths = [13, 31, 41, 30, 20, 45]
        else:
            widths = [WIDTH / count] * count

        def layout(values, header=False):
            if header:
                values = [value.replace("가중치 (%)", "가중치\n(%)") for value in values]
            lines = [self.lines(value, width - 4, self.table_size, "SUIT-SemiBold" if header else "SUIT")
                     for value, width in zip(values, widths)]
            height = max(len(cell) for cell in lines) * self.table_line + 4
            if score_table and not header:
                height = max(height, 12)
            return lines, height

        def draw(lines, height, header=False, index=0, values=None):
            x, y = MARGIN, self.y
            self.set_fill_color(*(MAIN if header else (WHITE if index % 2 == 0 else BACKGROUND)))
            self.rect(x, y, WIDTH, height, style="F")
            for column, (cell, width) in enumerate(zip(lines, widths)):
                for offset, text in enumerate(cell):
                    self.line_text(text, x + 2, y + 2 + offset * self.table_line, width - 4,
                                   self.table_line, self.table_size,
                                   "SUIT-SemiBold" if header else "SUIT", BACKGROUND if header else INK)
                if score_table and not header and column == 2 and values[0] != "합계":
                    value = float(values[2])
                    self.set_fill_color(*LINE)
                    self.rect(x + 2, y + height - 3, width - 4, 1, style="F")
                    self.set_fill_color(*MAIN)
                    self.rect(x + 2, y + height - 3, (width - 4) * value / 10, 1, style="F")
                x += width
            self.set_draw_color(*LINE)
            self.line(MARGIN, y + height, self.w - MARGIN, y + height)
            self.set_xy(MARGIN, y + height)

        header_lines, header_height = layout(table.columns, True)
        prepared = [(row, *layout(row)) for row in table.rows]
        first_height = prepared[0][2] if prepared else 5
        self.heading(table.title, following=header_height + first_height)
        if not prepared:
            self.paragraph("공개 정보 부족: 표시할 데이터가 없습니다.", size=self.table_size, color=MUTED)
            return
        draw(header_lines, header_height, True)
        for index, (values, lines, height) in enumerate(prepared):
            if header_height + height > self.h - 2 * MARGIN:
                raise ReportLayoutError("표 한 행이 한 페이지 높이를 초과합니다. 근거를 삭제하지 않고 조판할 수 없습니다.")
            if self.y + height > self.h - MARGIN:
                self.add_page()
                draw(header_lines, header_height, True)
            draw(lines, height, index=index, values=values)
        self.set_y(self.y + 3)


def render_pdf(report: ReportDocument, *, compact: bool = False) -> RenderedPDF:
    """메모리에 조판한다. 페이지 상한 판정·재생성은 보고서 노드가 맡는다."""
    validate_assets()
    pdf = _ReportPDF(compact)
    pdf.set_title(report.title)
    pdf.set_author("Energy AI Startup Evaluation")
    pdf.add_page()
    summary_bottom = pdf.opening(report)
    if report.total_score is not None:
        pdf.bars([(report.decision, report.total_score)], title="종합 투자 점수", threshold=INVEST_THRESHOLD, highlight=0)
    for chapter in report.chapters:
        pdf.heading(f"{chapter.number}. {chapter.title}", chapter=True, following=18)
        for block in chapter.blocks:
            pdf.heading(block.subtitle)
            pdf.paragraph(block.text)
        for table in chapter.tables:
            if report.route == "hold" and table.columns[:3] == ["기업", "총점", "판정"] and table.rows:
                values = [(row[0], float(row[1])) for row in table.rows]
                best = max(range(len(values)), key=lambda index: values[index][1])
                pdf.bars(values, title="후보 총점 비교 · 전체 보류 (빨강: 최고점 후보)",
                         threshold=INVEST_THRESHOLD, highlight=best)
            pdf.table(table)
    pdf.heading("REFERENCE", chapter=True, following=18)
    for ref_type, title in REFERENCE_GROUPS.items():
        entries = report.references.get(ref_type, [])
        if entries:
            pdf.heading(title)
            for ref in entries:
                pdf.paragraph(format_reference(ref), size=pdf.table_size, keep=True)
    if not any(report.references.values()):
        pdf.paragraph("기록된 활용 출처 없음.")
    result = RenderedPDF(bytes(pdf.output()), pdf.page_no(), summary_bottom)
    logger.debug("PDF 조판 완료: pages=%d compact=%s", result.page_count, compact)
    return result


def write_pdf(report: ReportDocument, path: str | Path, *, compact: bool = False) -> Path:
    """이미 생성된 보고서의 단독 출력. 5페이지 초과 파일은 쓰지 않는다."""
    result = render_pdf(report, compact=compact)
    if result.page_count > MAX_PAGES:
        raise ReportTooLongError("보고서가 REFERENCE 포함 5페이지를 초과합니다.")
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("PDF 산출물 경로는 .pdf여야 합니다.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.content)
    return path
