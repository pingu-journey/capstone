"""PDF와 공유할 ReportDocument를 검토용 Markdown으로 출력한다."""

import html
import logging
import re
from pathlib import Path

from report.templates import ReportDocument
from tools.references import REFERENCE_GROUPS, format_reference

logger = logging.getLogger(__name__)


def _escape(value: str) -> str:
    text = html.escape(value, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text)


def to_markdown(report: ReportDocument) -> str:
    lines = [f"# {_escape(report.title)}", "", f"작성일: {report.run_date}", "", "## SUMMARY", "",
             _escape(report.summary), ""]
    for chapter in report.chapters:
        lines.extend([f"## {chapter.number}. {_escape(chapter.title)}", ""])
        for block in chapter.blocks:
            lines.extend([f"### {_escape(block.subtitle)}", "", _escape(block.text), ""])
        for table in chapter.tables:
            lines.extend([f"### {_escape(table.title)}", ""])
            if not table.rows:
                lines.extend(["공개 정보 부족: 표시할 데이터가 없습니다.", ""])
                continue
            lines.append("| " + " | ".join(_escape(value) for value in table.columns) + " |")
            lines.append("| " + " | ".join("---" for _ in table.columns) + " |")
            for row in table.rows:
                lines.append("| " + " | ".join(_escape(value).replace("\n", "<br>") for value in row) + " |")
            lines.append("")
    lines.extend(["## REFERENCE", ""])
    for ref_type, title in REFERENCE_GROUPS.items():
        entries = report.references.get(ref_type, [])
        if entries:
            lines.extend([f"### {title}", ""])
            lines.extend(f"- {_escape(format_reference(ref))}" for ref in entries)
            lines.append("")
    if not any(report.references.values()):
        lines.extend(["기록된 활용 출처 없음.", ""])
    return "\n".join(lines)


def write_markdown(report: ReportDocument, path: str | Path) -> Path:
    """본문 단계에서는 명시적 .md 경로에 저장하며 PDF report_path로 반환하지 않는다."""
    path = Path(path)
    if path.suffix.lower() != ".md":
        raise ValueError("Markdown 산출물 경로는 .md여야 합니다.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(report), encoding="utf-8")
    logger.info("보고서 Markdown 저장 완료: report_path=%s", path)
    return path
