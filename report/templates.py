"""보고서 경로별 목차·입력·예산과 PDF/Markdown 공용 본문 모델."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Route = Literal["investment", "hold"]
SUMMARY_BUDGET = 550


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, hide_input_in_errors=True)


class Block(ReportModel):
    subtitle: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ChapterDraft(ReportModel):
    blocks: list[Block] = Field(min_length=1)


class InvestmentSummary(ReportModel):
    conclusion: str = Field(min_length=1)
    investment_points: list[str] = Field(min_length=3, max_length=3)
    risks: list[str] = Field(min_length=2, max_length=2)


class HoldSummary(ReportModel):
    conclusion: str = Field(min_length=1)
    candidate_count: int = Field(strict=True, ge=0)
    common_reasons: list[str] = Field(min_length=1)


class Table(ReportModel):
    title: str
    columns: list[str]
    rows: list[list[str]]


class Chapter(ChapterDraft):
    number: int
    title: str
    char_budget: int
    tables: list[Table] = Field(default_factory=list)


class ReportDocument(ReportModel):
    route: Route
    title: str
    run_date: str
    decision: Literal["투자", "보류"]
    total_score: float | None
    summary: str
    chapters: list[Chapter]
    references: dict[str, list[dict]]


@dataclass(frozen=True)
class ChapterTemplate:
    title: str
    inputs: tuple[str, ...]
    char_budget: int
    guidance: str


TEMPLATES = {
    "investment": (
        ChapterTemplate("기업 및 사업 개요", ("current_startup", "tech_summary"), 900,
                        "사업 아이디어, 핵심 제품·기술과 TRL, 사업모델·투자 단계를 설명한다."),
        ChapterTemplate("시장 및 경쟁 분석", ("market_analysis", "competitor_analysis"), 1100,
                        "시장 규모·성장성, 수요·정책, 경쟁 차별성을 설명한다. 비교표를 반복하지 않는다."),
        ChapterTemplate("기업 역량 및 리스크", ("team", "traction", "weaknesses", "policy_risks", "competition_risks"), 1100,
                        "창업자·팀, 사업 성과, 기술·시장·규제·경쟁 리스크를 설명한다."),
        ChapterTemplate("종합 투자 평가", ("scores", "total_score", "decision", "insufficient_info"), 700,
                        "확정된 점수·판정과 근거를 설명한다. 분석의 한계는 코드가 별도 블록으로 추가한다."),
    ),
    "hold": (
        ChapterTemplate("평가 대상 및 선정 과정", ("domain", "requirements", "evaluated", "search_round"), 900,
                        "탐색 조건과 평가 후보 목록을 설명한다. 평가하지 않은 대기 후보를 포함하지 않는다."),
        ChapterTemplate("후보별 평가 결과 비교", ("evaluation_history",), 1100,
                        "누적 이력의 총점·판정·취약 항목을 비교한다. 이전 후보 상세 정보를 추정하지 않는다."),
        ChapterTemplate("주요 보류 사유 및 시장 환경", ("hold_reasons", "market_cache"), 1100,
                        "공통 취약 항목과 후보별 보류 사유, 캐시에 남은 시장·규제 정보를 구분한다."),
        ChapterTemplate("분석의 한계", ("limitations", "run_date"), 700,
                        "공개 정보의 한계와 추가 확인 사항을 설명한다. 원문 한계 목록은 코드가 추가한다."),
    ),
}


def chapter_budget(template: ChapterTemplate, reduction: int = 0) -> int:
    """다음 PDF 단계의 최대 두 번, 회당 20% 축소를 위한 동일 예산 함수."""
    if type(reduction) is not int or reduction not in (0, 1, 2):
        raise ValueError("본문 예산 축소 횟수는 0~2여야 합니다.")
    return template.char_budget * 4**reduction // 5**reduction


def blocks_length(blocks: list[Block]) -> int:
    # 소제목과 본문 사이 및 블록 사이 줄바꿈도 공백 포함 글자 예산에 센다.
    return len("\n\n".join(f"{block.subtitle}\n{block.text}" for block in blocks))
