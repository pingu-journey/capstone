from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Segment = Literal[
    "demand_forecasting", "generation_vpp", "ess_operation", "grid_management"
]
SEGMENT_KO = {
    "demand_forecasting": "전력 수요 예측",
    "generation_vpp": "발전 예측 · VPP",
    "ess_operation": "배터리(ESS) 운영",
    "grid_management": "전력망 관리",
}


class Candidate(BaseModel):
    name: str
    country: str
    stage: str
    funding: str
    segment: Segment
    homepage: str | None = None
    priority: float = 0.0
    evidence_urls: list[str] = []


class TeamInfo(BaseModel):
    founders: list[str]
    key_members: list[str] = []
    assessment: str


class TechSummary(BaseModel):
    core_tech: str
    products: list[str]
    trl: int = Field(ge=1, le=9)
    trl_rationale: str
    performance: list[str]
    strengths: list[str]
    weaknesses: list[str]
    scalability: str
    team: TeamInfo
    insufficient: list[str] = []


class MarketFigure(BaseModel):
    metric: str
    value: str
    year: str
    source_id: str


class MarketAnalysis(BaseModel):
    segment: Segment
    country: str
    market_size: list[MarketFigure]
    growth: str
    demand_drivers: list[str]
    customers: list[str]
    policy_risks: list[str]
    summary: str
    sources: list[dict] = []


class Competitor(BaseModel):
    name: str
    country: str
    status: str
    offering: str
    vs_target: str


class CompetitorAnalysis(BaseModel):
    competitors: list[Competitor]
    differentiation: str
    moat: list[str]
    traction: list[str]
    competition_risks: list[str]


class ItemScore(BaseModel):
    item_id: str
    score: int = Field(ge=0, le=10)
    weight: int
    rationale: str
    insufficient_info: bool = False


class EvalRecord(BaseModel):
    startup: str
    segment: Segment
    total_score: float
    decision: Literal["투자", "보류"]
    knockout: str | None = None
    weakest_items: list[str]
    key_reason: str


class Reference(BaseModel):
    id: str
    startup: str
    ref_type: Literal["report", "paper", "web"]
    author_org: str
    date: str
    title: str
    venue: str | None = None
    volume_issue: str | None = None
    pages: str | None = None
    url: str | None = None
    used_by: str


TaskKind = Literal["company_tech", "industry", "team", "market_size", "growth", "demand", "regulation"]


class OrchestrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, str_strip_whitespace=True)


class TaskSpec(OrchestrationModel):
    task_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    kind: TaskKind
    query: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    target_fields: list[str] = Field(min_length=1)


class TaskResult(OrchestrationModel):
    task_id: str
    status: Literal["success", "insufficient", "failed"]
    findings: str
    source_ids: list[str]
    evidence_ref: str | None
    attempts: int = Field(strict=True, ge=1, le=2)
    error_code: str | None = None


class QualityCheck(OrchestrationModel):
    passed: bool = Field(strict=True)
    reason: str = Field(min_length=1)
    feedback: list[str]


class ReportIssue(OrchestrationModel):
    chapter: int = Field(strict=True, ge=0, le=4, description="0=SUMMARY, 1~4=본문 장 번호")
    block: int = Field(strict=True, ge=1, description="장 안의 1-based 블록 위치, SUMMARY는 1")
    kind: Literal["numeric", "citation", "length", "groundedness", "neutrality", "bias", "structure"]
    text: str = Field(default="", description="코드가 지정된 위치의 실제 블록 원문으로 채운다.")
    source_ids: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class QualityAssessment(OrchestrationModel):
    report_version: int = Field(strict=True, ge=1, le=3)
    groundedness: QualityCheck
    neutrality: QualityCheck
    bias: QualityCheck
    structure_passed: bool = Field(strict=True)
    structure_reasons: list[str]
    recoverable: bool = Field(strict=True)


class QualityVerdict(QualityAssessment):
    issues: list[ReportIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.structure_passed and all(
            check.passed for check in (self.groundedness, self.neutrality, self.bias)
        )


class ReportIssueTargets(OrchestrationModel):
    issues: list[ReportIssue] = Field(min_length=1)
