from typing import Literal

from pydantic import BaseModel, Field


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
