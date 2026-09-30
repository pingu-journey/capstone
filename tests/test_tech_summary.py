"""기술 요약 에이전트 run() 오프라인 테스트 (서브그래프·LLM·웹 검색은 가짜)."""
import json
from pathlib import Path
from types import SimpleNamespace

import agents.tech_summary as ts
from schemas import Reference, TeamInfo, TechSummary
from tools.references import make_web_ref


STATE = json.loads((Path(__file__).parent / "fixtures" / "sample_state.json").read_text(encoding="utf-8"))
TEAM_RESULTS = [
    {"title": "Amperon CEO Sean Kelly", "url": "https://news.example.com/ceo", "content": "Amperon founder"},
    {"title": "Amperon team", "url": "https://news.example.com/team", "content": "Amperon CTO"},
]


def fake_rag(question, company, startup, run_date=None, web_queries=None, entity=None):
    kind = "company" if entity else "industry"  # 질문 1만 entity를 받음
    ref = make_web_ref({"title": question, "url": f"https://rag.example.com/{kind}"},
                       startup, "tech_summary", run_date)
    return {"answer": "근거 답변", "sources": [ref], "path": "web", "rewrites": 0,
            "log": ["[tech_summary/agentic_rag] fake"]}


class FakeLLM:
    def __init__(self, draft):
        self.draft = draft

    def with_structured_output(self, model):
        return SimpleNamespace(invoke=self.invoke)

    def invoke(self, prompt):
        if self.draft is None:
            raise ValueError("invalid structured output")
        return self.draft


def patch(monkeypatch, draft):
    monkeypatch.setattr(ts.agentic_rag, "run", fake_rag)
    monkeypatch.setattr(ts, "search", lambda query, max_results=5: TEAM_RESULTS)
    monkeypatch.setattr(ts, "get_llm", lambda: FakeLLM(draft))


def test_run_returns_valid_tech_summary(monkeypatch):
    patch(monkeypatch, ts.TechSummaryDraft(
        core_tech="AI 수요 예측", products=[], trl=8, trl_rationale="상용 운영",
        performance=["예측 정확도 98%"], strengths=["확장성"], weaknesses=[],
        scalability="소프트웨어 중심",
        team=TeamInfo(founders=["Sean Kelly — CEO"], assessment="도메인 경험"),
        insufficient=["팀 정보"],  # LLM 자기보고는 무시된다
        used_team_sources=[1],
    ))
    out = ts.run(STATE)

    assert set(out) == {"tech_summary", "references", "log"}
    summary = TechSummary(**out["tech_summary"])
    assert summary.insufficient == ["제품"]  # 빈 필드 기준으로 코드가 계산
    refs = [Reference(**r) for r in out["references"]]
    assert all(r.startup == "Amperon" and r.used_by == "tech_summary" for r in refs)
    assert len(refs) == 3  # 서브그래프 2건 + 사용한 팀 자료 1건
    assert out["log"][:2] == ["[tech_summary/agentic_rag] fake"] * 2
    assert out["log"][-1].startswith("[tech_summary] Amperon TRL 8")


def test_run_falls_back_when_llm_fails(monkeypatch):
    patch(monkeypatch, None)
    summary = TechSummary(**ts.run(STATE)["tech_summary"])
    assert summary.trl == 1 and summary.trl_rationale == ts.INSUFFICIENT
    assert set(summary.insufficient) == set(ts.INSUFFICIENT_TO_RUBRIC)
