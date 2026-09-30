"""경쟁사 비교의 근거 번호 검증 함수 단위 테스트."""
from agents.competitor import INSUFFICIENT, split_citations, verify_items, verify_status


def test_split_citations_strips_numbers_and_drops_out_of_range():
    assert split_citations("유틸리티 A사와 상용 계약 [2][9]", 5) == ("유틸리티 A사와 상용 계약", [2])
    assert split_citations("특허 [1] 보유", 3) == ("특허 보유", [1])


def test_verify_items_keeps_only_cited_items():
    items = ["상용 계약 [1]", "근거 없는 주장", "범위 밖 [7]", f"{INSUFFICIENT} [2]", "데이터 [2][3]"]
    kept, used = verify_items(items, 3)
    assert kept == ["상용 계약", "데이터"]
    assert used == {1, 2, 3}


def test_verify_items_empty_when_nothing_cited():
    assert verify_items(["근거 없는 주장"], 3) == ([], set())


def test_verify_status():
    assert verify_status("Series A", 2, 3) == ("Series A", 2)
    assert verify_status("상장", None, 3) == (INSUFFICIENT, None)
    assert verify_status("상장", 5, 3) == (INSUFFICIENT, None)


# --- run() 오프라인 테스트 (LLM·웹 검색은 가짜) ---
import json
from pathlib import Path
from types import SimpleNamespace

import agents.competitor as comp
from schemas import CompetitorAnalysis, Reference


STATE = json.loads((Path(__file__).parent / "fixtures" / "sample_state.json").read_text(encoding="utf-8"))
SEARCH_RESULTS = {  # 쿼리 순서대로 [1]~[4]
    "competitors": {"title": "Top Amperon Alternatives", "url": "https://cb.example.com/a", "content": "Amperon competitors"},
    "startups": {"title": "AI grid startups", "url": "https://news.example.com/s", "content": "grid forecasting startups"},
    "기업": {"title": "국내 전력 수요 예측 기업", "url": "https://kr.example.com/k", "content": "전력 수요 예측"},
    "customers": {"title": "Amperon signs utility contract", "url": "https://news.example.com/c", "content": "Amperon contract"},
}


def fake_search(query, max_results=5):
    return [next(r for key, r in SEARCH_RESULTS.items() if key in query)]


def test_run_returns_valid_competitor_analysis(monkeypatch):
    draft = comp.CompetitorDraft(
        competitors=[
            comp.CompetitorDraftItem(name="GridCo", country="US", status="Series B", status_source=2,
                                     offering="예측", vs_target="차이"),
            comp.CompetitorDraftItem(name="PowerCo", country="US", status="상장", status_source=None,
                                     offering="분석", vs_target="차이"),
            comp.CompetitorDraftItem(name="Amperon", country="US", status="Series B", status_source=1,
                                     offering="자기 자신", vs_target="-"),
            comp.CompetitorDraftItem(name="StartCo", country="US", status="Series A", status_source=9,
                                     offering="플랫폼", vs_target="차이"),
        ],
        differentiation="차별성 문장",
        moat=["근거 없는 진입장벽"],
        traction=["유틸리티 상용 계약 [4]"],
        competition_risks=[],
    )
    llm = SimpleNamespace(with_structured_output=lambda model: SimpleNamespace(invoke=lambda prompt: draft))
    monkeypatch.setattr(comp, "search", fake_search)
    monkeypatch.setattr(comp, "get_llm", lambda: llm)
    out = comp.run(STATE)

    assert set(out) == {"competitor_analysis", "references", "log"}
    analysis = out["competitor_analysis"]
    CompetitorAnalysis(**analysis)
    assert [c["name"] for c in analysis["competitors"]] == ["GridCo", "PowerCo", "StartCo"]  # 자기 자신 제외
    assert [c["status"] for c in analysis["competitors"]] == ["Series B", INSUFFICIENT, INSUFFICIENT]
    assert analysis["traction"] == ["유틸리티 상용 계약"]
    assert analysis["moat"] == []
    assert analysis["insufficient"] == ["진입장벽"]
    refs = [Reference(**r) for r in out["references"]]
    assert [r.url for r in refs] == [SEARCH_RESULTS["startups"]["url"], SEARCH_RESULTS["customers"]["url"]]
    assert all(r.startup == "Amperon" and r.used_by == "competitor" for r in refs)


def test_verify_items_accepts_tech_citation_only_when_allowed():
    items = ["자체 수요 데이터 [T]", "근거 없는 주장", "상용 계약 [2]"]
    assert verify_items(items, 3, allow_tech=True) == (["자체 수요 데이터", "상용 계약"], {2})
    assert verify_items(items, 3) == (["상용 계약"], {2})  # competition_risks 등은 [T] 불인정
