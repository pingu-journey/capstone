"""Agentic RAG 서브그래프 세 경로 검증 (LLM·웹 검색·임베딩은 가짜, 오프라인)."""
import re
import uuid
from types import SimpleNamespace

import chromadb
import pytest

import rag.agentic_rag as ar


WEB_RESULTS = [
    {"title": "Amperon news", "url": "https://example.com/a", "content": "전력망 수요 예측 웹 근거", "published_date": None},
]


class FakeEmbedder:
    """재작성된 검색어("재작성" 포함)면 다른 방향 벡터를 돌려준다."""

    def __init__(self, rewrite_changes_vector: bool = True):
        self.rewrite_changes_vector = rewrite_changes_vector

    def embed_query(self, query: str) -> list[float]:
        if self.rewrite_changes_vector and "재작성" in query:
            return [0.0, 0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0, 0.0]


class FakeJudge:
    """청크 본문에 YES가 있으면 관련으로 판정한다. fail=True면 항상 예외."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def with_structured_output(self, model):
        return self

    def invoke(self, prompt: str):
        self.calls += 1
        if self.fail:
            raise ValueError("invalid structured output")
        chunks = re.findall(r"^\[\d+\] (.*)$", prompt.split("[청크]")[-1], re.M)
        return ar.GradeOutput(verdicts=["yes" if "YES" in c else "no" for c in chunks])


class FakeLLM:
    """rewrite(일반 호출)와 generate(구조화 출력)를 모두 흉내 낸다."""

    def __init__(self, answer: str):
        self.answer = answer

    def with_structured_output(self, model):
        return SimpleNamespace(invoke=lambda prompt: ar.GenerateOutput(answer=self.answer))

    def invoke(self, prompt: str):
        return SimpleNamespace(content="재작성된 검색어")


def make_collection(chunks: list[tuple[str, str, list[float]]]):
    collection = chromadb.EphemeralClient().create_collection(
        f"tech_docs_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[cid for cid, _, _ in chunks],
        documents=[text for _, text, _ in chunks],
        embeddings=[vec for _, _, vec in chunks],
        metadatas=[{"doc_id": cid.split(":")[0], "page": int(cid.split(":")[1])} for cid, _, _ in chunks],
    )
    return collection


def run_rag():
    return ar.run("질문", "Amperon", "Amperon", run_date="2026-09-30")


@pytest.fixture
def patch_rag(monkeypatch):
    def _patch(chunks, judge, answer="답변 [1]", rewrite_changes_vector=True):
        collection = make_collection(chunks)
        monkeypatch.setattr(ar, "TOP_K", 2)
        monkeypatch.setattr(ar, "_get_tech_collection", lambda: collection)
        monkeypatch.setattr(ar, "_get_embedder", lambda: FakeEmbedder(rewrite_changes_vector))
        monkeypatch.setattr(ar, "get_judge_llm", lambda: judge)
        monkeypatch.setattr(ar, "get_llm", lambda: FakeLLM(answer))
        monkeypatch.setattr(ar, "search", lambda query, max_results=5: WEB_RESULTS)
        monkeypatch.setattr(
            ar, "make_doc_ref",
            lambda doc_id, startup, used_by: {"id": doc_id, "startup": startup, "used_by": used_by},
        )

    return _patch


def test_relevant_enough_goes_to_generate(patch_rag):
    patch_rag(
        [
            ("docA:1:0", "기준선 YES", [1.0, 0.0, 0.0, 0.0]),
            ("docB:2:0", "TRL 근거 YES", [0.9, 0.1, 0.0, 0.0]),
            ("docC:3:0", "무관 NO", [0.0, 0.0, 1.0, 0.0]),
        ],
        FakeJudge(),
    )
    result = run_rag()
    assert result["path"] == "rag"
    assert result["rewrites"] == 0


def test_rewrite_accumulates_relevant_chunks(patch_rag):
    patch_rag(
        [
            ("docA:1:0", "기준선 YES", [1.0, 0.0, 0.0, 0.0]),
            ("docA:1:1", "무관 NO", [0.9, 0.1, 0.0, 0.0]),
            ("docB:5:0", "TRL 근거 YES", [0.0, 0.0, 1.0, 0.0]),
            ("docB:5:1", "무관 NO", [0.0, 0.0, 0.9, 0.1]),
        ],
        FakeJudge(),
        answer="답변 [2]",
    )
    result = run_rag()
    # 재작성 전 관련 1개 + 재작성 후 관련 1개 → 누적 2개로 generate
    assert result["path"] == "rag"
    assert result["rewrites"] == 1
    # 본문 [2] 인용만 Reference로 반환 (근거는 거리순이라 [2]는 docB)
    assert [s["id"] for s in result["sources"]] == ["docB"]


def test_web_fallback_after_max_rewrites(patch_rag):
    judge = FakeJudge()
    patch_rag(
        [
            ("docA:1:0", "기준선 YES", [1.0, 0.0, 0.0, 0.0]),
            ("docA:1:1", "무관 NO", [0.9, 0.1, 0.0, 0.0]),
            ("docC:3:0", "무관 NO", [0.0, 0.0, 1.0, 0.0]),
        ],
        judge,
        answer="답변 [1] [2]",
        rewrite_changes_vector=False,
    )
    result = run_rag()
    assert result["path"] == "rag+web"
    assert result["rewrites"] == 2
    assert judge.calls == 1  # 이미 판정한 청크는 다시 판정하지 않음
    assert len(result["sources"]) == 2


def test_grade_always_fails_goes_to_web(patch_rag):
    patch_rag(
        [
            ("docA:1:0", "기준선 YES", [1.0, 0.0, 0.0, 0.0]),
            ("docC:3:0", "무관 NO", [0.0, 0.0, 1.0, 0.0]),
        ],
        FakeJudge(fail=True),
    )
    result = run_rag()
    assert result["path"] == "web"
    assert result["rewrites"] == 2
    assert any("grade failed → treated as irrelevant" in line for line in result["log"])


def test_tech_docs_unavailable_goes_web_only(patch_rag, monkeypatch):
    patch_rag([("docA:1:0", "기준선 YES", [1.0, 0.0, 0.0, 0.0])], FakeJudge())

    def unavailable():
        raise ModuleNotFoundError("rag.ingest")

    monkeypatch.setattr(ar, "_get_tech_collection", unavailable)
    result = run_rag()
    assert result["path"] == "web"
    assert result["log"][0].endswith("tech_docs unavailable → web only")



ENTITY = {"name": "Amperon", "segment": "demand_forecasting", "homepage": "https://www.amperon.co"}
HOMEPAGE = {"title": "About us", "url": "https://www.amperon.co/about-us", "content": "Our team"}
SERIES_B = {"title": "Amperon Raises $20 Million Series B to Accelerate Energy Analytics",
            "url": "https://news.example.com/b", "content": "Amperon helps grid operators ..."}
CB_INSIGHTS = {"title": "Top Amperon Alternatives, Competitors", "url": "https://www.cbinsights.com/x",
               "content": "Amperon's top competitors include ..."}
INTERVIEW = {"title": "Amperon Co-Founder and CEO Sean Kelly - Watt It Takes", "url": "https://podcasts.example.com/y",
             "content": "Sean Kelly talks about founding Amperon"}
NAMESAKE = {"title": "Amperon Technologies - Simplifying manufacturing excellence",
            "url": "https://tn.linkedin.com/company/amperon-technologies",
            "content": "Amperon Technologies is on a mission to empower all the world's factories. "
                       "The startup's technology reads the power signature of a machine. ISO 9001 certified."}
INDUSTRY = {"title": "AI in grid operations", "url": "https://iea.example.org/x",
            "content": "AI improves renewables forecasting"}
MYST_AI = {"title": "Myst AI Closes $6M Series A To Forecast Energy Demand", "url": "https://news.example.com/m",
           "content": "Myst AI helps companies forecast energy consumption"}
OFF_TOPIC = {"title": "Top 10 SaaS startups", "url": "https://blog.example.com/y", "content": "marketing tools"}


def test_company_filter_with_terms():
    """기술 질문 기업 쿼리: 도메인 일치 또는 (기업명 + 전용 용어)."""
    kept, dropped = ar.filter_web_results([HOMEPAGE, SERIES_B, NAMESAKE, INDUSTRY], ENTITY)
    assert kept == [HOMEPAGE, SERIES_B]
    assert dropped == [NAMESAKE, INDUSTRY]  # 동명 제조업체, 기업명 없는 업계 자료


def test_company_filter_name_only():
    """팀·경쟁사 기업 쿼리: 도메인 일치 또는 기업명 포함 (동명 기업은 프롬프트로 차단)."""
    kept, dropped = ar.filter_web_results([CB_INSIGHTS, INTERVIEW, OFF_TOPIC], ENTITY, require_terms=False)
    assert kept == [CB_INSIGHTS, INTERVIEW]
    assert dropped == [OFF_TOPIC]


def test_industry_filter_broad_terms():
    """탐색 쿼리: 넓은 목록. 기업을 특정하지 않으므로 제조업체 페이지("power signature")도 통과한다."""
    kept, dropped = ar.filter_web_results([INDUSTRY, MYST_AI, NAMESAKE, OFF_TOPIC])
    assert kept == [INDUSTRY, MYST_AI, NAMESAKE]
    assert dropped == [OFF_TOPIC]


def test_industry_term_matching():
    assert not ar.has_industry_term("ISO 9001 certified")
    assert not ar.has_industry_term("der Energiemarkt in Deutschland")
    assert not ar.has_industry_term("Windows software to empower teams")
    assert ar.has_industry_term("DER aggregation")
    assert ar.has_industry_term("ESS를 운영하는 스타트업")
    assert ar.has_industry_term("renewables integration")
    assert not ar.has_industry_term("forecast energy consumption")  # 엄격 목록에는 없음
    assert ar.has_industry_term("forecast energy consumption", broad=True)
    assert not ar.has_industry_term("empower factories", broad=True)
