"""E2E에서 발견한 병렬 실행·인터페이스 문제의 회귀 테스트 (모델·LLM은 가짜, 오프라인)."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import sentence_transformers

import agents.market_eval as market_eval
from rag.embeddings import Embedder
from schemas import MarketAnalysis, MarketFigure
from tools.references import make_doc_ref


class FakeSentenceTransformer:
    """동시에 encode가 실행되면 기록한다 (실제 torch는 첫 인코딩 동시 실행 시 segfault)."""

    created = 0
    active = 0
    max_active = 0
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        FakeSentenceTransformer.created += 1

    def encode(self, texts, **kwargs):
        with FakeSentenceTransformer.lock:
            FakeSentenceTransformer.active += 1
            FakeSentenceTransformer.max_active = max(FakeSentenceTransformer.max_active,
                                                     FakeSentenceTransformer.active)
        time.sleep(0.05)
        with FakeSentenceTransformer.lock:
            FakeSentenceTransformer.active -= 1
        return np.ones((len(texts), 1024), dtype=np.float32)


def test_embedder_serializes_model_load_and_encode(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    embedder = Embedder("local")
    barrier = threading.Barrier(4)

    def work():
        barrier.wait()
        embedder.embed_query("전력 수요 예측")

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert FakeSentenceTransformer.created == 1
    assert FakeSentenceTransformer.max_active == 1


def test_market_figure_source_id_is_reference_id(monkeypatch):
    doc_id = "iea_energy_ai_2025"
    analysis = MarketAnalysis(
        segment="demand_forecasting", country="US",
        market_size=[MarketFigure(metric="시장 규모", value="10GW", year="2030", source_id=doc_id)],
        growth="성장", demand_drivers=[], customers=[], policy_risks=[], summary="요약",
    )
    llm = SimpleNamespace(with_structured_output=lambda *args, **kwargs: SimpleNamespace(invoke=lambda prompt: analysis))
    monkeypatch.setattr(market_eval, "_collect_context", lambda country, segment: ("근거", [doc_id]))
    monkeypatch.setattr(market_eval, "get_llm", lambda: llm)
    state = {
        "current_startup": {"name": "Amperon", "country": "US", "stage": "Series B", "funding": "-",
                            "segment": "demand_forecasting"},
        "market_cache": {},
    }

    out = market_eval.run(state)

    ref_id = make_doc_ref(doc_id, startup="Amperon", used_by="market_eval")["id"]
    assert out["market_analysis"]["market_size"][0]["source_id"] == ref_id
    assert ref_id in {ref["id"] for ref in out["references"]}
