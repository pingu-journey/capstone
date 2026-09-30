"""시장성 평가용 Advanced RAG 검색 파이프라인."""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from rag.embeddings import get_embedder
from rag.ingest import SEGMENTS, get_collection


COLLECTION_NAME = "market_docs"
DENSE_TOP_N = 20
BM25_TOP_N = 20
RRF_TOP_N = 20
RRF_K = 60
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _get_kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    print("[advanced_rag] reranker 로딩...", flush=True)
    return CrossEncoder(RERANKER_MODEL)


def _build_where(country: str, segment: str, use_segment: bool = True) -> dict:
    countries = list(dict.fromkeys([country, "GLOBAL"]))
    country_filter = {"country": {"$in": countries}}
    if not use_segment:
        return country_filter
    return {"$and": [country_filter, {f"seg_{segment}": True}]}


def _tokenize(text: str) -> list[str]:
    """한국어 명사·용언과 영문·숫자를 BM25 토큰으로 만든다."""
    if not text:
        return []
    tokens = [
        token.form.lower()
        for token in _get_kiwi().tokenize(text)
        if token.tag.startswith(("N", "V"))
    ]
    tokens.extend(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    return tokens


@lru_cache(maxsize=32)
def _load_filtered_bm25(country: str, segment: str, use_segment: bool):
    """필터별 후보 청크와 BM25 인덱스를 프로세스 메모리에 캐시한다."""
    where = _build_where(country, segment, use_segment)
    result = get_collection(COLLECTION_NAME).get(
        where=where,
        include=["documents", "metadatas"],
    )
    chunks = [
        {"id": chunk_id, "text": text, "metadata": metadata or {}}
        for chunk_id, text, metadata in zip(
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        )
    ]
    if not chunks:
        return [], None, where
    corpus = [_tokenize(chunk["text"]) for chunk in chunks]
    return chunks, BM25Okapi(corpus), where


def _get_search_context(country: str, segment: str):
    context = _load_filtered_bm25(country, segment, True)
    if len(context[0]) >= 5:
        return context
    return _load_filtered_bm25(country, segment, False)


def _dense_search(query: str, where: dict, candidate_count: int) -> list[dict]:
    if candidate_count == 0:
        return []
    result = get_collection(COLLECTION_NAME).query(
        query_embeddings=[get_embedder().embed_query(query)],
        where=where,
        n_results=min(DENSE_TOP_N, candidate_count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": chunk_id,
            "text": text,
            "metadata": metadata or {},
            "rank": rank,
            "distance": float(distance),
        }
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(
                result.get("ids", [[]])[0],
                result.get("documents", [[]])[0],
                result.get("metadatas", [[]])[0],
                result.get("distances", [[]])[0],
            ),
            start=1,
        )
    ]


def _bm25_search(query: str, chunks: list[dict], bm25) -> list[dict]:
    if not chunks or bm25 is None:
        return []
    ranked = [
        {**chunk, "bm25_score": float(score)}
        for chunk, score in zip(chunks, bm25.get_scores(_tokenize(query)))
    ]
    ranked.sort(key=lambda item: item["bm25_score"], reverse=True)
    ranked = ranked[: min(BM25_TOP_N, len(ranked))]
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def _rrf_merge(dense_results: list[dict], bm25_results: list[dict]) -> list[dict]:
    merged = {}
    for results in (dense_results, bm25_results):
        for item in results:
            current = merged.setdefault(
                item["id"],
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0,
                },
            )
            current["rrf_score"] += 1.0 / (RRF_K + item["rank"])
    return sorted(
        merged.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )[:RRF_TOP_N]


def _rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    if not results:
        return []
    scores = _get_reranker().predict(
        [[query, item["text"]] for item in results],
        show_progress_bar=False,
    )
    reranked = [
        {**item, "rerank_score": float(score)}
        for item, score in zip(results, scores)
    ]
    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。])\s+|\n+", text or "")
        if sentence.strip()
    ]


def _contains_numeric_evidence(sentence: str) -> bool:
    return bool(
        re.search(
            r"\d+(?:\.\d+)?\s*(?:%|GW|MW|TWh|USD|KRW|달러|원)"
            r"|\$\s*\d+|(?:19|20)\d{2}",
            sentence,
            flags=re.IGNORECASE,
        )
    )


def _compress_context(query: str, text: str, max_chars: int = 1200) -> str:
    query_tokens = {token for token in _tokenize(query) if len(token) >= 2}
    selected = []
    for sentence in _split_sentences(text):
        sentence_tokens = {
            token for token in _tokenize(sentence) if len(token) >= 2
        }
        if query_tokens & sentence_tokens or _contains_numeric_evidence(sentence):
            selected.append(sentence)
    return (" ".join(selected) if selected else text)[:max_chars]


def search(
    query: str,
    country: str,
    segment: str,
    k: int = 5,
) -> list[dict]:
    """필터·하이브리드 검색·RRF·리랭킹·압축 결과를 반환한다."""
    if not query.strip():
        raise ValueError("query는 비어 있지 않아야 합니다.")
    country = country.strip().upper()
    if not country:
        raise ValueError("country는 비어 있지 않아야 합니다.")
    if segment not in SEGMENTS:
        raise ValueError(f"지원하지 않는 segment: {segment}")
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")

    chunks, bm25, where = _get_search_context(country, segment)
    if not chunks:
        return []
    dense = _dense_search(query, where, len(chunks))
    sparse = _bm25_search(query, chunks, bm25)
    merged = _rrf_merge(dense, sparse)
    reranked = _rerank(query, merged, max(k, 5))

    output = []
    for item in reranked[:k]:
        metadata = item["metadata"]
        output.append(
            {
                "text": _compress_context(query, item["text"]),
                "doc_id": metadata.get("doc_id", ""),
                "page": metadata.get("page"),
                "score": item["rerank_score"],
            }
        )
    return output


if __name__ == "__main__":
    for index, result in enumerate(
        search("미국 VPP 시장 전망", "US", "generation_vpp"),
        start=1,
    ):
        print(
            f"[{index}] {result['doc_id']} p.{result['page']} "
            f"score={result['score']:.4f}\n{result['text'][:500]}\n"
        )
