"""기술·시장 문서에서 공통으로 사용하는 임베딩 인터페이스"""

import os
from functools import lru_cache

import numpy as np

from config import ROOT  # config가 프로젝트의 .env를 로드


MODEL_NAME = "BAAI/bge-m3"


def _normalize(vectors, expected_count: int) -> list[list[float]]:
    """문장 벡터 형태를 확인하고 길이를 1로 정규화합니다."""
    array = np.asarray(vectors, dtype=np.float32)

    if array.ndim == 1 and expected_count == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2 or array.shape[0] != expected_count:
        raise ValueError(
            f"임베딩 응답 형태 오류: {array.shape}, "
            f"예상 문장 수: {expected_count}"
        )

    if array.shape[1] != 1024:
        raise ValueError(f"bge-m3 벡터 차원 오류: {array.shape[1]}")

    if not np.isfinite(array).all():
        raise ValueError("임베딩에 NaN 또는 무한대가 포함돼 있습니다.")

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("길이가 0인 임베딩이 반환됐습니다.")

    return (array / norms).tolist()


class Embedder:
    def __init__(self, backend: str):
        self.backend = backend
        self._model = None
        self._client = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("임베딩 입력은 비어 있지 않은 문자열이어야 합니다.")

        if self.backend == "local":
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    MODEL_NAME,
                    cache_folder=str(ROOT / "outputs" / "cache" / "models"),
                )

            vectors = self._model.encode(
                texts,
                batch_size=8,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return _normalize(vectors, len(texts))

        if self._client is None:
            from huggingface_hub import InferenceClient

            token = os.getenv("HF_TOKEN")
            if not token:
                raise ValueError("hf_api 사용 시 .env에 HF_TOKEN이 필요합니다.")

            self._client = InferenceClient(
                model=os.getenv("HF_EMBEDDING_ENDPOINT") or MODEL_NAME,
                token=token,
                timeout=120,
            )

        # API 제공 환경에 따라 모델 지원 여부가 다를 수 있음
        result = []
        for text in texts:
            vectors = self._client.feature_extraction(text)
            result.extend(_normalize(vectors, 1))
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=2)
def _get_embedder(backend: str) -> Embedder:
    return Embedder(backend)


def get_embedder() -> Embedder:
    backend = os.getenv("EMBEDDING_BACKEND", "local").strip().lower()
    if backend not in {"local", "hf_api"}:
        raise ValueError(
            "EMBEDDING_BACKEND는 local 또는 hf_api여야 합니다."
        )
    return _get_embedder(backend)