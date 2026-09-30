"""세 임베딩 모델의 PDF 검색 품질을 Hit Rate@5와 MRR로 비교한다."""

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

from config import ROOT
from rag.ingest import build_chunks, load_registry


TOP_K = 5
DATASETS = {
    "tech": ROOT / "data" / "eval" / "qa_tech.jsonl",
    "market": ROOT / "data" / "eval" / "qa_market.jsonl",
}
EXPECTED_QUESTIONS = {"tech": 20, "market": 30}
OUTPUT_PATH = ROOT / "outputs" / "embedding_eval.md"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    query_mode: str = "plain"
    document_prefix: str = ""
    max_seq_length: int | None = None
    trust_remote_code: bool = False


MODEL_SPECS = {
    "bge": ModelSpec("bge", "BAAI/bge-m3"),
    "qwen": ModelSpec(
        "qwen",
        "Qwen/Qwen3-Embedding-0.6B",
        query_mode="prompt_name",
        trust_remote_code=True,
    ),
    "e5": ModelSpec(
        "e5",
        "intfloat/multilingual-e5-large",
        query_mode="e5_prefix",
        document_prefix="passage: ",
        max_seq_length=512,
    ),
}


def load_questions(path: Path) -> list[dict]:
    """JSONL 평가셋을 읽고 필수 필드와 중복 질문을 검사한다."""
    questions = []
    seen = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: 잘못된 JSON") from error

        required = {"question", "doc_id", "page"}
        if set(item) != required:
            raise ValueError(
                f"{path}:{line_number}: 필드는 {sorted(required)}만 허용됩니다."
            )
        if not all(
            isinstance(item[field], str) and item[field].strip()
            for field in ("question", "doc_id")
        ):
            raise ValueError(
                f"{path}:{line_number}: question과 doc_id가 비어 있습니다."
            )
        if not isinstance(item["page"], int) or item["page"] < 1:
            raise ValueError(f"{path}:{line_number}: page는 양의 정수여야 합니다.")
        if item["question"] in seen:
            raise ValueError(f"{path}:{line_number}: 질문이 중복됐습니다.")
        seen.add(item["question"])
        questions.append(item)

    if not questions:
        raise ValueError(f"{path}: 평가 질문이 없습니다.")
    return questions


def prepare_evaluation():
    """인제스트와 동일한 청크를 만들고 정답 페이지가 존재하는지 확인한다."""
    chunks_by_group = build_chunks(load_registry())
    questions_by_group = {
        group: load_questions(path) for group, path in DATASETS.items()
    }

    for group, questions in questions_by_group.items():
        available = {
            (chunk["metadata"]["doc_id"], int(chunk["metadata"]["page"]))
            for chunk in chunks_by_group[group]
        }
        missing = sorted(
            {
                (item["doc_id"], item["page"])
                for item in questions
                if (item["doc_id"], item["page"]) not in available
            }
        )
        if missing:
            formatted = ", ".join(f"{doc_id} p.{page}" for doc_id, page in missing)
            raise ValueError(f"{group}: 청크에 없는 정답 페이지: {formatted}")

        actual = len(questions)
        expected = EXPECTED_QUESTIONS[group]
        if actual > expected:
            raise ValueError(f"{group}: 질문 수가 {actual}개로 명세({expected}개)를 초과합니다.")
        if actual < expected:
            print(
                f"[주의] {group} 평가셋: {actual}/{expected}개 "
                "(C 담당 기술 질문을 합치면 전체 평가 가능)",
                flush=True,
            )
        else:
            print(f"{group} 평가셋: {actual}/{expected}개", flush=True)

    return chunks_by_group, questions_by_group


class EvaluationEncoder:
    """모델별 문서·질의 입력 규칙을 적용하는 임베딩 래퍼."""

    def __init__(self, spec: ModelSpec, batch_size: int):
        from sentence_transformers import SentenceTransformer

        print(f"\n[{spec.name}] 모델 로딩...", flush=True)
        self.spec = spec
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            spec.name,
            cache_folder=str(ROOT / "outputs" / "cache" / "models"),
            trust_remote_code=spec.trust_remote_code,
        )
        if spec.max_seq_length is not None:
            self.model.max_seq_length = spec.max_seq_length

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [self.spec.document_prefix + text for text in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        kwargs = {}
        if self.spec.query_mode == "prompt_name":
            kwargs["prompt_name"] = "query"
        elif self.spec.query_mode == "e5_prefix":
            text = "query: " + text

        vector = self.model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **kwargs,
        )
        return vector[0].tolist()


def _create_collection(client, name: str, chunks: list[dict], encoder, batch_size: int):
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk["text"] for chunk in batch]
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=texts,
            metadatas=[chunk["metadata"] for chunk in batch],
            embeddings=encoder.embed_documents(texts),
        )
    return collection


def calculate_metrics(collection, questions: list[dict], encoder) -> dict:
    """정답 (doc_id, page)의 Top-5 적중률과 첫 적중 역순위를 계산한다."""
    hits = 0
    reciprocal_rank_sum = 0.0
    details = []

    for index, item in enumerate(questions, start=1):
        result = collection.query(
            query_embeddings=[encoder.embed_query(item["question"])],
            n_results=min(TOP_K, collection.count()),
            include=["metadatas"],
        )
        metadatas = result.get("metadatas", [[]])[0]
        first_rank = next(
            (
                rank
                for rank, metadata in enumerate(metadatas, start=1)
                if metadata.get("doc_id") == item["doc_id"]
                and int(metadata.get("page", -1)) == item["page"]
            ),
            None,
        )
        if first_rank is not None:
            hits += 1
            reciprocal_rank_sum += 1.0 / first_rank
        details.append({**item, "rank": first_rank})
        print(
            f"  {index:02d}/{len(questions)} "
            f"{'적중' if first_rank else '실패'}"
            + (f" @{first_rank}" if first_rank else ""),
            flush=True,
        )

    count = len(questions)
    return {
        "count": count,
        "hit_rate_at_5": hits / count,
        "mrr": reciprocal_rank_sum / count,
        "details": details,
    }


def _release_accelerator_cache():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        pass


def render_report(results: dict, question_counts: dict) -> str:
    lines = [
        "# 임베딩 모델 검색 평가",
        "",
        f"동일 청크와 동일 질의로 Top-{TOP_K} 검색을 비교했다. "
        "정답은 `(doc_id, page)`가 일치할 때 적중으로 계산했다.",
        "",
        f"- 기술 질문: {question_counts['tech']}개 "
        f"(최종 명세 {EXPECTED_QUESTIONS['tech']}개)",
        f"- 시장 질문: {question_counts['market']}개 "
        f"(최종 명세 {EXPECTED_QUESTIONS['market']}개)",
        "",
        "| 모델 | 기술 Hit Rate@5 | 기술 MRR | 시장 Hit Rate@5 | 시장 MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for spec in MODEL_SPECS.values():
        if spec.key not in results:
            continue
        tech = results[spec.key]["tech"]
        market = results[spec.key]["market"]
        lines.append(
            f"| `{spec.name}` | {tech['hit_rate_at_5']:.4f} | "
            f"{tech['mrr']:.4f} | {market['hit_rate_at_5']:.4f} | "
            f"{market['mrr']:.4f} |"
        )

    if question_counts["tech"] < EXPECTED_QUESTIONS["tech"]:
        lines.extend(
            [
                "",
                "> 기술 평가는 B 담당 10개 질문으로 산출한 중간 결과다. "
                "C 담당 10개를 `qa_tech.jsonl`에 추가한 뒤 최종 실행한다.",
            ]
        )
    return "\n".join(lines) + "\n"


def run(model_keys: list[str], batch_size: int, validate_only: bool = False):
    chunks_by_group, questions_by_group = prepare_evaluation()
    if validate_only:
        print("평가셋·청크 검사 완료: 모델 로딩 없음", flush=True)
        return {}

    import chromadb

    results = {}
    for key in model_keys:
        spec = MODEL_SPECS[key]
        encoder = EvaluationEncoder(spec, batch_size)
        client = chromadb.EphemeralClient()
        results[key] = {}
        for group in DATASETS:
            print(f"[{spec.name}] {group} 인덱싱...", flush=True)
            collection = _create_collection(
                client,
                f"{group}_{key}_eval",
                chunks_by_group[group],
                encoder,
                batch_size,
            )
            print(f"[{spec.name}] {group} 평가...", flush=True)
            results[key][group] = calculate_metrics(
                collection,
                questions_by_group[group],
                encoder,
            )
            del collection
        del encoder, client
        _release_accelerator_cache()

    report = render_report(
        results,
        {group: len(items) for group, items in questions_by_group.items()},
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"\n{report}", end="")
    print(f"저장: {OUTPUT_PATH}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help="평가할 모델 키. 기본값: bge qwen e5",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="모델을 받지 않고 평가셋과 정답 페이지만 검사",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")
    run(args.models, args.batch_size, args.validate_only)
