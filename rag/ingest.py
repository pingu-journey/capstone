"""PDF 발췌본을 청킹하고 임베딩해 Chroma에 저장한다."""

import argparse
import hashlib
import re
from collections import Counter
from functools import lru_cache

import yaml
from pypdf import PdfReader

from config import DATA_DIR, ROOT
from rag.embeddings import MODEL_NAME, get_embedder


REGISTRY_PATH = ROOT / "config" / "docs.yaml"
COLLECTIONS = {"tech": "tech_docs", "market": "market_docs"}
SEGMENTS = (
    "demand_forecasting",
    "generation_vpp",
    "ess_operation",
    "grid_management",
)
INGEST_VERSION = "page-chunks-v1"


def parse_pages(value: str) -> list[int]:
    """'1-3,8-9'를 [1, 2, 3, 8, 9]로 변환한다."""
    result = []
    for part in str(value).split(","):
        part = part.strip()
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise ValueError(f"잘못된 페이지 범위: {value}")
        bounds = [int(number) for number in part.split("-")]
        start, end = bounds if len(bounds) == 2 else (bounds[0], bounds[0])
        if start < 1 or end < start:
            raise ValueError(f"잘못된 페이지 범위: {value}")
        result.extend(range(start, end + 1))
    if len(result) != len(set(result)):
        raise ValueError(f"중복 페이지: {value}")
    return result


@lru_cache(maxsize=1)
def _client():
    import chromadb

    return chromadb.PersistentClient(path=str(DATA_DIR / "vectorstore"))


def get_collection(name: str):
    """기술·시장 컬렉션을 반환한다. 인제스트 전이면 빈 컬렉션을 만든다."""
    if name not in COLLECTIONS.values():
        raise ValueError(f"지원하지 않는 컬렉션: {name}")
    return _client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def load_registry():
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    prepared = []
    seen_ids = set()
    total = 0

    for group in COLLECTIONS:
        for doc in registry[group]:
            if doc["id"] in seen_ids:
                raise ValueError(f"중복 doc_id: {doc['id']}")
            seen_ids.add(doc["id"])

            path = ROOT / doc["file"]
            reader = PdfReader(path)
            selected = parse_pages(doc["pages"])
            original = parse_pages(doc["original_pages"])

            if len(original) != len(reader.pages):
                raise ValueError(
                    f"{doc['id']}: 원본 대응표 {len(original)}쪽과 "
                    f"PDF {len(reader.pages)}쪽이 다릅니다."
                )
            if max(selected) > len(reader.pages):
                raise ValueError(f"{doc['id']}: PDF 범위를 벗어났습니다.")
            if group == "market":
                if not doc.get("country") or not doc.get("segments"):
                    raise ValueError(f"{doc['id']}: 국가·세그먼트가 필요합니다.")
                if set(doc["segments"]) - set(SEGMENTS):
                    raise ValueError(f"{doc['id']}: 알 수 없는 세그먼트")

            total += len(selected)
            prepared.append((group, doc, path, reader, selected, original))

    print(f"사용 페이지: {total} / 200", flush=True)
    if total > 200:
        raise ValueError("200쪽 제한 초과: 인제스트를 중단합니다.")
    return prepared


def clean_pages(reader, selected):
    """반복되는 페이지 상·하단 문구와 하이픈 줄바꿈을 정리한다."""
    raw = {}
    edge_counts = Counter()
    for number in selected:
        text = reader.pages[number - 1].extract_text() or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        raw[number] = lines
        edge_counts.update(set(lines[:2] + lines[-2:]))

    threshold = max(3, (len(selected) + 1) // 2)
    repeated = {
        line
        for line, count in edge_counts.items()
        if count >= threshold and len(line) < 180
    }

    cleaned = {}
    for number, lines in raw.items():
        kept = [
            line
            for index, line in enumerate(lines)
            if not (
                (index < 2 or index >= len(lines) - 2)
                and (line in repeated or re.fullmatch(r"\d+", line))
            )
        ]
        text = "\n".join(kept)
        text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)
        cleaned[number] = text.strip()
    return cleaned


def split_text(text: str, lang: str) -> list[str]:
    """약 800토큰, overlap 100토큰에 해당하는 문자 수로 분할한다."""
    size, overlap = (1200, 150) if lang == "ko" else (3000, 375)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary >= 0:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def build_chunks(prepared):
    grouped = {group: [] for group in COLLECTIONS}
    for group, doc, _, reader, selected, original in prepared:
        pages = clean_pages(reader, selected)
        for file_page in selected:
            text = pages[file_page]
            if not text:
                print(f"[주의] 텍스트 없음: {doc['id']} 발췌본 {file_page}쪽")
                continue

            original_page = original[file_page - 1]
            metadata = {
                "doc_id": doc["id"],
                "source": doc["title"],
                "year": str(doc["date"]),
                "page": original_page,
                "file_page": file_page,
                "lang": doc["lang"],
            }
            if group == "market":
                metadata["country"] = doc["country"]
                for segment in SEGMENTS:
                    metadata[f"seg_{segment}"] = segment in doc["segments"]

            for index, chunk in enumerate(split_text(text, doc["lang"])):
                grouped[group].append(
                    {
                        "id": f"{doc['id']}:{original_page}:{index}",
                        "text": chunk,
                        "metadata": dict(metadata),
                    }
                )

    for group, chunks in grouped.items():
        if not chunks:
            raise ValueError(f"{group}: 추출된 청크가 없습니다.")
        print(f"{COLLECTIONS[group]}: {len(chunks)}개 청크", flush=True)
    return grouped


def fingerprint(prepared, group):
    digest = hashlib.sha256()
    digest.update(REGISTRY_PATH.read_bytes())
    digest.update(f"{MODEL_NAME}|{INGEST_VERSION}".encode())
    for item_group, _, path, _, _, _ in prepared:
        if item_group == group:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def ingest(rebuild=False, check_only=False):
    prepared = load_registry()
    grouped = build_chunks(prepared)
    if check_only:
        print("검사 완료: 모델 호출·벡터스토어 변경 없음")
        return

    plans = []
    for group, name in COLLECTIONS.items():
        collection = get_collection(name)
        expected = fingerprint(prepared, group)
        metadata = collection.metadata or {}
        ready = (
            metadata.get("fingerprint") == expected
            and metadata.get("complete") is True
            and collection.count() == len(grouped[group])
        )
        if ready and not rebuild:
            print(f"{name}: 기존 컬렉션 재사용", flush=True)
            continue
        if collection.count() and not rebuild:
            raise ValueError(
                f"{name}: 문서 변경 또는 미완료 인덱스입니다. "
                "python -m rag.ingest --rebuild 로 재생성하세요."
            )
        plans.append((group, name, collection, expected))

    for group, name, collection, expected in plans:
        if rebuild:
            _client().delete_collection(name)
            collection = get_collection(name)

        chunks = grouped[group]
        collection.modify(metadata={"fingerprint": expected, "complete": False})
        embedder = get_embedder()
        for start in range(0, len(chunks), 16):
            batch = chunks[start : start + 16]
            texts = [chunk["text"] for chunk in batch]
            collection.upsert(
                ids=[chunk["id"] for chunk in batch],
                documents=texts,
                metadatas=[chunk["metadata"] for chunk in batch],
                embeddings=embedder.embed_documents(texts),
            )
            print(
                f"{name}: {min(start + 16, len(chunks))}/{len(chunks)}",
                flush=True,
            )

        collection.modify(metadata={"fingerprint": expected, "complete": True})
        print(f"{name}: 저장 완료 ({collection.count()}개)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    ingest(rebuild=args.rebuild, check_only=args.check_only)
