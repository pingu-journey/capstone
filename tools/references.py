import hashlib
from datetime import date
from email.utils import parsedate_to_datetime
from functools import lru_cache
from urllib.parse import urlparse

import yaml

from config import ROOT
from schemas import Reference


DOCS_YAML = ROOT / "config" / "docs.yaml"


def _ref_id(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:10]


@lru_cache(maxsize=1)
def _load_docs() -> dict[str, dict]:
    if not DOCS_YAML.exists():
        raise FileNotFoundError(f"{DOCS_YAML}가 없습니다. 문서 레지스트리를 먼저 작성하세요.")
    registry = yaml.safe_load(DOCS_YAML.read_text(encoding="utf-8")) or {}
    return {doc["id"]: doc for group in registry.values() for doc in group}


def _site_name(url: str) -> str:
    host = urlparse(url).netloc
    return host.removeprefix("www.") or url


def _web_date(published: str | None, run_date: str) -> str:
    if not published:
        return run_date
    try:
        return date.fromisoformat(published[:10]).isoformat()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(published).date().isoformat()
    except (TypeError, ValueError):
        return run_date


def make_doc_ref(doc_id: str, startup: str, used_by: str) -> dict:
    doc = _load_docs().get(doc_id)
    if doc is None:
        raise KeyError(f"docs.yaml에 없는 doc_id: {doc_id}")
    return Reference(
        id=_ref_id(doc_id),
        startup=startup,
        ref_type=doc["ref_type"],
        author_org=doc["author_org"],
        date=str(doc["date"]),
        title=doc["title"],
        venue=doc.get("venue"),
        volume_issue=doc.get("volume_issue"),
        pages=doc.get("article_pages"),
        url=doc.get("url"),
        used_by=used_by,
    ).model_dump()


def make_web_ref(result: dict, startup: str, used_by: str, run_date: str) -> dict:
    url = result["url"]
    site = _site_name(url)
    return Reference(
        id=_ref_id(url),
        startup=startup,
        ref_type="web",
        author_org=result.get("author") or site,
        date=_web_date(result.get("published_date"), run_date),
        title=result.get("title") or site,
        venue=site,
        url=url,
        used_by=used_by,
    ).model_dump()
