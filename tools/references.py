"""출처 생성 및 보고서용 필터·정렬. 반환값은 State에 저장할 dict다."""

import logging
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from hashlib import sha1
from typing import Iterable
from urllib.parse import urlsplit

import yaml

from config import ROOT
from schemas import Reference


logger = logging.getLogger(__name__)

DOCS_PATH = ROOT / "config" / "docs.yaml"
REFERENCE_GROUPS = {
    "report": "기관 보고서",
    "paper": "학술 논문",
    "web": "웹페이지",
}

def _source_id(source: str) -> str:
    return sha1(source.encode("utf-8")).hexdigest()[:10]

def _required_text(value: object, field: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"출처의 {field} 값이 비어 있습니다.")
    return str(value).strip()

def make_doc_ref(doc_id: str, startup: str, used_by: str) -> dict:
    """tech/market 문서 레지스트리에서 출처를 만든다. ID는 doc_id 기준이다."""
    if not DOCS_PATH.is_file():
        raise FileNotFoundError(f"문서 레지스트리가 없습니다: {DOCS_PATH}")
    registry = yaml.safe_load(DOCS_PATH.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("docs.yaml은 tech/market 목록을 가진 매핑이어야 합니다.")
    documents = {}
    for group in ("tech", "market"):
        entries = registry.get(group, [])
        if not isinstance(entries, list):
            raise ValueError(f"docs.yaml의 {group}은 목록이어야 합니다.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"docs.yaml의 {group} 항목은 매핑이어야 합니다.")
            entry_id = _required_text(entry.get("id"), "doc_id")
            if entry_id in documents:
                raise ValueError(f"docs.yaml에 중복된 문서 ID가 있습니다: {entry_id}")
            documents[entry_id] = entry
    if doc_id not in documents:
        raise ValueError(f"docs.yaml에 문서 ID가 없습니다: {doc_id}")
    metadata = documents[doc_id]
    ref_type = metadata.get("ref_type")
    if ref_type not in ("report", "paper"):
        raise ValueError(f"문서 출처 유형은 report 또는 paper여야 합니다: {doc_id}")
    # pages는 인제스트 범위다. 논문 서지의 페이지/기사 번호와 구분한다.
    pages = metadata.get("article_pages") if ref_type == "paper" else None
    return Reference(
        id=_source_id(doc_id),
        startup=startup,
        used_by=used_by,
        ref_type=ref_type,
        author_org=_required_text(metadata.get("author_org"), "author_org"),
        date=_required_text(metadata.get("date"), "date"),
        title=_required_text(metadata.get("title"), "title"),
        url=metadata.get("url") or None,
        venue=metadata.get("venue") or None,
        volume_issue=metadata.get("volume_issue") or None,
        pages=str(pages) if pages is not None else None,
    ).model_dump()


def _web_date(value: object, run_date: str, source_id: str) -> str:
    fallback = date.fromisoformat(run_date).isoformat()
    if value:
        text = str(value).strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            try:
                return parsedate_to_datetime(text).date().isoformat()
            except (ValueError, TypeError, OverflowError):
                logger.warning("게시일 해석 실패, 조회일 사용: source_id=%s", source_id)
    return fallback


def make_web_ref(result: dict, startup: str, used_by: str, run_date: str) -> dict:
    """검색 결과를 출처로 변환한다. 게시일을 확인할 수 없으면 조회일을 쓴다."""
    url = _required_text(result.get("url"), "url")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"웹 출처 URL은 HTTP(S) 주소여야 합니다: {url}")
    site_name = result.get("site_name") or parsed.hostname.removeprefix("www.")
    source_id = _source_id(url)
    return Reference(
        id=source_id,
        startup=startup,
        used_by=used_by,
        ref_type="web",
        author_org=result.get("author_org") or result.get("author") or site_name,
        date=_web_date(result.get("published_date"), run_date, source_id),
        title=_required_text(result.get("title"), "title"),
        venue=site_name,
        url=url,
    ).model_dump()


def format_reference(r: Reference | dict) -> str:
    """명세의 유형별 서지 형식으로 출력한다. 없는 선택 필드는 생략한다."""
    ref = Reference.model_validate(r)
    prefix = f"{ref.author_org}({ref.date}). {ref.title}."
    if ref.ref_type == "report":
        return f"{prefix} {ref.url or ''}".strip()
    if ref.ref_type == "paper":
        publication = ", ".join(
            value for value in (ref.venue, ref.volume_issue, ref.pages) if value
        )
        return f"{prefix} {publication}." if publication else prefix
    location = ", ".join(value for value in (ref.venue, ref.url) if value)
    return f"{prefix} {location}".strip()


def filter_references(
    references: Iterable[Reference | dict], startup: str | None = None
) -> list[dict]:
    """startup=None은 전체 보류 경로. 기업 지정 시 공통(*) 출처도 포함한다."""
    result = []
    before_count = 0
    for item in references:
        before_count += 1
        ref = Reference.model_validate(item)
        if startup is None or ref.startup in (startup, "*"):
            result.append(ref.model_dump())
    logger.debug(
        "출처 기업 필터: startup=%s before=%d after=%d", startup, before_count, len(result)
    )
    return result


def deduplicate_references(references: Iterable[Reference | dict]) -> list[dict]:
    """URL 또는 출처 ID가 같은 항목을 제거하고 첫 번째 출처를 유지한다."""
    result = []
    seen_urls = set()
    seen_ids = set()
    before_count = 0
    for item in references:
        before_count += 1
        ref = Reference.model_validate(item)
        duplicate = ref.id in seen_ids or (ref.url and ref.url in seen_urls)
        seen_ids.add(ref.id)
        if ref.url:
            seen_urls.add(ref.url)
        if not duplicate:
            result.append(ref.model_dump())
    logger.debug("출처 중복 제거: before=%d after=%d", before_count, len(result))
    return result


def group_references(
    references: Iterable[Reference | dict], startup: str | None = None
) -> dict[str, list[dict]]:
    """기업 필터 후 중복 제거. 유형 순서와 유형 내 날짜 내림차순을 보장한다."""
    selected = deduplicate_references(filter_references(references, startup))
    groups = {ref_type: [] for ref_type in REFERENCE_GROUPS}
    for ref in selected:
        groups[ref["ref_type"]].append(ref)
    for items in groups.values():
        items.sort(key=lambda ref: ref["date"], reverse=True)
    return groups
