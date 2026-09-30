import logging
from copy import deepcopy
from hashlib import sha1

import pytest
import yaml

from schemas import Reference
from tools import references


@pytest.fixture
def registry(tmp_path, monkeypatch):
    data = {
        "tech": [
            {"id": "tech_report", "ref_type": "report", "author_org": "DOE",
             "date": 2024, "title": "AI for Energy", "url": "https://example.org/report",
             "pages": "1-59"},
            {"id": "tech_paper", "ref_type": "paper", "author_org": "Kim, J.",
             "date": "2026", "title": "Battery AI", "venue": "Batteries",
             "volume_issue": "12(9)", "article_pages": "353", "pages": "10-20"},
        ],
        "market": [
            {"id": "market_report", "ref_type": "report", "author_org": "IEA",
             "date": "2025", "title": "Energy and AI", "url": "https://example.org/market"},
        ],
    }
    path = tmp_path / "docs.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(references, "DOCS_PATH", path)
    return data, path


def web_ref(startup="Amperon", **updates):
    result = {"url": "https://www.example.org/news", "title": "투자 소식"}
    result.update(updates)
    return references.make_web_ref(result, startup, "competitor", "2026-09-30")


# 기술·시장 레지스트리에서 스키마에 맞는 출처와 문서 ID 기반 식별자를 생성하는지 확인한다.
def test_document_sources_from_both_collections(registry):
    for doc_id, year in (("tech_report", "2024"), ("market_report", "2025")):
        ref = references.make_doc_ref(doc_id, "Amperon", "market_eval")
        assert Reference.model_validate(ref).date == year
        assert ref["startup"] == "Amperon"
        assert ref["used_by"] == "market_eval"
        assert ref["id"] == sha1(doc_id.encode()).hexdigest()[:10]


# 논문 출처에 인제스트 페이지 범위 대신 논문 번호를 사용하는지 확인한다.
def test_paper_uses_article_number_not_ingest_pages(registry):
    ref = references.make_doc_ref("tech_paper", "Amperon", "tech_summary")
    assert ref["pages"] == "353"
    assert references.format_reference(ref) == "Kim, J.(2026). Battery AI. Batteries, 12(9), 353."


# 기관 보고서 출처가 발행기관·연도·제목·URL 순서로 출력되는지 확인한다.
def test_report_format(registry):
    ref = references.make_doc_ref("tech_report", "Amperon", "tech_summary")
    assert references.format_reference(Reference(**ref)) == (
        "DOE(2024). AI for Energy. https://example.org/report"
    )


# 웹 출처의 작성자·게시일 누락 시 사이트명·조회일을 사용하고 형식에 맞게 출력하는지 확인한다.
def test_web_fallback_and_format():
    ref = web_ref()
    assert ref["id"] == sha1(ref["url"].encode()).hexdigest()[:10]
    assert ref["author_org"] == ref["venue"] == "example.org"
    assert references.format_reference(ref) == (
        "example.org(2026-09-30). 투자 소식. example.org, https://www.example.org/news"
    )


# 여러 게시일 형식을 날짜로 변환하고 해석 불가능한 값은 조회일로 대체하는지 확인한다.
@pytest.mark.parametrize("published, expected", [
    ("2025-06-01", "2025-06-01"),
    ("2025-06-01T23:10:00Z", "2025-06-01"),
    ("Sun, 01 Jun 2025 12:00:00 GMT", "2025-06-01"),
    (None, "2026-09-30"),
    ("날짜 미상", "2026-09-30"),
    ("2025", "2026-09-30"),
])
def test_web_dates(published, expected):
    assert web_ref(published_date=published)["date"] == expected


# 검색 결과에 작성자와 사이트명이 있으면 해당 값을 유지하는지 확인한다.
def test_web_explicit_author_and_site():
    ref = web_ref(author="작성자", site_name="뉴스 사이트", published_date="2026-01-01")
    assert ref["author_org"] == "작성자"
    assert ref["venue"] == "뉴스 사이트"


# 잘못된 웹 URL이나 빈 제목이 들어오면 예외를 발생시키는지 확인한다.
@pytest.mark.parametrize("updates", [{"url": ""}, {"url": "not-a-url"},
                                     {"url": "file:///tmp/news"}, {"title": ""}])
def test_invalid_web_source(updates):
    with pytest.raises(ValueError):
        web_ref(**updates)


# 문서 레지스트리 파일이 없으면 원인을 알 수 있는 파일 누락 예외를 발생시키는지 확인한다.
def test_missing_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(references, "DOCS_PATH", tmp_path / "absent.yaml")
    with pytest.raises(FileNotFoundError, match="문서 레지스트리"):
        references.make_doc_ref("missing", "Amperon", "tech_summary")


# 레지스트리에 없는 문서 ID를 요청하면 예외를 발생시키는지 확인한다.
def test_unknown_document(registry):
    with pytest.raises(ValueError, match="문서 ID가 없습니다"):
        references.make_doc_ref("missing", "Amperon", "tech_summary")


# 기술·시장 목록에 같은 문서 ID가 중복 등록되면 예외를 발생시키는지 확인한다.
def test_duplicate_document_id(registry):
    data, path = registry
    data["market"].append(data["tech"][0])
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="중복된 문서 ID"):
        references.make_doc_ref("tech_report", "Amperon", "tech_summary")


# 레지스트리의 최상위 구조나 문서 목록 형식이 잘못되면 예외를 발생시키는지 확인한다.
@pytest.mark.parametrize("data", [None, [], {"tech": {}}, {"tech": ["bad"]}])
def test_invalid_registry(registry, data):
    _, path = registry
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        references.make_doc_ref("tech_report", "Amperon", "tech_summary")


# 기업 필터를 먼저 적용해 대상·공통 출처를 보존하고 원본 데이터를 변경하지 않는지 확인한다.
def test_filter_before_deduplication_preserves_target_source():
    refs = [web_ref("Other"), web_ref("Amperon"),
            web_ref("*", url="https://example.org/common")]
    original = deepcopy(refs)
    groups = references.group_references(refs, "Amperon")
    assert [r["startup"] for r in groups["web"]] == ["Amperon", "*"]
    assert refs == original


# URL 또는 문서 기반 식별자가 같은 출처는 첫 항목만 남기는지 확인한다.
def test_dedup_by_url_or_document_id(registry):
    doc = references.make_doc_ref("tech_paper", "Amperon", "tech_summary")
    other_use = {**doc, "startup": "Other", "used_by": "market_eval"}
    web = web_ref()
    same_url = {**web, "id": "different"}
    assert references.deduplicate_references([doc, other_use, web, same_url]) == [doc, web]


# 전체 보류 경로에서 기업 구분 없이 출처를 모아 유형 순서와 날짜 내림차순으로 정렬하는지 확인한다.
def test_group_order_and_dates_for_all_hold(registry):
    old = references.make_doc_ref("tech_report", "Amperon", "tech_summary")
    new = references.make_doc_ref("market_report", "Other", "market_eval")
    paper = references.make_doc_ref("tech_paper", "Amperon", "tech_summary")
    groups = references.group_references([web_ref(), old, paper, new])
    assert list(groups) == ["report", "paper", "web"]
    assert groups["report"] == [new, old]
    assert groups["paper"] == [paper]


# 논문의 선택 필드가 없으면 None 문자열이나 불필요한 구두점 없이 출력하는지 확인한다.
def test_missing_optional_fields_do_not_print_none(registry):
    ref = references.make_doc_ref("tech_paper", "Amperon", "tech_summary")
    ref.update(venue=None, volume_issue=None, pages=None)
    assert references.format_reference(ref) == "Kim, J.(2026). Battery AI."


# 출처가 없어도 세 유형의 빈 목록을 반환하는지 확인한다.
def test_empty_references():
    assert references.group_references([]) == {"report": [], "paper": [], "web": []}


# 게시일 해석 실패 시 출처 ID만 경고에 남기고 원문과 URL 토큰은 노출하지 않는지 확인한다.
def test_invalid_date_logs_warning_without_raw_data(caplog):
    with caplog.at_level(logging.DEBUG, logger="tools.references"):
        ref = web_ref(published_date="private-date-value",
                      url="https://example.org/news?token=secret")
    assert ref["date"] == "2026-09-30"
    assert caplog.record_tuples == [
        ("tools.references", logging.WARNING,
         f"게시일 해석 실패, 조회일 사용: source_id={ref['id']}")
    ]
    assert "secret" not in caplog.text
    assert "private-date-value" not in caplog.text


# 게시일이 없거나 정상적인 경우 불필요한 로그를 남기지 않는지 확인한다.
@pytest.mark.parametrize("published", [None, "", "2026-01-01"])
def test_expected_dates_do_not_log(caplog, published):
    with caplog.at_level(logging.DEBUG, logger="tools.references"):
        web_ref(published_date=published)
    assert not caplog.records


# 일회성 이터레이터와 빈 입력에서도 필터·중복 제거 전후 건수를 DEBUG로 정확히 기록하는지 확인한다.
@pytest.mark.parametrize("empty", [False, True])
def test_debug_counts_with_generator_input(caplog, empty):
    refs = [] if empty else [web_ref("Other"), web_ref(), web_ref()]
    with caplog.at_level(logging.DEBUG, logger="tools.references"):
        groups = references.group_references(iter(refs), "Amperon")
    before, filtered, unique = (0, 0, 0) if empty else (3, 2, 1)
    assert len(groups["web"]) == unique
    assert caplog.record_tuples == [
        ("tools.references", logging.DEBUG,
         f"출처 기업 필터: startup=Amperon before={before} after={filtered}"),
        ("tools.references", logging.DEBUG,
         f"출처 중복 제거: before={filtered} after={unique}"),
    ]


# INFO 레벨에서는 출처 처리 건수의 DEBUG 로그가 출력되지 않는지 확인한다.
def test_info_level_hides_debug_counts(caplog):
    with caplog.at_level(logging.INFO, logger="tools.references"):
        references.group_references([web_ref(), web_ref()])
    assert not caplog.records


# 레지스트리 조회 실패 시 하위 모듈이 중복 로그 없이 예외만 전달하는지 확인한다.
def test_registry_failure_does_not_log_twice(registry, caplog):
    with caplog.at_level(logging.DEBUG, logger="tools.references"):
        with pytest.raises(ValueError, match="문서 ID가 없습니다"):
            references.make_doc_ref("missing", "Amperon", "tech_summary")
    assert not caplog.records
