"""Actual excerpts remain attached to selected references and isolated across threads."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools import artifacts
from tools.artifacts import read_json, run_dir, write_json
from tools.evidence import capture, index_evidence, record, save_node
from tools.references import make_web_ref
from tools.references import unsupported_numbers


def test_parallel_capture_isolates_workers():
    def work(name):
        with capture() as records:
            ref = make_web_ref({"title": name, "url": f"https://example.org/{name}", "content": f"{name} 원문"}, name, "worker", "2026-10-13")
        return records, ref
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, ["A", "B", "C", "D"]))
    for (records, ref), name in zip(results, ["A", "B", "C", "D"]):
        assert len(records) == 1 and records[0]["startup"] == name
        assert records[0]["source_id"] == ref["id"] and records[0]["text"] == f"{name} 원문"


def test_unselected_search_result_does_not_become_reference_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUTPUT_DIR", tmp_path)
    state = {"run_id": "selection", "search_round": 1}
    with capture() as records:
        first = make_web_ref({"title": "활용", "url": "https://example.org/a", "content": "채택한 사실"}, "A", "test", "2026-10-13")
        make_web_ref({"title": "미활용", "url": "https://example.org/b", "content": "미활용 검색 결과"}, "A", "test", "2026-10-13")
    save_node(state, "discover", {"references": [first]}, records)
    index = read_json(index_evidence(state))
    assert len(index) == 1 and index[0]["text"] == "채택한 사실"


def test_original_document_page_and_full_excerpt_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUTPUT_DIR", tmp_path)
    reference = {"id": "doc_id", "startup": "A", "url": "https://example.org/doc"}
    original = "발췌문 " * 1000
    with capture() as records:
        record(reference, original, 109)
    path = write_json(run_dir({"run_id": "page"}) / "evidence/doc.json", records)
    assert read_json(path)[0]["page"] == 109 and read_json(path)[0]["text"] == original


def test_numeric_provenance_uses_cited_excerpts_and_thousands_separator():
    evidence = [{"source_id": "power", "text": "2024 demand 150,540 GWh"},
                {"source_id": "funding", "text": "2023 Series B $20 million"}]
    assert unsupported_numbers("2024년 150540 GWh [출처: power]", evidence) == set()
    assert unsupported_numbers("2023년 $20 million [출처: power]", evidence) == {"2023", "20"}
    assert unsupported_numbers("2023년 $20 million [출처: funding]", evidence) == set()


def test_amount_unit_conversion_is_not_literal_provenance():
    # The report contract preserves the source's amount and unit spelling.
    evidence = [{"source_id": "funding", "text": "$20 million"}]
    assert unsupported_numbers("유치액은 2000만 달러다 [출처: funding]", evidence) == {"2000"}


def test_trl_assessment_is_not_a_value_whitelist():
    evidence = [{"source_id": "tech", "text": "현장 실증 근거"}, {"source_id": "market", "text": "전력 시장 근거"}]
    kwargs = {"tech_summary": {"trl": 7, "trl_rationale": "현장 실증 근거", "insufficient": []},
              "assessment_source_ids": {"tech"}}
    assert unsupported_numbers("TRL 7로 평가한다 [출처: tech]", evidence, **kwargs) == set()
    assert unsupported_numbers("TRL 7로 평가한다. 용량 7 GW다 [출처: tech]", evidence, **kwargs) == {"7"}
    assert unsupported_numbers("TRL 7로 평가한다 [출처: market]", evidence, **kwargs) == {"7"}
    assert unsupported_numbers("TRL 7이다 [출처: tech]", evidence, **kwargs) == {"7"}
    kwargs["tech_summary"]["trl_rationale"] = "공개 정보 부족"
    assert unsupported_numbers("TRL 7로 평가한다 [출처: tech]", evidence, **kwargs) == {"7"}


def test_trl_assessment_state_mismatch_even_if_number_is_in_excerpt():
    evidence = [{"source_id": "tech", "text": "현장 실증 8건"}]
    assert unsupported_numbers("TRL 8로 평가한다 [출처: tech]", evidence,
        tech_summary={"trl": 7, "trl_rationale": "현장 실증 근거", "insufficient": []},
        assessment_source_ids={"tech"}) == {"8"}
