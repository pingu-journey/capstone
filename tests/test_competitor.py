"""경쟁사 비교의 근거 번호 검증 함수 단위 테스트."""
from agents.competitor import INSUFFICIENT, split_citations, verify_items, verify_status


def test_split_citations_strips_numbers_and_drops_out_of_range():
    assert split_citations("유틸리티 A사와 상용 계약 [2][9]", 5) == ("유틸리티 A사와 상용 계약", [2])
    assert split_citations("특허 [1] 보유", 3) == ("특허 보유", [1])


def test_verify_items_keeps_only_cited_items():
    items = ["상용 계약 [1]", "근거 없는 주장", "범위 밖 [7]", f"{INSUFFICIENT} [2]", "데이터 [2][3]"]
    kept, used = verify_items(items, 3)
    assert kept == ["상용 계약", "데이터"]
    assert used == {1, 2, 3}


def test_verify_items_empty_when_nothing_cited():
    assert verify_items(["근거 없는 주장"], 3) == ([], set())


def test_verify_status():
    assert verify_status("Series A", 2, 3) == ("Series A", 2)
    assert verify_status("상장", None, 3) == (INSUFFICIENT, None)
    assert verify_status("상장", 5, 3) == (INSUFFICIENT, None)
