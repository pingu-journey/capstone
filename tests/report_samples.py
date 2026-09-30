"""조판 검증용 가상 데이터. 실제 기업 분석·투자 판단 결과가 아니다."""

from report.templates import Block, Chapter, ReportDocument, TEMPLATES, Table
from agents.investment_judge import load_rubric
from agents.report_writer import _candidate_table, _competitor_table, _market_table, _score_table
from tools.references import group_references, make_web_ref


def sample_report(route="investment", count=3, stress=False):
    source = make_web_ref({"title": "조판 검증용 가상 데이터", "url": "https://example.org/test-evidence"},
                          "검증용 에너지 기업", "test", "2026-09-30")
    refs = [source]
    if stress:
        refs += [make_web_ref({"title": f"검증용 참고 자료 {i + 1}",
                              "url": f"https://example.org/evidence/{i}/" + "long-url-segment-" * 12},
                             "*", "test", "2026-09-30") for i in range(12)]
    citation = f" [출처: {source['id']}]"
    investment = [
        [("사업 아이디어와 제품", "검증용 에너지 기업은 전력 수요 예측 소프트웨어를 공급하는 가상의 기업이다. "
          "고객은 발전량과 수요의 차이를 확인하여 운영 계획을 수립한다. 본 보고서의 기업·고객·성과는 모두 조판 테스트를 위한 예시이며 실제 투자 근거로 사용할 수 없다."),
         ("기술 성숙도와 사업모델", "TRL 8, Series B 단계로 설정한 가상 데이터다. 전력 사용량과 기상 데이터를 결합하는 구독형 서비스를 가정했다. "
          "제공된 데이터 범위 밖의 성능이나 계약 조건은 공개 정보 부족으로 표시한다.")],
        [("시장과 고객 수요", "시장 수치는 표의 값·단위·기준 연도를 함께 확인해야 한다. 가상 시장 용량은 2030년 100 GW로 설정했다. "
          "유틸리티와 전력 판매사가 주요 수요처라는 가정으로 작성했으며, 실제 시장 전망을 의미하지 않는다." + citation),
         ("경쟁 환경과 차별성", "아래 비교표는 같은 기준으로 경쟁사의 제공 기술과 대상 대비 차이를 정리한다. 정성 정보를 임의의 점수로 바꾸지 않는다. "
          "데이터 연동 범위와 제품 적용 조건을 비교하고, 확인되지 않은 경쟁우위는 추가 검증 대상으로 남긴다.")],
        [("팀과 사업 성과", "가상의 창업팀은 에너지 운영과 AI 개발 경험을 보유한 것으로 설정했다. 고객과의 실증은 진행된 것으로 가정하지만 "
          "매출·계약 규모는 공개 정보 부족이다. 자료의 부족과 실제 역량의 부족을 구분해야 한다."),
         ("핵심 리스크", "입력 데이터 품질, 지역별 제도 변화, 경쟁 제품의 도입 비용을 주요 검토 항목으로 설정했다. "
          "실제 투자를 검토할 때는 고객 확인과 기술 검증, 규제 검토가 추가로 필요하다.")],
        [("종합 판단", "가상 점수 80.0점은 기준 70점을 넘으며 필수 탈락 조건은 없다. 이 결과는 점수 계산과 표·그래프의 일치 여부를 검증하기 위한 값이다."),
         ("분석의 한계", "이 문서는 조판 검증 전용이다. 실제 회사·시장·거래 조건을 조사하지 않았으며 모든 수치는 예시다. "
          "실제 보고서는 분석 에이전트가 수집한 근거와 확정된 채점 결과에서 생성해야 한다.")],
    ]
    hold = [
        [("평가 조건과 대상", f"Energy 분야 비상장·Seed~Series C·Exit 미완료 기업을 탐색하는 조건으로 구성한 가상 평가다. 평가 완료 후보는 {count}곳이다. "
          + ("평가 가능한 후보를 확보하지 못함." if not count else "후보별 점수와 보류 사유는 누적 평가 이력에서 가져왔다."))],
        [("후보별 결과", "각 후보의 총점과 판정, 취약 항목은 아래 표와 같다. 마지막 후보의 점수를 전체 후보의 대표 점수로 사용하지 않는다. "
          "모두 보류된 상태이며 최고점 표시가 투자 추천을 뜻하지 않는다.")],
        [("반복되는 취약 항목", "가상 평가 이력에서는 투자조건과 재무·투자 가능성을 공통 취약 항목으로 설정했다. 가중 점수 하위 항목과 실제 사업 위험은 구분해서 해석한다."),
         ("시장·규제 환경", "시장 정보는 보존된 캐시의 수치만 사용한다. 캐시에 없는 지역별 규제나 후보의 상세 분석을 새로 추정하지 않는다. "
          "실제 데이터가 없으므로 추가 확인이 필요하다.")],
        [("자료와 평가 범위", "이 보고서는 가상 평가 이력에 기반한 조판 검증용 문서다. 이전 후보의 상세 분석과 개별 항목 점수는 보존하지 않는 조건을 가정했다. "
          "공개 정보 부족으로 인한 5점 적용은 실제 사업의 안정성이나 성과를 보증하지 않는다.")],
    ]
    if not count:
        hold[1] = [("평가 결과 없음", "평가 기록이 없어 후보별 점수·판정을 비교할 수 없다. 평가 가능한 후보를 확보하지 못함.")]
        hold[2] = [("공통 사유 확인 불가", "평가 이력이 없어 공통 취약 항목을 도출할 수 없다. 보존된 시장 캐시도 없어 시장·규제 환경을 추정하지 않는다.")]
        refs = []
    chapters = [Chapter(number=i + 1, title=template.title, char_budget=template.char_budget,
                        blocks=[Block(subtitle=subtitle, text=text) for subtitle, text in content])
                for i, (template, content) in enumerate(zip(TEMPLATES[route], investment if route == "investment" else hold))]
    market = {"country": "US", "segment": "demand_forecasting", "market_size": [
        {"metric": "가상 시장 용량", "value": "100 GW", "year": "2030", "source_id": source["id"]}]}
    if route == "investment":
        scores = {item["id"]: {"score": 8, "rationale": "조판 검증용 가상 근거. 실제 투자 평가가 아님." + citation}
                  for item in load_rubric()}
        if stress:
            scores["team"]["rationale"] = "긴 근거의 줄바꿈과 행 높이를 확인한다. " * 25 + citation
        chapters[1].tables = [_competitor_table({"competitors": [
            {"name": f"가상 경쟁사 {i + 1}", "country": "KR", "status": "Series A",
             "offering": "전력 예측·운영 최적화", "vs_target": "연동 범위와 운영 방식이 다른 것으로 가정한 정성 비교"}
            for i in range(3)]}), _market_table([market])]
        chapters[3].tables = [_score_table(scores, load_rubric(), 80)]
        summary = ("투자 추천, 총점 80.0점. 조판 검증용 가상 판정이며 실제 투자 추천이 아니다.\n"
                   "투자 포인트: 기술 적용 가능성; 가상 고객 수요; 팀 경험의 가정\n"
                   "리스크: 실제 근거 미확인; 거래 조건 추가 검증 필요")
    else:
        history = [{"startup": f"가상 후보 {i + 1}", "total_score": 51.2 + i * 2.1, "decision": "보류",
                    "weakest_items": ["투자조건", "재무·투자 가능성"], "knockout": None,
                    "key_reason": "총점 기준 미달. 공개 정보 부족: 투자조건(5점 적용)."} for i in range(count)]
        chapters[1].tables = [_candidate_table(history)]
        chapters[2].tables = [_market_table([market])] if count else []
        summary = f"투자 대상 없음(전체 보류). 평가 후보 {count}곳. 조판 검증용 가상 결과다.\n"
        summary += "공통 보류 사유: " + (f"투자조건·재무 정보 부족 ({count}/{count}곳)." if count else "평가 가능한 후보를 확보하지 못함.")
    return ReportDocument(route=route, title="[조판 검증용] " + ("에너지 기업 투자 평가" if route == "investment" else "Energy AI 평가 결과"),
                          run_date="2026-09-30", decision="투자" if route == "investment" else "보류",
                          total_score=80 if route == "investment" else None, summary=summary,
                          chapters=chapters, references=group_references(refs))
