# Orchestrator-Workers State 설계

기존 InvestState의 분석·점수·판정 계약을 유지하면서 제어와 외부 artifact 참조를 확장했다.

| 설계 항목 | 구현과 이유 |
|---|---|
| 제어 / 페이로드 | run_id, tasks, report_version, quality_result, execution_status는 실행 제어. worker_results, evidence_index_ref, report_draft_ref, report_preview_ref는 작업 요약·외부 파일 참조. 원문과 초안을 State에 넣지 않는다. |
| 관측성 | plans·workers·analyses·quality 파일과 표준 로깅·LangSmith에 판단 근거를 기록한다. State.log는 최종 결과와 함께 보는 짧은 주요 이력이다. |
| 지속성 비용 | task/result는 select에서 초기화한다. 평가 이력은 최대 8개. 원문·검색 청크·초안은 실행 디렉터리 파일에 보존한다. references와 짧은 log는 종료 상한으로 유한하다. |
| 상관 | app가 만든 run_id를 디렉터리·manifest·작업 파일·LangSmith run metadata에 사용한다. task_id는 run_id+기업명의 hash+작업 종류다. 다른 실행 파일을 집계하지 않는다. |
| 복구 | TaskResult.status는 success/insufficient/failed, attempts는 1~2, error_code는 안전한 분류값이다. 일시 실패는 1회 재시도하고 다른 worker는 계속한다. 인증·설정은 실행 실패. checkpoint 재개는 범위 밖이며 새 실행에서 검색 캐시를 재사용한다. |
| 동시성 | worker는 worker_results만 반환한다. task_id 기준 정렬된 merge_worker_results는 동일 delta 재전달에 멱등이며 서로 다른 결과 충돌을 거부한다. select는 Overwrite({})를 worker 실행 전에 한 번 쓴다. 파일은 작업마다 분리하고 JSON 캐시는 원자적 교체한다. 로컬 torch 임베딩·리랭커는 같은 lock, 시장 검색은 별도 lock으로 보호한다. |
| 종료 | 탐색 3회·평가 8개·RAG rewrite 2회·호출 최대 2회·보고서 최초+개선 2회. 후보 루프는 후보당 select/plan/worker/join/competitor/judge 6개 superstep이며 최대 48. 초기화·탐색·품질 루프를 포함해 recursion_limit=100 안에서 종료한다. |

## Reader / Writer

- app/initialize: run_id·공통 설정·초기 상태.
- discover/select: 후보 큐·평가 대상·후보별 초기화.
- orchestrator: tasks 저장. 캐시의 항목별 내용과 원문 근거를 검사한다.
- research_worker: task별 최소 문맥을 받고 worker_results delta만 쓴다. 점수·판정·캐시를 직접 쓰지 않는다.
- synthesizer: 계획된 모든 결과를 검증하고 tech_summary·market_analysis·market_cache·research_cache·references의 신규 delta와 evidence_index_ref를 쓴다. 순서는 계획을 따른다.
- competitor/judge: 기존 분석·점수·판정 계약. judge는 새 EvalRecord 하나만 추가한다.
- report_draft: report_version을 1 증가시키고 초안·preview의 파일 참조를 쓴다. 최초 생성 이후에는 이전 artifact의 정상 블록을 유지하고 quality_result.issues의 장·블록 위치만 수정한다. 실패 원문·수치·출처와 원문 발췌를 블록 수정 입력에 전달하며, 수정 후에도 코드 검증에 실패하면 해당 블록을 정보 부족으로 대체한다. SUMMARY는 수정된 본문으로 재생성하고 수치·출처·판정·분량을 다시 검사한다.
- report_quality: 코드 구조·분량·판정·수치 검사와 LLM의 의미·중립성·편향 검사를 결합한 QualityVerdict를 쓴다. 실패한 코드 검사를 LLM이 덮어쓸 수 없다.
- report_finalize: 통과 버전과 PDF·본문 hash가 일치할 때만 최종 파일을 저장하고 complete.
- quality_failed: report_path=None, 검토용 파일·미달 사유를 보존하고 종료.

## 근거와 캐시

웹 본문 및 RAG 발췌의 source_id·기업·URL·페이지를 캡처한다. 최종 분석에 실제 채택한 Reference를 기업별로 연결하고, 품질 Judge는 그 source_id의 발췌를 읽는다. 시장 캐시 재사용 시 현재 기업으로 출처와 근거를 다시 연결한다. 유효 ID만 있고 원문이 없는 캐시는 재조사한다.

작업 0개이면 Send를 만들지 않고 synthesizer로 직접 이동한다. worker 결과 키 집합은 계획의 task_id 집합과 정확히 일치해야 한다. 부족·실패 작업은 기존 fallback과 정보 부족 표시로 반영하며 기존 캐시의 성공 항목을 덮어쓰지 않는다.
시장 집계에는 success 작업의 payload만 넣고, task별 target_fields와 실제 부족 필드의 교집합만 갱신한다. 캐시가 없어도 fallback에서 시작해 실패 관점을 다른 관점의 전체 출력으로 채우지 않는다. HTTP 400/401/403 등 요청·인증 오류는 안전한 ConfigurationError로 전파하며 408/429/5xx·연결·timeout만 기존 상한 안에서 재시도한다.

## 보고서 revision

분량 축소는 품질 검사 전에 완료한다. 품질 미달은 다음 version에 고정 장·블록 위치를 전달하며, 코드가 해당 위치의 실제 원문·출처·수치를 채운다. 동일 블록이 다음 품질 평가에서도 실패하면 추가 LLM 수정 대신 정보 부족으로 대체한다. 판정 전용 QualityAssessment와 미달 항목 위치 전용 ReportIssueTargets를 분리한다. 통과하면 위치 모델을 호출하지 않는다. 각 단계의 호출·응답 검증은 최대 두 번이며, 검증 실패 또는 안전한 수정 위치가 없으면 recoverable=false로 종료한다. 수정 전·후와 실패 이력은 draft artifact의 repair_log에 보존한다. 최종 저장 때 초안을 재생성하거나 PDF를 다시 렌더하지 않는다. source ID의 존재는 의미적 근거의 증명이 아니며 독립 Judge가 발췌와 주장을 대조한다.
투자 보고서 1장의 TRL 평가 문구는 State trl과 일치하고 판단근거·기술 출처 발췌가 있을 때 해당 TRL 숫자 위치만 원문 숫자 검사에서 제외한다. 같은 값의 시장 수치나 State와 다른 TRL 평가는 제외하지 않는다. Judge에는 tech_summary를 전달해 평가값·판단근거·인용 원문의 의미를 대조한다. 천 단위 쉼표와 정확히 동치인 규모·단위 표현을 Decimal로 정규화한다($20 million=2000만 달러, 1.3 GW=1300 MW). 통화·차원이 달라지면 거부하며 환율 환산·반올림은 수행하지 않는다. 대상·기간·주장 의미는 독립 Judge가 검사한다.


## 과제 분량 기준

최신 Multi-Agent Orchestration 지침의 최대 10쪽을 적용한다. PDF 렌더러·생성·평가·확정·E2E는 공통 MAX_PAGES=10을 사용하며 REFERENCE를 포함한 실제 페이지 수를 검사한다. SUMMARY는 연결된 기존 보고서 지침의 반 페이지 제한을 유지한다. 장별 char_budget과 SUMMARY 550자는 생성 길이 안내이고 품질 탈락 기준이 아니다. 글자 수 초과만으로 근거 있는 본문을 정보 부족 문구로 대체하지 않는다.
