# LangSmith 실행 화면 캡처

실행 ID: `b0a8c7d6874440c98a669596e47110e6`
브라우저: 사용자 Aside의 실제 LangSmith UI
[전체 트레이스](https://smith.langchain.com/o/9f11065d-205c-42b4-a4fd-a6b3e8cf79de/a/energy-orchestrator-workers/projects/p/0517a558-40a2-4809-9495-a00c50123fde/trace/b0a8c7d6-8744-40c9-8a66-9596e47110e6/run/b0a8c7d6-8744-40c9-8a66-9596e47110e6)

| 파일 | 화면 내용 |
|---|---|
| tracing-1.png | 실행 시작·탐색·7개 worker·집계, 최종 complete와 동일 실행 PDF 경로 |
| tracing-2.png | 첫 후보 판단 이후 다음 후보 선택·7개 worker 반복 |
| tracing-3.png | Splight 조사 계획 3개와 실제 3개 worker·집계 |
| tracing-4.png | 임팩티브AI 조사 계획 7개와 7개 worker·보고서 생성/평가/확정 흐름 |
| tracing-5.png | Groundedness·중립성·편향 모두 passed=true인 실제 품질 평가 결과 |
| tracing-6.png | report_finalize의 complete와 report_v1.pdf 저장 경로 |

이 실행은 웹 탐색으로 Piq Energy, Tensor Energy, Splight, 임팩티브AI를 평가했고 조사 계획 수는 7·7·3·7개다. 보고서 v1에서 품질을 통과했으므로 보고서 재평가 루프가 실행된 증거로 표시하지 않는다.

PNG는 화면의 빈 흰색 여백만 잘랐다. 편집 전 전체 캡처와 ZIP은 로컬 outputs/runs/b0a8c7d6874440c98a669596e47110e6/langsmith/에 보존했다.
