# 구현 작업 분배 (4인)

> 이 문서는 `IMPLEMENTATION_PLAN.md`(이하 **명세서**)를 4명이 **동시에** 구현하기 위한 분배표다.
> 구현 세부는 명세서의 해당 절을 따르고, 이 문서는 **누가 어떤 파일을 책임지는지, 서로 어떤 함수로 연결되는지, 언제까지 무엇을 끝내는지**만 정한다.
> 마감: **DAY 3 15:00** (GitHub 링크 + 투자 보고서 PDF 제출)

---

## 0. 한눈에 보기

| 담당 | 역할 | 소유 파일 | 명세서 절 | 관련 평가 항목 |
|---|---|---|---|---|
| **A** | 공통 기반 · 그래프 · 탐색 | `config.py` `state.py` `schemas.py` `tools/llm.py` `tools/web_search.py` `graph/*` `agents/discovery.py` `agents/select_candidate.py` `app.py` `README.md` `tests/test_routing.py` | 1, 2, 3, 4, 6.1, 6.2, 9, 10 | 설계 구현 충실도, 코드 구조, 재현성, README |
| **B** | RAG 인프라 · 시장성 평가 | `config/docs.yaml` `rag/ingest.py` `rag/embeddings.py` `rag/advanced_rag.py` `eval/embedding_eval.py` `data/eval/*` `agents/market_eval.py` | 5.1, 5.2, 5.4, 5.5, 6.4 | **RAG Pipeline(20)** |
| **C** | Agentic RAG · 기술 요약 · 경쟁사 비교 | `rag/agentic_rag.py` `agents/tech_summary.py` `agents/competitor.py` `prompts/tech_summary.md` `prompts/competitor.md` | 5.3, 6.3, 6.5 | **RAG Pipeline(20)**, Agent 구현 |
| **D** | 투자 판단 · 보고서 · REFERENCE | `config/rubric.yaml` `agents/investment_judge.py` `agents/report_writer.py` `report/*` `tools/references.py` `assets/fonts/*` `tests/test_scoring.py` `tests/test_references.py` | 7, 8, 6.6, 6.7 | **Output 보고서(20)**, Agent 구현 |

- **공통 스키마(`schemas.py`, `state.py`)는 A만 수정한다.** 필드가 더 필요하면 A에게 요청하고, A가 반영 후 공지한다.
- 프롬프트 파일은 각 에이전트 담당자가 소유한다 (`prompts/{에이전트}.md`).
- `main` 머지는 **4명 모두** 할 수 있다 (6장 협업 규칙). 최종 E2E 실행은 **A**가 책임진다.

---

## 1. 전원 공통 — Phase 0 (시작 후 30분 안에 끝낼 것)

병렬 작업이 막히지 않도록 **인터페이스부터 먼저 고정**한다.

| 순서 | 누가 | 할 일 | 산출물 |
|---|---|---|---|
| 1 | A | 저장소 생성, 디렉터리 뼈대(명세서 2장), `requirements.txt`, `.env.example`, `.gitignore`(`.env`, `data/vectorstore/`, `outputs/cache/`) | `main` 브랜치 첫 커밋 |
| 2 | A | `schemas.py`, `state.py`, `config.py`(상수값 포함) 작성 후 push | 모든 담당자가 import 가능 |
| 3 | A | `tools/llm.py`, `tools/web_search.py`(캐시 포함) 작성 | `search()`, `get_llm()` 사용 가능 |
| 4 | D | `tools/references.py`의 **생성 헬퍼 2개**를 먼저 push (포맷터는 나중에) | `make_doc_ref`, `make_web_ref` 사용 가능 |
| 5 | A | `tests/fixtures/sample_state.json` — 시드 후보 1곳(Amperon)의 가짜 분석 결과가 채워진 State | B·C·D가 단독 테스트에 사용 |
| 6 | 전원 | 각자 브랜치 생성: `feat/a-graph`, `feat/b-rag`, `feat/c-agentic`, `feat/d-report` | — |

---

## 2. 인터페이스 약속 (함수 시그니처는 바꾸지 않는다)

| 제공자 | 함수 | 반환 | 사용자 |
|---|---|---|---|
| A | `tools.llm.get_llm()` / `get_judge_llm()` | `ChatOpenAI` (temperature=0) | 전원 |
| A | `tools.web_search.search(query: str, max_results: int = 5)` | `list[dict]` — `{"title","url","content","published_date"}` (캐시 자동) | A, C |
| D | `tools.references.make_doc_ref(doc_id: str, startup: str, used_by: str)` | `dict` (Reference) — `docs.yaml` 메타데이터로 생성 | B, C |
| D | `tools.references.make_web_ref(result: dict, startup: str, used_by: str, run_date: str)` | `dict` (Reference) | A, C |
| B | `rag.embeddings.get_embedder()` | `.embed_documents(list[str])`, `.embed_query(str)` | B, C |
| B | `rag.advanced_rag.search(query: str, country: str, segment: str, k: int = 5)` | `list[dict]` — `{"text","doc_id","page","score"}` | B(시장성) |
| B | `rag.ingest.get_collection(name: str)` | Chroma 컬렉션 (`"tech_docs"` / `"market_docs"`) | C |
| C | `rag.agentic_rag.run(question: str, company: str, startup: str)` | `dict` — `{"answer": str, "sources": list[Reference], "path": "rag" \| "web" \| "rag+web", "rewrites": int}` | C(기술 요약) |
| 각 담당 | `agents.<name>.run(state) -> dict` | 명세서 3.3 Writer 표에 적힌 키만 반환 | A(그래프) |

**B가 벡터스토어를 만들기 전(Phase 1 초반)에 C가 막히지 않도록:** C는 `get_collection("tech_docs")`가 비어 있으면 곧바로 웹 검색 경로로 가는 분기를 먼저 구현하고, B의 인제스트가 끝나면 RAG 경로를 붙인다.

---

## 3. 담당별 상세

### 담당 A — 공통 기반 · 그래프 · 탐색

**구현 목록**
1. `config.py` — 명세서 4.4의 상수, `.env` 로딩, 경로 상수(`ROOT`, `DATA_DIR`, `OUTPUT_DIR`)
2. `schemas.py`, `state.py` — 명세서 3.1, 3.2 그대로
3. `tools/llm.py`, `tools/web_search.py` — Tavily 래퍼, `outputs/cache/search/{sha1}.json` 캐시, `--no-cache` 지원
4. `agents/select_candidate.py` — 명세서 6.2 (코드 그대로)
5. `agents/discovery.py` — 명세서 6.1 (고정 쿼리 템플릿, 요건 검증, 유망도 정렬, 시드 대비책, `outputs/discovery_round{n}.json`)
6. `data/seed_candidates.json` — 명세서 6.1의 5곳 (홈페이지 URL 접속 확인)
7. `graph/builder.py`, `graph/routing.py` — 명세서 4.2, 4.3 그대로. fan-in은 `add_edge(["tech_summary","market_eval"], "competitor")`
8. `app.py` — 명세서 9.1 (`--seed-only`, `--no-cache`, `outputs/run_log.json`, `outputs/graph.mmd` 저장)
9. `tests/test_routing.py` — 명세서 9.3
10. `README.md` — 명세서 10장 형식, `docs/graph.png`(설계서 그래프 이미지) 삽입

**단독 테스트:** 다른 에이전트가 없을 때는 `graph/builder.py`에 `stub=True` 옵션을 두어 각 노드를 `sample_state.json` 값을 반환하는 가짜 함수로 대체하고, 투자/보류/재탐색/모두 보류 4가지 경로가 모두 끝까지 도는지 확인한다.

**완료 조건 (DoD)**
- [ ] `python app.py --seed-only`가 stub 노드로 END까지 실행
- [ ] `pytest tests/test_routing.py` 통과
- [ ] 탐색 에이전트가 실제 웹 검색으로 요건을 충족한 후보 1곳 이상을 `candidates`에 넣음
- [ ] 새 폴더 clone → 설치 → 실행 절차를 README 그대로 따라 성공 (Windows 1회 이상)

---

### 담당 B — RAG 인프라 · 시장성 평가

**구현 목록**
1. **문서 확보 (가장 먼저):** 6개 PDF 다운로드 → `data/docs/tech/`, `data/docs/market/`에 명세서 5.1 파일명으로 저장. `docs.yaml`의 `TODO(verify)` 페이지 범위를 PDF를 열어 확정. **총 200쪽 이내 확인**. 기술 문서 3종의 파일은 C에게도 공유.
2. `rag/embeddings.py` — bge-m3, `local` / `hf_api` 백엔드
3. `rag/ingest.py` — 명세서 5.2 (페이지 범위 추출, 200쪽 초과 시 중단, 800토큰 청킹, 메타데이터, 세그먼트 불리언 플래그, `--rebuild`)
4. `rag/advanced_rag.py` — 명세서 5.4 (메타데이터 필터 → BM25 + 밀집 → RRF → `bge-reranker-v2-m3` → 문장 압축)
5. `agents/market_eval.py` + `prompts/market_eval.md` — 명세서 6.4 (고정 질문 4개, `market_cache`, 캐시 재사용 시 references 재발행)
6. `data/eval/qa_tech.jsonl`(20쌍), `qa_market.jsonl`(30쌍: 한→영 20 + 한→한 10) — `{"question","doc_id","page"}` 형식. **qa_tech는 C와 절반씩 작성.**
7. `eval/embedding_eval.py` — 명세서 5.5 (bge-m3 / Qwen3-Embedding-0.6B / multilingual-e5-large, Hit Rate@5, MRR → `outputs/embedding_eval.md`)

**단독 테스트:** `python -m rag.ingest` 후 `advanced_rag.search("미국 VPP 시장 전망", "US", "generation_vpp")`가 DOE VPP 문서 청크를 상위에 반환하는지 확인. `market_eval.run(sample_state)`가 `MarketAnalysis` 검증을 통과하는지 확인.

**완료 조건 (DoD)**
- [ ] `python -m rag.ingest` → 두 컬렉션 생성, 총 페이지 수 ≤ 200 출력
- [ ] `python -m eval.embedding_eval` → 3개 모델 비교표 출력, 결과를 README Tech Stack·설계서 4.3 표에 기입하도록 A에게 전달
- [ ] 시장성 결과의 모든 수치에 `source_id`와 기준 연도가 있음
- [ ] 같은 세그먼트의 두 번째 후보에서 캐시가 재사용되는 것을 log로 확인

---

### 담당 C — Agentic RAG · 기술 요약 · 경쟁사 비교

**구현 목록**
1. `rag/agentic_rag.py` — 명세서 5.3의 **LangGraph 서브그래프** (`retrieve → grade → rewrite/web_search → generate`, 재작성 최대 2회). 서브그래프를 `outputs/agentic_rag.mmd`로도 저장해 README에 쓸 수 있게 한다.
2. `agents/tech_summary.py` + `prompts/tech_summary.md` — 명세서 6.3 (서브그래프 질문 2개 + 팀 정보 웹 검색, DOE TRL 1~9 판단 근거)
3. `agents/competitor.py` + `prompts/competitor.md` — 명세서 6.5 (경쟁사 3~5곳, 동일 기준 비교, 진입장벽, 대상 기업 Traction)
4. `data/eval/qa_tech.jsonl` 중 10쌍 작성 (B와 분담)

**단독 테스트:** `agentic_rag.run("Amperon의 핵심 기술과 성능 지표는?", "Amperon", "Amperon")`이 `path`와 `rewrites`를 로그로 남기며 동작하는지 확인. 벡터스토어가 없을 때는 웹 경로만 타는지 확인. `tech_summary.run(sample_state)`, `competitor.run(sample_state)`가 스키마 검증을 통과하는지 확인.

**완료 조건 (DoD)**
- [ ] 서브그래프가 세 경로(관련 문서 충분 → 생성 / 재작성 후 재검색 / 웹 보완)를 모두 한 번 이상 타는 것을 로그로 확인
- [ ] `TechSummary.team`에 창업자 정보가 채워지거나, 없으면 `insufficient`에 "팀 정보" 기록
- [ ] 경쟁사 비교에 대상 기업 Traction 근거가 1개 이상 있거나 "공개 정보 부족" 명시
- [ ] 웹 근거마다 `make_web_ref`로 Reference 반환

---

### 담당 D — 투자 판단 · 보고서 · REFERENCE

**구현 목록**
1. `tools/references.py` — Phase 0에 `make_doc_ref`, `make_web_ref` 먼저 push → 이후 `format_reference`, 중복 제거, 기업 태그 필터, 유형별 그룹화 (명세서 8.4)
2. `config/rubric.yaml` — 명세서 7장 그대로 (로딩 시 가중치 합 100 assert)
3. `agents/investment_judge.py` + `prompts/investment_judge.md` — 명세서 6.6 (LLM은 점수·근거만, **합산·판정·knockout은 코드**, `EvalRecord` 생성)
4. `report/templates.py` — 경로 A/B 목차, 장별 입력 필드, 글자 예산 (명세서 8.1~8.3)
5. `agents/report_writer.py` + `prompts/report_*.md` — 장별 생성 → SUMMARY 마지막 생성 → 일관성 검사 → 5장 초과 시 예산 20% 축소 재생성
6. `report/pdf_renderer.py` — fpdf2 + NanumGothic, 판정 배지, SUMMARY 박스, 경쟁사 비교표·점수표·후보 비교표를 코드로 조판 (명세서 8.5)
7. `assets/fonts/NanumGothic-Regular.ttf`, `NanumGothic-Bold.ttf` 추가 (OFL 라이선스 파일 포함)
8. `tests/test_scoring.py`, `tests/test_references.py` — 명세서 9.3

**단독 테스트:** `sample_state.json`으로 `investment_judge.run()` → `report_writer.run()`을 이어 실행해 경로 A PDF 생성. `decision`을 "보류"로 바꾸고 `evaluation_history`에 가짜 기록 3개를 넣어 경로 B PDF 생성.

**완료 조건 (DoD)**
- [ ] 경로 A·B 모두 **5장 이내** PDF, 맨 앞 SUMMARY(550자 이내), 맨 끝 REFERENCE
- [ ] SUMMARY에 판정("투자"/"보류")과 총점이 포함됨 (코드 검사)
- [ ] REFERENCE가 기관 보고서 / 학술 논문 / 웹페이지 순서, 과제 표기 형식과 일치
- [ ] `pytest tests/test_scoring.py tests/test_references.py` 통과

---

## 4. 의존 관계

```
Phase 0 (A 스키마·도구, D 레퍼런스 헬퍼)
   │
   ├─▶ B: 문서 확보 → ingest → advanced_rag → market_eval → embedding_eval
   │                     │
   │                     └─▶ C: agentic_rag의 RAG 경로 연결 (웹 경로는 먼저 개발)
   ├─▶ C: agentic_rag(웹 경로) → tech_summary → competitor
   ├─▶ D: rubric → investment_judge → report_writer → pdf_renderer
   └─▶ A: discovery → select → graph(stub) → app
                                  │
            13:00 통합 ◀──────────┘ (B·C·D 노드를 stub 대신 연결)
```

---

## 5. 일정 (마감 DAY 3 15:00 역산 예시)

| 시간 | A | B | C | D |
|---|---|---|---|---|
| ~10:10 | Phase 0 (스키마·도구·뼈대) | PDF 다운로드, 페이지 범위 확정 | 서브그래프 골격 | 레퍼런스 헬퍼, rubric.yaml |
| 10:10~11:30 | discovery, select, stub 그래프 | ingest, embeddings | agentic_rag (웹 경로 우선) | investment_judge, test_scoring |
| 11:30~13:00 | routing 테스트, app.py, README 초안 | advanced_rag, market_eval, 평가셋 | tech_summary, competitor, RAG 경로 연결 | report_writer, pdf_renderer |
| **13:00 체크포인트** | **각자 브랜치를 PR로 `main`에 머지 (권장 순서: D → B → C → A)** | | | |
| 13:00~14:00 | E2E 실행·오류 수정 총괄 | embedding_eval 실행, 수치 전달 | E2E에서 분석 품질 점검 | 경로 A·B PDF 품질·분량 점검 |
| 14:00~14:30 | README 완성(수치·Contributors), clone 재현성 검증 | Hit@5·MRR 결과 README 반영 확인 | 발표용 차별점 정리 | 최종 보고서 PDF 확정 |
| 14:30~15:00 | **제출:** GitHub 링크 + `RAG-Output_{캠퍼스}-{X반}_{이름…}.pdf` Slack 스레드 업로드 | | | |

일정이 밀리면 줄이는 순서: ① 임베딩 비교 모델을 3개 → 2개 ② 재탐색 경로는 테스트로만 확인 ③ 경로 B PDF는 1회 생성 확인만.

---

## 6. 협업 규칙

- **브랜치:** `feat/<담당>-<기능>`. `main`에는 PR로만 반영하고, 4명 누구나 머지할 수 있다. 머지 전 `pytest` 통과 필수.
- **소유 파일 외 수정 금지.** 다른 사람 파일에 버그가 있으면 해당 담당자에게 알린다.
- **API 비용:** 개발 중에는 `--seed-only`와 검색 캐시를 기본으로 사용. 캐시 폴더(`outputs/cache/`)는 커밋하지 않는다.
- **키 관리:** `.env`는 커밋 금지. 키는 팀 채널로 공유.
- **설계서와 불일치 금지:** 에이전트 이름, State 필드명, 그래프 구조, 평가 항목을 바꿔야 하면 전원 합의 후 설계서와 README를 함께 수정한다.

---

## 7. README Contributors 예시 (PM·PL 역할 금지, 개인별 수행 역할만)

```
- {A 이름} : LangGraph 그래프·State 설계 구현, 스타트업 탐색 에이전트, 실행 스크립트
- {B 이름} : RAG 문서 인제스트, Advanced RAG(하이브리드 검색·리랭킹), 시장성 평가 에이전트, 임베딩 성능 평가
- {C 이름} : Agentic RAG 서브그래프, 기술 요약 에이전트, 경쟁사 비교 에이전트
- {D 이름} : 투자 판단 에이전트(평가 루브릭), 보고서 생성 에이전트, PDF 조판·REFERENCE 생성
```
