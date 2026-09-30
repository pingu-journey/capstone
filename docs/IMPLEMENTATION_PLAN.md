# Energy AI 스타트업 투자 평가 에이전트 — 구현 명세서

> **이 문서의 독자:** 이 저장소를 처음부터 구현할 코딩 에이전트(또는 개발자).
> **목표:** 아래 명세를 그대로 구현하면 `python app.py` 한 번으로 Energy AI 스타트업을 발굴·평가하고 5장 이내의 투자 보고서 PDF를 생성하는 LangGraph 기반 Multi-Agent + Agentic RAG 시스템이 완성된다.
> **원칙:** 이 명세는 이미 제출된 **설계 산출물(RAG-Design PDF)** 과 1:1로 대응한다. 평가 항목 "설계 구현 충실도(15점)"가 설계서와 코드의 일치 여부를 보므로, **에이전트 이름·State 필드명·그래프 구조·평가 기준을 임의로 바꾸지 말 것.** 명세에 없는 세부 구현은 자유롭게 결정하되, 결정 내용은 README에 한 줄로 남긴다.

---

## 0. 과제 요구사항 요약 (반드시 충족)

| 구분 | 요구사항 |
|---|---|
| 목표 | LangGraph 기반 Multi Agent + Agentic RAG 설계 및 개발, 목적에 맞는 도구(웹 검색·문서 요약 등) 정의 |
| 도메인 | **Energy** — AI로 전력 수요 예측, 발전/배터리 운영, 전력망 관리를 최적화하는 기술 |
| 스타트업 요건 | 비상장(코스피·코스닥 등 상장사 제외), 투자 단계 Seed ~ Series C, M&A·IPO 등 Exit 미완료 |
| RAG | "RAG 여부 O" 에이전트 중 최소 1개 RAG 적용. **문서 총 200페이지 이내** |
| 임베딩 | **오픈소스 임베딩 필수** |
| 그래프 | 순차 흐름(수집 → 분석 → 평가 → 보고서) + 보류 시 다른 스타트업으로 반복 + 모두 보류면 루프 종료 후 보고서 생성 |
| 보고서 | 5장 이내. 맨 앞 **SUMMARY**(1/2페이지 이내, 개요 장표 아님), 맨 끝 **REFERENCE**(실제 활용 자료만, 정해진 표기 형식) |
| README | 과제 샘플 형식 준수(10장 참고). Contributors에 PM/PL 역할 금지 |
| 재현성 | 새 환경에서 clone → 설치 → `python app.py` 실행 시 보고서 PDF가 실제로 생성되어야 함 |

### 개발 산출물 평가 기준 (구현 우선순위 판단에 사용)

| 항목 | 배점 | 이 명세에서 대응하는 절 |
|---|---|---|
| 설계 구현 충실도 | 15 | 전체 (특히 3, 4, 7장) |
| Agent 구현 | 15 | 6장 |
| **RAG Pipeline 구현** | **20** | 5장 |
| 코드 구조 및 프로젝트 구성 | 10 | 2장 |
| 실행 결과 재현성 | 10 | 9장 |
| **Output – 보고서** | **20** | 8장 |
| Output – README | 10 | 10장 |

---

## 1. 기술 스택

| 영역 | 선택 | 비고 |
|---|---|---|
| 언어 | Python 3.11 | Windows/macOS/Linux 모두 동작해야 함. 경로는 `pathlib` 사용 |
| 오케스트레이션 | `langgraph` | StateGraph, 조건부 엣지, fan-out/fan-in |
| LLM | `langchain-openai` (`ChatOpenAI`) | 모델명은 `.env`의 `LLM_MODEL`(생성용), `JUDGE_MODEL`(채점·관련성 평가용)로 주입. `temperature=0`. 코드에 모델명을 하드코딩하지 말 것 |
| 구조화 출력 | `pydantic` v2 + `llm.with_structured_output(Model)` | 모든 에이전트 출력은 Pydantic 모델로 검증 |
| 웹 검색 | `tavily-python` | `TAVILY_API_KEY`. 래퍼 `tools/web_search.py`로 감싸고 결과를 캐시 |
| PDF 로딩 | `pypdf` | 지정 페이지 범위만 로드 |
| 임베딩 | **`BAAI/bge-m3`** via `sentence-transformers` (로컬, 기본) | 옵션: HF Inference API (`EMBEDDING_BACKEND=hf_api`) |
| Vector DB | `chromadb` (PersistentClient, `data/vectorstore/`) | 컬렉션 2개: `tech_docs`, `market_docs` |
| 키워드 검색 | `rank-bm25` + `kiwipiepy`(한국어 형태소) | 시장성 하이브리드 검색용 |
| 리랭커 | `BAAI/bge-reranker-v2-m3` via `sentence_transformers.CrossEncoder` | 오픈소스, 한·영 지원 |
| 보고서 PDF | `fpdf2` + 한글 TTF 폰트(`assets/fonts/NanumGothic-Regular.ttf`, `-Bold.ttf`) | 순수 파이썬이라 Windows에서도 추가 설치 없이 동작. 폰트는 OFL 라이선스로 저장소에 포함 |
| 테스트 | `pytest` | 채점·라우팅·REFERENCE 포맷 단위 테스트 |

`requirements.txt`(버전은 설치 시점 최신 안정판으로 고정해 기록):
```
langgraph
langchain-core
langchain-openai
pydantic>=2
python-dotenv
tavily-python
pypdf
chromadb
sentence-transformers
rank-bm25
kiwipiepy
fpdf2
pyyaml
pytest
```

`.env.example`:
```
OPENAI_API_KEY=
TAVILY_API_KEY=
LLM_MODEL=            # 과정에서 제공된 GPT 모델명 기입 (생성용)
JUDGE_MODEL=          # 채점·관련성 평가용 (생성용과 같아도 됨)
EMBEDDING_BACKEND=local   # local | hf_api
HF_TOKEN=             # EMBEDDING_BACKEND=hf_api 일 때만
```

---

## 2. 디렉터리 구조

과제 README 샘플의 구조(`data/`, `agents/`, `prompts/`, `outputs/`, `app.py`)를 뼈대로 확장한다.

```
.
├── app.py                      # 실행 진입점: 그래프 빌드 → 실행 → 결과 저장
├── config.py                   # 상수(임계값, 가중치, 루프 상한 등) + .env 로딩
├── state.py                    # InvestState(TypedDict) + reducer 함수
├── schemas.py                  # Pydantic 하위 타입 (Candidate, TechSummary, ...)
├── graph/
│   ├── builder.py              # build_graph(): 노드·엣지 연결
│   └── routing.py              # 조건부 엣지 라우팅 함수
├── agents/
│   ├── discovery.py            # 스타트업 탐색 에이전트
│   ├── select_candidate.py     # 후보 선택 유틸 노드 (LLM 미사용)
│   ├── tech_summary.py         # 기술 요약 에이전트 (Agentic RAG)
│   ├── market_eval.py          # 시장성 평가 에이전트 (Advanced RAG)
│   ├── competitor.py           # 경쟁사 비교 에이전트
│   ├── investment_judge.py     # 투자 판단 에이전트
│   └── report_writer.py        # 보고서 생성 에이전트
├── rag/
│   ├── ingest.py               # PDF → 청크 → 임베딩 → Chroma 저장 (1회 실행)
│   ├── embeddings.py           # bge-m3 임베딩 래퍼 (local / hf_api)
│   ├── agentic_rag.py          # 기술 요약용 Agentic RAG 서브그래프
│   └── advanced_rag.py         # 시장성용 하이브리드 검색 + 리랭킹 + 압축
├── tools/
│   ├── web_search.py           # Tavily 래퍼 + 디스크 캐시
│   ├── llm.py                  # get_llm(), get_judge_llm()
│   └── references.py           # Reference 생성·중복 제거·REFERENCE 포맷터
├── report/
│   ├── pdf_renderer.py         # fpdf2 기반 PDF 조판
│   └── templates.py            # 경로 A/B 목차 정의, 장별 분량 예산
├── prompts/                    # 프롬프트 템플릿 (.md) — 6.x절 참고
├── config/
│   ├── docs.yaml               # RAG 문서 레지스트리 (5장)
│   └── rubric.yaml             # 11개 평가 항목·가중치·루브릭 (7장)
├── data/
│   ├── docs/tech/              # 기술 요약용 PDF 3종
│   ├── docs/market/            # 시장성용 PDF 3종
│   ├── seed_candidates.json    # 탐색 실패 시 대비용 후보 5곳
│   ├── eval/qa_tech.jsonl      # 임베딩 평가셋 (20쌍)
│   ├── eval/qa_market.jsonl    # 임베딩 평가셋 (30쌍)
│   └── vectorstore/            # Chroma 영속 저장소 (ingest 결과)
├── eval/
│   └── embedding_eval.py       # Hit Rate@5, MRR 비교 스크립트
├── assets/fonts/               # NanumGothic TTF
├── docs/graph.png              # 설계서의 그래프 이미지 (README에 삽입)
├── outputs/                    # 보고서 PDF, 실행 로그, 캐시
├── tests/
│   ├── test_scoring.py
│   ├── test_routing.py
│   └── test_references.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. State 설계 (설계서 6장과 동일해야 함)

### 3.1 하위 타입 — `schemas.py` (Pydantic v2)

State에는 `.model_dump()`한 dict를 저장한다. LLM 출력 검증은 이 모델로 한다.

```python
from typing import Literal
from pydantic import BaseModel, Field

Segment = Literal["demand_forecasting", "generation_vpp", "ess_operation", "grid_management"]
SEGMENT_KO = {
    "demand_forecasting": "전력 수요 예측",
    "generation_vpp": "발전 예측 · VPP",
    "ess_operation": "배터리(ESS) 운영",
    "grid_management": "전력망 관리",
}

class Candidate(BaseModel):
    name: str
    country: str                 # "KR", "US", ... (ISO 2자리)
    stage: str                   # "Seed" | "Series A" | "Series B" | "Series C"
    funding: str                 # 최근 라운드·누적 투자액 요약 (재무·투자조건 채점 근거)
    segment: Segment
    homepage: str | None = None
    priority: float = 0.0        # 유망도 (대기열 정렬 키, 높을수록 먼저)
    evidence_urls: list[str] = []

class TeamInfo(BaseModel):
    founders: list[str]          # "이름 — 역할, 주요 이력"
    key_members: list[str] = []
    assessment: str              # 팀 역량 요약 (2~3문장)

class TechSummary(BaseModel):
    core_tech: str               # 핵심 기술 (3~5문장)
    products: list[str]
    trl: int = Field(ge=1, le=9)
    trl_rationale: str
    performance: list[str]       # 정량 성능·고객가치 근거 (없으면 빈 리스트)
    strengths: list[str]
    weaknesses: list[str]
    scalability: str             # 소프트웨어/하드웨어 의존도, 확장성
    team: TeamInfo
    insufficient: list[str] = [] # 공개 정보 부족 항목

class MarketFigure(BaseModel):
    metric: str                  # "미국 VPP 용량 전망" 등
    value: str                   # "80~160GW"
    year: str                    # 기준 연도
    source_id: str               # Reference.id

class MarketAnalysis(BaseModel):
    segment: Segment
    country: str
    market_size: list[MarketFigure]
    growth: str                  # 성장성 요약 (CAGR 등)
    demand_drivers: list[str]
    customers: list[str]         # 주요 수요처
    policy_risks: list[str]      # 정책·규제 환경 및 리스크
    summary: str
    sources: list[dict] = []     # 캐시 재사용 시 references 재발행용 (Reference dict)

class Competitor(BaseModel):
    name: str
    country: str
    status: str                  # 투자 단계 또는 상장 여부
    offering: str
    vs_target: str               # 대상 기업 대비 차이

class CompetitorAnalysis(BaseModel):
    competitors: list[Competitor]        # 3~5곳
    differentiation: str                 # 대상 기업의 차별성
    moat: list[str]                      # 진입장벽: 특허·데이터·규제 인증·네트워크 효과
    traction: list[str]                  # 대상 기업의 고객·매출·계약·Pilot 근거
    competition_risks: list[str]

class ItemScore(BaseModel):
    item_id: str                 # rubric.yaml의 id
    score: int = Field(ge=0, le=10)
    weight: int                  # 코드가 채움 (LLM이 채우지 않음)
    rationale: str
    insufficient_info: bool = False

class EvalRecord(BaseModel):
    startup: str
    segment: Segment
    total_score: float
    decision: Literal["투자", "보류"]
    knockout: str | None = None
    weakest_items: list[str]     # 점수가 낮은 항목 2~3개
    key_reason: str

class Reference(BaseModel):
    id: str                      # url 또는 doc_id의 sha1 앞 10자리
    startup: str                 # 관련 기업명, 공통 문서는 "*"
    ref_type: Literal["report", "paper", "web"]
    author_org: str
    date: str                    # report/paper: "YYYY", web: "YYYY-MM-DD"
    title: str
    venue: str | None = None     # paper: 학술지명, web: 사이트명
    volume_issue: str | None = None   # paper: "12(9)"
    pages: str | None = None          # paper: "353"
    url: str | None = None
    used_by: str                 # 기록한 에이전트명
```

### 3.2 State — `state.py`

```python
from operator import add
from typing import Annotated, Literal, TypedDict

def merge_dict(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}

class InvestState(TypedDict, total=False):
    # 설정·공통
    domain: str                                   # "Energy"
    run_date: str                                 # "YYYY-MM-DD"
    log: Annotated[list[str], add]
    # 탐색
    candidates: list[dict]                        # list[Candidate] — 대기열
    search_round: int                             # 탐색 실행 횟수
    evaluated: Annotated[list[str], add]          # 평가 완료 기업명
    current_startup: dict | None                  # Candidate
    # 분석
    tech_summary: dict | None                     # TechSummary
    market_analysis: dict | None                  # MarketAnalysis
    market_cache: Annotated[dict[str, dict], merge_dict]
    competitor_analysis: dict | None              # CompetitorAnalysis
    # 판단
    scores: dict[str, dict]                       # item_id -> ItemScore
    total_score: float
    decision: Literal["투자", "보류"] | None
    evaluation_history: Annotated[list[dict], add]  # list[EvalRecord]
    # 보고서
    references: Annotated[list[dict], add]        # list[Reference]
    report_path: str | None
```

### 3.3 Writer / Reader (구현 시 이 표를 벗어나 State를 쓰지 말 것)

| 필드 | Writer | Reader |
|---|---|---|
| `domain`, `run_date` | app | 탐색, 기술 요약, 경쟁사 비교, 보고서 |
| `log` | 모든 노드 | app |
| `candidates` | 탐색(채움), 후보 선택(꺼냄) | 후보 선택, 라우터 |
| `search_round` | 탐색 | 라우터 |
| `evaluated` | 후보 선택 | 탐색 |
| `current_startup` | 후보 선택 | 기술, 시장, 경쟁사, 판단, 보고서 |
| `tech_summary` | 기술 요약 | 경쟁사, 판단, 보고서 |
| `market_analysis` | 시장성 평가 | 경쟁사, 판단, 보고서 |
| `market_cache` | 시장성 평가 | 시장성 평가 |
| `competitor_analysis` | 경쟁사 비교 | 판단, 보고서 |
| `scores`, `total_score`, `decision` | 투자 판단 | 라우터, 보고서 |
| `evaluation_history` | 투자 판단 | 보고서 |
| `references` | 탐색, 기술, 시장, 경쟁사 | 보고서 |
| `report_path` | 보고서 생성 | app |

**중요한 구현 규칙**
- 노드 함수는 **바꾼 키만 dict로 반환**한다 (State 전체를 반환하지 말 것).
- 병렬 노드(기술 요약, 시장성 평가)는 서로 다른 키에 쓰고, 공통으로 쓰는 `references`, `log`, `market_cache`는 reducer가 있으므로 충돌하지 않는다. reducer 없는 키에 두 병렬 노드가 동시에 쓰면 `InvalidUpdateError`가 난다.
- 후보 단위 필드(`tech_summary`, `market_analysis`, `competitor_analysis`, `scores`, `total_score`, `decision`)는 **후보 선택 노드가 None/빈 값으로 초기화**한다. 누적 필드(`evaluated`, `evaluation_history`, `references`, `log`, `market_cache`)는 실행 내내 누적한다.

---

## 4. Graph 설계 (설계서 7장과 동일해야 함)

### 4.1 노드와 엣지

```
START → discover
discover ──(route_after_discover)──▶ select | discover | report
select → tech_summary          ┐ 병렬 (fan-out)
select → market_eval           ┘
[tech_summary, market_eval] → competitor   (fan-in: 둘 다 끝나야 실행)
competitor → judge
judge ──(route_after_judge)──▶ report | select | discover
report → END
```

설계서 그래프의 마름모 "대기열에 후보가 있는가?"는 별도 노드가 아니라 **라우팅 함수**로 구현한다 (`route_after_judge`, `route_after_discover` 안의 대기열 검사).

### 4.2 `graph/builder.py`

```python
from langgraph.graph import StateGraph, START, END
from state import InvestState
from agents import discovery, select_candidate, tech_summary, market_eval, competitor, investment_judge, report_writer
from graph.routing import route_after_discover, route_after_judge

def build_graph():
    g = StateGraph(InvestState)
    g.add_node("discover", discovery.run)
    g.add_node("select", select_candidate.run)
    g.add_node("tech_summary", tech_summary.run)
    g.add_node("market_eval", market_eval.run)
    g.add_node("competitor", competitor.run)
    g.add_node("judge", investment_judge.run)
    g.add_node("report", report_writer.run)

    g.add_edge(START, "discover")
    g.add_conditional_edges("discover", route_after_discover,
                            {"select": "select", "discover": "discover", "report": "report"})
    g.add_edge("select", "tech_summary")
    g.add_edge("select", "market_eval")
    g.add_edge(["tech_summary", "market_eval"], "competitor")   # fan-in
    g.add_edge("competitor", "judge")
    g.add_conditional_edges("judge", route_after_judge,
                            {"report": "report", "select": "select", "discover": "discover"})
    g.add_edge("report", END)
    return g.compile()
```

### 4.3 `graph/routing.py`

```python
from config import MAX_DISCOVERY_ROUNDS, MAX_EVALUATIONS

def _queue_route(state) -> str:
    if len(state.get("evaluated", [])) >= MAX_EVALUATIONS:
        return "report"                       # 평가 상한 도달
    if state.get("candidates"):
        return "select"                       # Loop ① 다음 후보
    if state.get("search_round", 0) < MAX_DISCOVERY_ROUNDS:
        return "discover"                     # Loop ② 재탐색
    return "report"                           # 모두 보류 / 후보 없음

def route_after_judge(state) -> str:
    if state.get("decision") == "투자":
        return "report"
    return _queue_route(state)

def route_after_discover(state) -> str:
    return _queue_route(state)
```

### 4.4 루프 상한 (`config.py`)

| 상수 | 값 | 의미 |
|---|---|---|
| `MAX_DISCOVERY_ROUNDS` | 3 | 탐색 실행 총 횟수 = 최초 1회 + **재탐색 최대 2회** (설계서의 "재탐색 최대 2회") |
| `MAX_CANDIDATES_PER_ROUND` | 5 | 한 번의 탐색에서 대기열에 넣는 최대 후보 수 |
| `MAX_EVALUATIONS` | 8 | 전체 실행에서 심층 평가하는 최대 후보 수 (비용 상한) |
| `RAG_MAX_REWRITES` | 2 | Agentic RAG 쿼리 재작성 최대 횟수 (Loop ③) |
| `RECURSION_LIMIT` | 100 | `graph.invoke(..., config={"recursion_limit": 100})` 안전장치 |
| `INVEST_THRESHOLD` | 70 | 투자 판정 기준 총점 |

---

## 5. RAG 파이프라인 (평가 20점 — 가장 공들일 부분)

### 5.1 문서 레지스트리 — `config/docs.yaml`

PDF는 팀원이 아래 URL에서 받아 `data/docs/...` 경로에 저장한다 (저장소에 포함하되, 라이선스상 불가하면 README에 다운로드 안내). **`pages`는 PDF 뷰어 기준 물리 페이지 번호(1부터)** 로 적는다. 인쇄된 쪽번호와 다를 수 있으므로 `TODO(verify)` 항목은 파일을 열어 확인 후 채운다.

```yaml
tech:                          # 컬렉션 tech_docs, 총 120쪽
  - id: doe_ai_energy_2024
    file: data/docs/tech/DOE_AI_for_Energy_2024.pdf
    url: https://www.energy.gov/sites/default/files/2024-04/AI%20EO%20Report%20Section%205.2g%28i%29_043024.pdf
    ref_type: report
    author_org: U.S. Department of Energy
    date: "2024"
    title: "AI for Energy: Opportunities for a Modern Grid and Clean Energy Economy"
    pages: "1-59"
    lang: en
  - id: wef_ai_energy_2021
    file: data/docs/tech/WEF_Harnessing_AI_Energy_Transition_2021.pdf
    url: https://www3.weforum.org/docs/WEF_Harnessing_AI_to_accelerate_the_Energy_Transition_2021.pdf
    ref_type: report
    author_org: World Economic Forum
    date: "2021"
    title: "Harnessing Artificial Intelligence to Accelerate the Energy Transition"
    pages: "1-25"
    lang: en
  - id: mdpi_bess_review_2026
    file: data/docs/tech/MDPI_Batteries_AI_BESS_Review_2026.pdf
    url: https://www.mdpi.com/2313-0105/12/9/353
    ref_type: paper
    author_org: "Benbouhenni, H., Bizon, N."     # TODO(verify) 논문 첫 페이지의 저자 표기 확인
    date: "2026"
    title: "Artificial Intelligence-Enabled Battery Energy Storage Systems for Renewable Energy: A Comprehensive Review of Technologies, Applications, Challenges, and Future Directions"
    venue: Batteries
    volume_issue: "12(9)"
    article_pages: "353"
    pages: "TODO(verify)"        # 4~6장(AI 기법, AI 기반 BMS, 에너지 관리 전략)만, 약 36쪽
    lang: en

market:                        # 컬렉션 market_docs, 총 80쪽
  - id: iea_energy_ai_2025
    file: data/docs/market/IEA_Energy_and_AI_2025.pdf
    url: https://www.iea.org/reports/energy-and-ai
    ref_type: report
    author_org: IEA
    date: "2025"
    title: "Energy and AI"
    pages: "TODO(verify)"        # 3장 'AI for energy optimisation' 인쇄 쪽번호 p.109~133 (약 25쪽)
    country: GLOBAL
    segments: [demand_forecasting, generation_vpp, ess_operation, grid_management]
    lang: en
  - id: doe_vpp_liftoff_2025
    file: data/docs/market/DOE_VPP_Liftoff_2025_Update.pdf
    url: https://liftoff.energy.gov/wp-content/uploads/2025/01/LIFTOFF_DOE_VirtualPowerPlants2025Update.pdf
    ref_type: report
    author_org: U.S. Department of Energy
    date: "2025"
    title: "Pathways to Commercial Liftoff: Virtual Power Plants 2025 Update"
    pages: "TODO(verify)"        # 인쇄 쪽번호 p.1~15, p.34~46 (약 28쪽)
    country: US
    segments: [demand_forecasting, generation_vpp, ess_operation, grid_management]
    lang: en
  - id: posri_ai_power_demand_2024
    file: data/docs/market/POSRI_AI_Power_Demand_2024.pdf
    url: https://www.posri.re.kr/download.do?fid=4662&pid=8671
    ref_type: report
    author_org: 포스코경영연구원
    date: "2024"
    title: "AI발 전력수요 증가에 따른 에너지 및 소재 시장 동향 점검"
    pages: "TODO(verify)"        # 재생에너지·전력망 파트만 (핵심광물 파트 제외), 약 27쪽
    country: KR
    segments: [demand_forecasting, generation_vpp, grid_management]
    lang: ko
```

### 5.2 인제스트 — `rag/ingest.py` (`python -m rag.ingest`)

1. `docs.yaml`을 읽어 `pages` 범위의 페이지만 `pypdf`로 추출. **추출한 총 페이지 수를 출력하고 200을 넘으면 예외로 중단**한다.
2. 페이지 텍스트 정제(하이픈 줄바꿈 결합, 머리글·바닥글 반복 제거).
3. 청킹: 약 **800토큰(≈ 한글 1,200자 / 영문 3,000자), overlap 100토큰**. 페이지 경계를 넘는 청크는 시작 페이지를 `page`로 기록. 표로 보이는 블록(연속된 숫자·구분자 행)은 가능한 한 한 청크에 유지.
4. 메타데이터: `doc_id`, `source`(title), `year`, `page`, `lang`, 시장 문서는 `country`와 세그먼트 불리언 플래그 `seg_demand_forecasting`, `seg_generation_vpp`, `seg_ess_operation`, `seg_grid_management` (Chroma 메타데이터는 리스트를 못 담으므로 불리언으로 풀어 저장).
5. `rag/embeddings.py`의 bge-m3로 임베딩(정규화) 후 Chroma `tech_docs` / `market_docs` 컬렉션에 저장. 청크 ID는 `f"{doc_id}:{page}:{idx}"`.
6. 이미 컬렉션이 있으면 건너뛰고(`--rebuild` 플래그로 재생성) 재실행 비용을 줄인다.

`rag/embeddings.py`:
- `EMBEDDING_BACKEND=local`: `SentenceTransformer("BAAI/bge-m3")`, `normalize_embeddings=True`.
- `EMBEDDING_BACKEND=hf_api`: `huggingface_hub.InferenceClient(model="BAAI/bge-m3").feature_extraction(...)`.
- 두 백엔드 모두 `embed_documents(list[str])`, `embed_query(str)` 인터페이스를 제공.

### 5.3 기술 요약용 Agentic RAG — `rag/agentic_rag.py`

**LangGraph 서브그래프**로 구현한다 (Loop ③을 코드에서 보이게 하기 위함).

```
retrieve → grade ──(relevant ≥ 2)──────────────▶ generate → END
                 ├─(rewrite_count < 2)─▶ rewrite → retrieve
                 └─(else)──────────────▶ web_search → generate → END
```

서브그래프 State (`RagState`): `question`, `company`, `query`, `docs`, `relevant_docs`, `rewrite_count`, `web_results`, `answer`, `sources`.

| 노드 | 동작 |
|---|---|
| `retrieve` | `tech_docs`에서 한국어 `query`로 밀집 검색 Top-5 |
| `grade` | `JUDGE_MODEL`이 청크마다 "질문에 답하는 데 필요한 정보가 있는가"를 `yes/no`로 판정 (구조화 출력, 청크 5개를 한 번에 판정) |
| `rewrite` | 질문 의도를 유지하며 검색어를 재작성 (영문 기술 용어 병기 허용), `rewrite_count += 1` |
| `web_search` | `tools/web_search.py`로 기업 홈페이지·특허·논문·기사 검색 (쿼리: `"{company} technology"`, `"{company} 특허 OR patent"` 등 2~3개) |
| `generate` | 관련 청크 + 웹 결과만 근거로 답변 생성. 사용한 근거마다 `Reference` 생성 |

기술 요약 에이전트는 이 서브그래프를 **질문 2개**로 호출한다.
1. `"{company}의 핵심 기술, 제품, 성능 지표는 무엇인가?"` → 기업 고유 정보 (대부분 웹 보완 경로로 감)
2. `"{segment} 분야 AI 기술의 업계 수준과 상용화 단계(TRL 판단 근거)는?"` → RAG 문서가 업계 기준선을 제공

팀 정보는 RAG 문서에 없으므로 서브그래프를 쓰지 않고 직접 웹 검색한다 (6.3절).

### 5.4 시장성용 Advanced RAG — `rag/advanced_rag.py`

```
[검색 전] 메타데이터 필터 → 하이브리드 검색(BM25 + 밀집, RRF) → [검색 후] 리랭킹 → 컨텍스트 압축
```

1. **메타데이터 필터:** `where={"$and": [{"country": {"$in": [country, "GLOBAL"]}}, {f"seg_{segment}": True}]}`. 결과가 5개 미만이면 세그먼트 조건을 빼고 국가 조건만으로 재검색.
2. **밀집 검색:** Chroma Top-20.
3. **BM25 검색:** 같은 필터를 적용한 청크 집합으로 BM25 인덱스를 만들어 Top-20. 토크나이저: 한국어는 `kiwipiepy` 명사·동사 어간, 영문은 소문자 단어 토큰. 인덱스는 컬렉션 로드 시 1회 생성 후 메모리 캐시.
4. **RRF 결합:** `score = Σ 1/(60 + rank)`, 상위 20개.
5. **리랭킹:** `CrossEncoder("BAAI/bge-reranker-v2-m3")`로 (질문, 청크) 점수 계산 → 상위 5개.
6. **컨텍스트 압축:** 청크를 문장 단위로 나눠 숫자(%, GW, 달러, 원, 연도)를 포함하거나 질문 키워드와 겹치는 문장만 남기고 청크당 최대 1,200자. LLM 미사용(재현성).
7. 반환: 압축된 컨텍스트 목록 + 각 청크의 `doc_id`, `page`.

시장성 에이전트는 아래 **고정 질문 4개**를 각각 검색한다 (질문이 정형적이라는 설계 근거와 일치).
- `"{segment_ko} 시장 규모와 전망 수치"`
- `"{segment_ko} 시장 성장률과 성장 요인"`
- `"{segment_ko} 수요처와 도입 수요"`
- `"{country_ko} {segment_ko} 관련 정책·규제 환경과 리스크"`

### 5.5 임베딩 평가 — `eval/embedding_eval.py` (`python -m eval.embedding_eval`)

설계서 4.3의 검증 계획을 구현한다.

- 평가셋 형식 (`data/eval/qa_tech.jsonl` 20줄, `qa_market.jsonl` 30줄 — 한→영 20 + 한→한 10):
  ```json
  {"question": "미국 VPP 용량은 2030년에 얼마로 전망되나?", "doc_id": "doe_vpp_liftoff_2025", "page": 3}
  ```
  정답을 **(doc_id, page)** 로 지정하므로 청킹 방식이나 모델이 바뀌어도 평가셋을 재사용할 수 있다. 검색된 청크의 `doc_id`가 같고 `page`가 정답 페이지와 같거나 청크가 그 페이지를 포함하면 적중으로 본다.
- 질문은 팀원이 실제 문서를 읽고 작성한다 (LLM 초안 생성 가능, 사람이 검수). 수치·표를 묻는 질문을 반드시 포함.
- 비교 모델 3종 (모델별 쿼리 접두사 규칙 준수):
  | 모델 | 쿼리 처리 | 문서 처리 |
  |---|---|---|
  | `BAAI/bge-m3` | 그대로 | 그대로 |
  | `Qwen/Qwen3-Embedding-0.6B` | `prompt_name="query"` | 그대로 |
  | `intfloat/multilingual-e5-large` | `"query: "` 접두사 | `"passage: "` 접두사, 512토큰 초과분 잘림 주의 |
- 모델마다 임시 인메모리 Chroma 컬렉션을 만들어 **동일 청크·동일 질의**로 Top-5 검색.
- 지표: **Hit Rate@5**, **MRR** (정답 첫 등장 순위의 역수 평균, 없으면 0).
- 결과를 표로 출력하고 `outputs/embedding_eval.md`로 저장. 이 값을 README Tech Stack과 설계서 4.3 표에 기입한다.

---

## 6. 에이전트 명세

공통 규칙
- 각 에이전트는 `run(state: InvestState) -> dict` 함수 하나를 노출한다.
- LLM 호출은 `tools/llm.py`의 `get_llm()` / `get_judge_llm()`만 사용하고 `with_structured_output(...)`으로 Pydantic 검증. 검증 실패 시 1회 재시도 후 실패하면 해당 필드를 `insufficient`에 기록하고 진행 (그래프를 멈추지 않는다).
- 프롬프트는 `prompts/*.md`에 두고 `str.format`으로 채운다. 모든 프롬프트에 "제공된 근거에 없는 사실은 쓰지 말고, 모르면 '공개 정보 부족'이라고 적어라"를 포함한다.
- 각 노드는 `log`에 `"[node] 요약 한 줄"`을 추가한다.
- 웹 검색 결과로 사실을 적을 때는 반드시 `Reference(ref_type="web")`를 함께 반환한다.

### 6.1 스타트업 탐색 — `agents/discovery.py` (RAG X, 웹서치)

**입력:** `domain`, `run_date`, `evaluated`, `search_round` **출력:** `candidates`, `search_round`, `references`, `log`

1. **발굴 검색:** 고정 쿼리 템플릿(재현성). 세그먼트 4개 × 언어 2개 = 8개 쿼리를 `search_round`에 따라 순서를 돌려 사용 (재탐색 시 다른 표현 사용).
   ```python
   QUERY_TEMPLATES = {
     "demand_forecasting": ["AI 전력 수요 예측 스타트업 투자 유치", "AI electricity demand forecasting startup raises Series"],
     "generation_vpp":     ["재생에너지 발전량 예측 VPP 스타트업 투자", "virtual power plant AI startup funding round"],
     "ess_operation":      ["ESS 배터리 운영 최적화 AI 스타트업 투자", "battery storage optimization AI startup Series A"],
     "grid_management":    ["AI 전력망 관리 스타트업 투자 유치", "AI grid management software startup raises"],
   }
   ```
2. **후보 추출:** LLM이 검색 결과에서 `Candidate` 목록 추출 (이름, 국가, 단계, 투자 요약, 세그먼트, 홈페이지, 근거 URL).
3. **요건 검증:** 후보마다 `"{name} IPO OR 상장 OR acquired OR 인수"` 검색 후 LLM이 `listed`, `exited`, `stage ∈ {Seed..Series C}`를 판정. 불충족·불확실하면 제외. `evaluated`에 있는 기업도 제외.
4. **유망도 정렬:** `priority` = LLM이 1~5로 매긴 "도메인 적합성 + 공개 정보 충분성" 점수. 내림차순 정렬 후 상위 `MAX_CANDIDATES_PER_ROUND`개를 `candidates`로 반환.
5. **대비책:** 결과가 0곳이면 `data/seed_candidates.json`에서 `evaluated`에 없는 기업을 로드 (log에 "fallback" 명시).
6. `search_round`를 1 증가시켜 반환. 발굴 결과를 `outputs/discovery_round{n}.json`에 저장.

`data/seed_candidates.json` (요건 충족을 2026-09 기준으로 확인한 5곳):
```json
[
  {"name": "해줌", "country": "KR", "stage": "Series B", "funding": "Series B 투자 유치", "segment": "generation_vpp", "homepage": "https://www.haezoom.com"},
  {"name": "식스티헤르츠", "country": "KR", "stage": "Seed", "funding": "Seed 투자 유치 (공개 정보 기준, 이후 라운드 확인 필요)", "segment": "generation_vpp", "homepage": "https://60hz.io"},
  {"name": "Amperon", "country": "US", "stage": "Series B", "funding": "Series B $20M (Energize Capital 주도), 2025 National Grid Partners 전략 투자", "segment": "demand_forecasting", "homepage": "https://www.amperon.co"},
  {"name": "Tyba", "country": "US", "stage": "Series A", "funding": "Series A $13.9M (2025)", "segment": "ess_operation", "homepage": "https://www.tyba.ai"},
  {"name": "Camus Energy", "country": "US", "stage": "Series A", "funding": "Series A 누적 $25M+ (2024 확장)", "segment": "grid_management", "homepage": "https://www.camus.energy"}
]
```
(홈페이지 URL은 구현 시 한 번 접속해 확인할 것.)

### 6.2 후보 선택 — `agents/select_candidate.py` (LLM 미사용 유틸 노드)

```python
def run(state):
    queue = list(state["candidates"])
    current = queue.pop(0)
    return {
        "candidates": queue,
        "current_startup": current,
        "evaluated": [current["name"]],
        "tech_summary": None, "market_analysis": None, "competitor_analysis": None,
        "scores": {}, "total_score": 0.0, "decision": None,
        "log": [f"[select] {current['name']} 평가 시작 (남은 후보 {len(queue)})"],
    }
```

### 6.3 기술 요약 — `agents/tech_summary.py` (RAG O · Agentic RAG)

**입력:** `current_startup`, `run_date` **출력:** `tech_summary`, `references`, `log`

1. 5.3의 Agentic RAG 서브그래프를 질문 2개로 호출.
2. 팀 정보: `"{name} founder CEO CTO"`, `"{name} 창업자 대표 이력"` 웹 검색 → 창업자·핵심 인력·이력 추출.
3. LLM(`prompts/tech_summary.md`)이 1~2의 결과를 종합해 `TechSummary` 생성. **TRL은 DOE TRL 정의(1~9)로 판단하고 근거를 `trl_rationale`에 기록.**
4. 서브그래프와 웹 검색에서 실제 사용된 근거를 `Reference`로 반환 (`startup=현재 기업명`, `used_by="tech_summary"`).

### 6.4 시장성 평가 — `agents/market_eval.py` (RAG O · Advanced RAG)

**입력:** `current_startup`, `market_cache` **출력:** `market_analysis`, `market_cache`, `references`, `log`

1. 캐시 키 `f"{country}|{segment}"`. 캐시에 있으면 재사용하되, `sources`의 Reference들을 `startup=현재 기업명`으로 복사해 `references`로 다시 반환 (REFERENCE에 누락되지 않도록).
2. 캐시에 없으면 5.4의 고정 질문 4개로 Advanced RAG 검색 → LLM(`prompts/market_eval.md`)이 `MarketAnalysis` 생성. **모든 수치에는 `source_id`와 기준 연도 필수.**
3. 국내 기업(KR)이면 포스코경영연구원 문서, 미국 기업(US)이면 DOE VPP 문서가 필터로 우선 검색된다 (IEA는 GLOBAL로 공통).
4. `market_cache`에 `{key: analysis}`로 저장.

### 6.5 경쟁사 비교 — `agents/competitor.py` (RAG X, 웹서치)

**입력:** `current_startup`, `tech_summary`, `market_analysis`, `run_date` **출력:** `competitor_analysis`, `references`, `log`

1. 검색 쿼리: `"{name} competitors"`, `"{segment_en} startups {country}"`, `"{segment_ko} 기업"`, `"{name} 고객 OR 계약 OR 파트너십 OR customers"`.
2. 같은 세그먼트·지역의 경쟁사 3~5곳 선정 (상장 대기업 포함 가능, 다른 후보 기업도 경쟁사가 될 수 있음).
3. LLM(`prompts/competitor.md`)이 `tech_summary`와 `market_analysis`를 함께 보고, 동일 기준(제공 기술, 투자 단계/상장 여부, 대상 대비 차이)으로 비교표·차별성·진입장벽·대상 기업의 Traction(고객·계약·Pilot)을 작성.

### 6.5.1 기술 요약·경쟁사 비교 출력 규칙 (투자 판단·보고서 참고)

- `insufficient`에는 공개 정보가 부족한 항목의 **한글 이름**을 코드가 빈 필드를 보고 기록한다. 루브릭 id 대응표는 `agents.tech_summary.INSUFFICIENT_TO_RUBRIC`, `agents.competitor.INSUFFICIENT_TO_RUBRIC`이다.
  - 기술 요약: 팀 정보, TRL, 성능 지표, 확장성, 핵심 기술, 제품, 강점·약점
  - 경쟁사 비교: Traction, 진입장벽, 경쟁사 비교
- `competitor_analysis`에는 `insufficient` 키가 추가로 들어 있다 (`CompetitorAnalysis(**dict)`로 읽으면 무시된다).
- 기술 요약 생성에 실패하면 `trl=1`, `trl_rationale="공개 정보 부족"`으로 채우고 `insufficient`에 "TRL"을 기록한다. 이때 투자 판단은 trl 항목을 `insufficient_info=True`, 5점으로 처리한다.
- 근거가 없는 리스트 항목은 제거되므로 `products`, `traction`, `moat` 등은 빈 리스트일 수 있고, 경쟁사 `status`는 "공개 정보 부족"일 수 있다. 보고서는 빈 리스트를 "공개 정보 부족"으로 표기한다.
- Reference `used_by`: 기술 요약 `"tech_summary"`, 경쟁사 비교 `"competitor"`.

### 6.6 투자 판단 — `agents/investment_judge.py` (RAG X, LLM 채점 + 코드 판정)

**입력:** `current_startup`, `tech_summary`, `market_analysis`, `competitor_analysis` **출력:** `scores`, `total_score`, `decision`, `evaluation_history`, `log`

1. `JUDGE_MODEL`이 `config/rubric.yaml`의 11개 항목 루브릭을 받아 항목별 `score(0~10)`, `rationale`, `insufficient_info`만 작성 (가중치는 LLM에 주지 않아 점수 부풀림을 막는다).
2. **코드가 합산·판정** (LLM이 판정하지 않음):
   ```python
   def compute(scores: dict, rubric: list) -> tuple[float, str, str | None]:
       total = sum(scores[i["id"]]["score"] * i["weight"] / 10 for i in rubric)
       knockout = next((i["name"] for i in rubric
                        if i.get("knockout") and scores[i["id"]]["score"] <= 3), None)
       decision = "투자" if total >= INVEST_THRESHOLD and knockout is None else "보류"
       return round(total, 1), decision, knockout
   ```
3. 공개 정보가 없어 판단 불가한 항목은 LLM이 `insufficient_info=True`, `score=5`로 적도록 프롬프트에 명시.
4. `EvalRecord`를 만들어 `evaluation_history`에 추가 (가중 점수 기준 하위 2~3개 항목을 `weakest_items`로).

### 6.7 보고서 생성 — `agents/report_writer.py` (RAG X)

8장 참고. `decision == "투자"`면 경로 A, 아니면 경로 B.

---

## 7. 투자 판단 기준 — `config/rubric.yaml`

설계서 5장과 동일. 가중치 합계 100, 판정 기준 70점, 필수 탈락 조건 2개.

```yaml
threshold: 70
items:
  - {id: team, name: 창업팀 역량, weight: 25, knockout: true, basis: VC 투자기준, input: tech_summary.team,
     high: "에너지·AI 창업/Exit 경험, 핵심 인력 완비", mid: "관련 경력은 있으나 공백 존재", low: "도메인 경험 부족"}
  - {id: market_size, name: 시장 규모·성장성, weight: 12, basis: DOE ARL, input: market_analysis,
     high: "대형 시장, 고성장(CAGR 20% 이상)", mid: "중간 규모 또는 성장 둔화", low: "틈새·정체 시장"}
  - {id: market_demand, name: 시장 수요, weight: 10, basis: DOE ARL, input: market_analysis,
     high: "유틸리티·발전사 등 명확한 수요 확인", mid: "수요는 있으나 구매 의사 불명확", low: "수요 근거 없음"}
  - {id: traction, name: 사업모델·Traction, weight: 10, basis: VC 투자기준, input: competitor_analysis.traction,
     high: "매출·상용 계약 확보", mid: "Pilot·MOU 단계", low: "실적 없음"}
  - {id: moat, name: 경쟁우위, weight: 8, basis: DOE ARL, input: competitor_analysis,
     high: "특허·데이터 등 진입장벽 보유", mid: "차별점은 있으나 모방 가능", low: "차별성 없음"}
  - {id: trl, name: 기술 성숙도, weight: 7, basis: DOE TRL, input: tech_summary.trl,
     high: "TRL 7 이상 (상용화·실증 완료)", mid: "TRL 4~6 (실증 중)", low: "TRL 3 이하 (PoC 이전)"}
  - {id: tech_value, name: 기술 성능·고객가치, weight: 7, basis: DOE ARL, input: tech_summary.performance,
     high: "비용·효율 개선이 정량으로 입증됨", mid: "정성적 개선", low: "기존 대비 이점 불명확"}
  - {id: regulation, name: 규제·정책 위험, weight: 6, knockout: true, basis: DOE ARL, input: market_analysis.policy_risks,
     high: "제도가 우호적, 인허가 불필요", mid: "제도 변화에 일부 의존", low: "제도 부재·중대 장벽"}
  - {id: scalability, name: 공급·확장 가능성, weight: 5, basis: DOE ARL, input: tech_summary.scalability,
     high: "소프트웨어 중심, 해외 확장 용이", mid: "하드웨어·현장 설치에 일부 의존", low: "인프라 제약이 큼"}
  - {id: finance, name: 재무·투자 가능성, weight: 5, basis: VC 투자기준, input: current_startup.funding,
     high: "자금 여력 충분, 후속 투자 유치", mid: "1년 내 추가 조달 필요", low: "자금 고갈 위험"}
  - {id: deal_terms, name: 투자조건, weight: 5, basis: VC 투자기준, input: current_startup.funding,
     high: "밸류에이션 적정, 지분 확보 가능", mid: "다소 고평가", low: "과도한 고평가"}
```
루브릭 구간: 상 = 8~10점, 중 = 4~7점, 하 = 0~3점. `knockout: true` 항목이 0~3점이면 총점과 관계없이 보류.
로딩 시 가중치 합이 100인지 assert.

---

## 8. 보고서 생성 (평가 20점)

### 8.1 목차 (설계서 8장과 동일)

**경로 A — 투자 추천 시**
```
SUMMARY
  - 투자 결론(투자 / 총점) 및 핵심 근거
  - 핵심 투자 포인트 3 · 핵심 리스크 2
1. 기업 및 사업 개요
  - 사업 아이디어(핵심 컨셉) / 핵심 제품·서비스 및 기술(TRL) / 비즈니스 모델 및 투자 단계
2. 시장 및 경쟁 분석
  - 시장 규모 및 성장성 / 고객 수요 및 정책·제도 환경 / 주요 경쟁사 비교 및 차별성
3. 기업 역량 및 리스크
  - 핵심 창업자 및 팀 역량 / 주요 사업 성과 / 기술·시장·규제·경쟁 리스크
4. 종합 투자 평가
  - 평가 항목별 점수 및 총점 / 최종 투자 판단 및 근거 / 분석의 한계
REFERENCE
```

**경로 B — 모두 보류 시**
```
SUMMARY
  - 투자 대상 없음, 평가 후보 수, 공통 보류 사유
1. 평가 대상 및 선정 과정   - 탐색 조건(도메인·스타트업 요건), 후보 목록
2. 후보별 평가 결과 비교    - 후보별 총점·판정·취약 항목 비교표
3. 주요 보류 사유 및 시장 환경 - 공통 리스크, 시장·규제 환경
4. 분석의 한계
REFERENCE
```

### 8.2 장별 입력 State (프롬프트에 해당 필드만 넣는다)

| 장 (경로 A) | 입력 |
|---|---|
| SUMMARY | `decision`, `total_score`, `scores`(상위 3·하위 2 항목), 각 장 생성 결과 요약 |
| 1 | `current_startup`, `tech_summary` (팀 제외) |
| 2 | `market_analysis`, `competitor_analysis` (비교표는 코드가 표로 조판) |
| 3 | `tech_summary.team`, `competitor_analysis.traction`, `tech_summary.weaknesses`, `market_analysis.policy_risks`, `competitor_analysis.competition_risks` |
| 4 | `scores`, `total_score`, `decision`, `insufficient_info` 항목 목록 (점수표는 코드가 조판) |
| REFERENCE | `references` 중 `startup ∈ {현재 기업명, "*"}` (경로 B는 전체) |

| 장 (경로 B) | 입력 |
|---|---|
| SUMMARY | `evaluation_history` |
| 1 | `domain`, 스타트업 요건, `evaluated`, `search_round` |
| 2 | `evaluation_history` (비교표는 코드가 조판) |
| 3 | `evaluation_history`의 `weakest_items`·`key_reason`, `market_cache` |
| 4 | 공개 정보 한계, `insufficient_info` 항목, `run_date` |

### 8.3 생성 절차

1. **SUMMARY를 마지막에 생성**: 1~4장을 먼저 생성한 뒤, 그 결과와 판정 필드로 SUMMARY 작성. SUMMARY는 **550자 이내**(1/2페이지), 개요 나열 금지, 첫 문장에 결론(투자/보류와 총점).
2. 장별 LLM 호출은 구조화 출력 `{"blocks": [{"subtitle": str, "text": str}]}` 형태. 장별 글자 예산(공백 포함): 1장 900자, 2장 1,100자, 3장 1,100자, 4장 700자.
3. 표는 LLM이 아니라 **코드가 조판**: 경쟁사 비교표(2장), 11개 항목 점수표(4장: 항목·가중치·점수·가중 점수·근거 요약), 후보 비교표(경로 B 2장).
4. **일관성 검사:** SUMMARY 텍스트에 `decision` 문자열("투자" 또는 "보류")과 `total_score`가 포함됐는지 코드로 확인, 없으면 1회 재생성.
5. **분량 검사:** PDF 렌더 후 페이지 수가 5를 넘으면 모든 장 글자 예산을 20% 줄여 재생성 (최대 2회). 그래도 넘으면 폰트 크기를 한 단계 줄인다.
6. 결과: `outputs/report_{YYYYMMDD_HHMM}.pdf`, 같은 이름의 `.md`(검토용)도 함께 저장. `report_path` 반환.

### 8.4 REFERENCE 포맷 — `tools/references.py`

과제 표기 형식을 그대로 따른다. **LLM이 아니라 코드가 생성**하며, 보고서에 실제로 전달된 근거만 포함한다 (`url` 또는 `doc_id`로 중복 제거).

```python
def format_reference(r: Reference) -> str:
    if r.ref_type == "report":   # 발행기관(YYYY). 보고서명. URL
        return f"{r.author_org}({r.date}). {r.title}. {r.url or ''}".strip()
    if r.ref_type == "paper":    # 저자(YYYY). 논문제목. 학술지명, 권(호), 페이지.
        return f"{r.author_org}({r.date}). {r.title}. {r.venue}, {r.volume_issue}, {r.pages}."
    # web: 기관명 또는 작성자(YYYY-MM-DD). 제목. 사이트명, URL
    return f"{r.author_org}({r.date}). {r.title}. {r.venue}, {r.url}"
```
- REFERENCE 장은 **기관 보고서 → 학술 논문 → 웹페이지** 소제목으로 묶고 각 그룹 안에서 날짜 내림차순.
- 웹 결과의 게시일을 알 수 없으면 `run_date`를 쓴다 (조회일). `author_org`를 알 수 없으면 사이트명을 쓴다.

### 8.5 PDF 조판 — `report/pdf_renderer.py`

- `fpdf2`, A4, 여백 15mm, `NanumGothic` 등록(Regular/Bold). 본문 9.5pt, 제목 14pt, 표 8pt.
- 첫 페이지 상단: 제목 `"{기업명} 투자 평가 보고서"`(경로 B는 `"Energy AI 스타트업 투자 평가 결과"`), 작성일, 판정 배지(투자/보류 + 총점).
- SUMMARY는 음영 박스로 구분.
- 경로 A 4장 점수표: 11행 + 합계 행, 가중치 합 100.

---

## 9. 실행 흐름과 재현성

### 9.1 `app.py`

```python
from datetime import date
from dotenv import load_dotenv
from graph.builder import build_graph
from config import RECURSION_LIMIT

def main():
    load_dotenv()
    graph = build_graph()
    init = {"domain": "Energy", "run_date": date.today().isoformat(),
            "candidates": [], "search_round": 0, "evaluated": [], "market_cache": {},
            "evaluation_history": [], "references": [], "log": []}
    final = graph.invoke(init, config={"recursion_limit": RECURSION_LIMIT})
    # outputs/run_log.json: evaluation_history, scores, decision, references, log 저장
    print(f"보고서: {final['report_path']}")

if __name__ == "__main__":
    main()
```
- 실행 전 벡터스토어가 없으면 `rag.ingest`를 자동 호출하거나, 명확한 오류 메시지로 `python -m rag.ingest` 실행을 안내.
- `--seed-only` 옵션: 웹 발굴 없이 `seed_candidates.json`만으로 실행 (API 한도·시연용).
- 그래프 구조를 `outputs/graph.mmd`(`graph.get_graph().draw_mermaid()`)로 저장.

### 9.2 재현성 체크리스트 (구현 완료 조건)

- [ ] `temperature=0`, 모델명은 `.env`로만 주입
- [ ] 웹 검색 결과를 `outputs/cache/search/{sha1(query)}.json`에 캐시하고, 캐시가 있으면 재사용 (`--no-cache`로 무시)
- [ ] 절대경로 금지, `pathlib`로 저장소 루트 기준 경로 사용
- [ ] 새 폴더에 clone → `pip install -r requirements.txt` → `.env` 작성 → `python -m rag.ingest` → `python app.py`로 PDF 생성 확인 (Windows에서 1회 이상 검증)
- [ ] `pytest` 통과

### 9.3 단위 테스트 (`tests/`)

| 테스트 | 검증 내용 |
|---|---|
| `test_scoring.py` | 가중치 합 100, 총점 계산, 70점 경계(69.9 보류 / 70.0 투자), knockout 시 보류 |
| `test_routing.py` | 투자 → report / 보류+후보 남음 → select / 대기열 빔+round<3 → discover / round=3 → report / 평가 상한 → report |
| `test_references.py` | 3개 유형 포맷이 과제 예시와 일치, 중복 URL 제거, 기업 태그 필터 |

---

## 10. README.md (과제 샘플 형식 — 제목과 섹션 순서 유지)

~~~markdown
# AI Startup Investment Evaluation Agent
본 프로젝트는 Energy(AI 기반 전력 수요 예측·발전/배터리 운영·전력망 관리) 스타트업에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다.

## Overview
- Objective : AI 스타트업의 기술력, 시장성, 경쟁력, 팀 역량 등을 기준으로 투자 적합성 분석
- Method : LangGraph Multi-Agent, Agentic RAG(기술 요약), Advanced RAG(시장성 평가)

## Features
- 웹 검색 기반 스타트업 자동 발굴 및 요건(비상장·Seed~Series C·Exit 없음) 검증
- PDF 자료 기반 정보 추출 (기술 문서 120쪽 + 시장 보고서 80쪽)
- 기술 요약과 시장성 평가 병렬 실행, 경쟁사 비교에서 합류
- 11개 항목 루브릭 채점 + 코드 기반 투자/보류 판정
- 보류 시 다음 후보 평가, 대기열 소진 시 재탐색(최대 2회)
- SUMMARY·REFERENCE를 갖춘 5장 이내 투자 보고서 PDF 자동 생성

## Tech Stack
- Framework : LangGraph
- LLM/Generator : {LLM_MODEL}
- LLM/Judge : {JUDGE_MODEL}
- Retrieval : Chroma - Hit Rate@5 {값}, MRR {값}   ← eval/embedding_eval.py 결과
- Embedding : BAAI/bge-m3 (오픈소스)
- Reranker : BAAI/bge-reranker-v2-m3

## Agents
- 스타트업 탐색 : 웹 검색으로 후보 발굴·요건 필터·유망도 정렬
- 기술 요약 : Agentic RAG로 기술력 핵심 요약(TRL, 장단점) 및 팀 구성 확인
- 시장성 평가 : Advanced RAG로 세그먼트 시장 규모·성장성·수요·정책 분석
- 경쟁사 비교 : 경쟁사 3~5곳 비교, 차별성·진입장벽 분석
- 투자 판단 : 11개 항목 채점, 코드로 합산·판정(70점, 필수 탈락 조건)
- 보고서 생성 : 판정 경로별 보고서 작성 및 PDF 저장

## Architecture
![graph](docs/graph.png)

## Directory Structure
(2장 트리를 요약해 기재)

## Usage
```bash
pip install -r requirements.txt
cp .env.example .env        # 키 입력
python -m rag.ingest        # 최초 1회
python app.py
```

## Contributors
- {이름} : {담당 역할}      ← 개인별 수행 역할만. PM, PL 역할은 쓰지 않는다
~~~

---

## 11. 구현 순서 (권장 마일스톤)

| 단계 | 작업 | 완료 조건 |
|---|---|---|
| M0 | 저장소 뼈대, `config.py`, `schemas.py`, `state.py`, `tools/llm.py`, `.env.example` | import 오류 없음 |
| M1 | `docs.yaml` 페이지 확정(TODO(verify) 해소), `rag/ingest.py` | 두 컬렉션 생성, 총 페이지 ≤ 200 출력 |
| M2 | `advanced_rag.py`, `agentic_rag.py` 서브그래프, `embedding_eval.py` | 평가표(Hit@5, MRR) 출력 |
| M3 | 에이전트 7개 노드 (탐색 → 선택 → 기술 ∥ 시장 → 경쟁사 → 판단) | 시드 후보 1곳으로 노드 단위 실행 성공 |
| M4 | `graph/builder.py`, `routing.py`, 단위 테스트 | 보류 루프·재탐색·모두 보류 경로를 테스트로 확인 |
| M5 | 보고서 생성, REFERENCE 포맷터, PDF 렌더러 | 경로 A·B 모두 5장 이내 PDF 생성 |
| M6 | README, 재현성 체크리스트 | 새 환경 clone 후 `python app.py` 성공 |

경로 B(모두 보류) 테스트 방법: `INVEST_THRESHOLD=101`로 임시 설정하거나 `--seed-only`로 실행해 강제로 확인.

---

## 12. 하지 말 것

- 에이전트를 추가·병합하거나 이름을 바꾸지 말 것 (설계서와 불일치 → 설계 구현 충실도 감점).
- 투자/보류 판정을 LLM에게 맡기지 말 것 (합산·판정은 반드시 코드).
- REFERENCE를 LLM이 작성하게 하지 말 것.
- 임베딩에 유료 API 모델(OpenAI embeddings 등)을 쓰지 말 것.
- RAG 문서를 200페이지 넘게 넣지 말 것 (인제스트에서 강제 검사).
- 근거 없는 수치·사실을 보고서에 쓰지 말 것. 모르면 "공개 정보 부족"으로 표기하고 4장 '분석의 한계'에 모은다.
