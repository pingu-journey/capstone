# 로깅 컨벤션

> 상태: 팀 공유용 제안 — 합의 후 A·B·C·D 공통 적용
>
> 목적: 실행 흐름과 장애 원인을 같은 형식으로 확인하고, 중복 로그와 민감 정보 노출을 방지한다.

## 1. 공통 설정과 담당

- Python 표준 `logging`을 사용한다. 별도 로깅 패키지는 추가하지 않는다.
- 각 모듈은 `logger = logging.getLogger(__name__)`으로 로거를 선언한다.
- `basicConfig`, 핸들러, 출력 레벨 설정은 **A의 `app.py`에서만** 수행한다. 개별 모듈은 전역 설정을 변경하지 않는다.
- 기본 레벨은 `INFO`, 상세 진단 시 `LOG_LEVEL=DEBUG` 환경변수로 전환한다.
- 기본 출력은 콘솔이다. 별도 로그 파일·회전 정책은 이번 범위에 포함하지 않는다.
- 로그 메시지는 한국어로 작성하고, 기업명·문서 ID·건수 등 식별 값은 `key=value`로 표시한다.

출력 형식:

```text
시각 | 레벨 | 모듈명 | 메시지
2026-09-30 11:30:00 | INFO | agents.investment_judge | 투자 판단 완료: startup=Amperon decision=보류 total_score=65.0
```

시각은 실행 환경의 로컬 시간을 사용한다.

## 2. 레벨 기준

| 레벨 | 기록 대상 | 예시 |
|---|---|---|
| `DEBUG` | 상세 처리 정보와 중간 건수 | 검색 결과 수, 출처 필터 전후 건수, 중복 제거 건수 |
| `INFO` | 주요 단계의 시작·완료와 실행 결과 | 후보 선택, 에이전트 완료, 투자 판정, 보고서 저장 |
| `WARNING` | 정상 경로에서 벗어났지만 대체 처리로 계속 진행할 수 있는 상황 | 잘못된 게시일을 조회일로 대체, 외부 검색 실패 후 시드 후보 사용 |
| `ERROR` | 현재 실행을 완료하지 못하는 오류 | 필수 문서 레지스트리 누락, 유효하지 않은 최종 출력, PDF 생성 실패 |

정상적으로 예상되는 분기를 오류로 기록하지 않는다.

- RAG 관련 문서가 부족해 설계된 웹 보완 경로로 이동: `INFO`
- 투자 판단 결과가 보류: `INFO`
- 게시일이 처음부터 없는 검색 결과: 정상 대체 처리이므로 경고 생략
- 게시일이 있지만 해석할 수 없어 조회일로 대체: `WARNING`
- 출처 한 건 생성 성공: 로그 생략. 필요하면 호출 단계에서 전체 건수를 기록

## 3. 예외 처리와 중복 방지

하위 모듈은 필요한 식별 정보를 담은 예외를 발생시킨다. 다시 던질 예외를 같은 위치에서 `ERROR`로 기록하지 않는다.

```python
raise ValueError(f"docs.yaml에 문서 ID가 없습니다: {doc_id}")
```

처리되지 않은 예외는 A의 최상위 실행 경계에서 **한 번만** `logger.exception()`으로 기록한다. 이때 traceback을 포함하고 종료 코드는 실패를 나타내야 한다.

```python
if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except Exception:
        logger.exception("실행 실패")
        raise SystemExit(1)
```

하위 모듈에서 오류를 복구해 계속 실행하는 경우에는 해당 위치에서 `WARNING`을 기록한다. 같은 오류를 다시 `ERROR`로 남기지 않는다. 예외가 필요한 곳에서만 처리하고, 오류를 숨기기 위한 광범위한 `except Exception`은 사용하지 않는다.

## 4. State.log와 Python 로그 구분

| 구분 | 목적 | 기록 내용 |
|---|---|---|
| `State.log` | 실행 결과와 함께 저장할 주요 이력 | 기업 선택, 경로 전환, 최종 판정, 보고서 저장 |
| Python `logging` | 개발·운영 중 상세 진단 | 레벨별 이벤트, 처리 건수, 복구 상황, 최상위 예외 traceback |

- 주요 이벤트는 두 곳에 기록할 수 있다. 모든 DEBUG 메시지를 `State.log`에 복사하지 않는다.
- 노드는 새 이력만 `{"log": [message]}` 형태로 반환한다. 기존 이력 전체를 반환하면 reducer에 의해 중복 누적된다.
- 출처·검색 등 공통 도구는 Python 로거만 사용한다. State 갱신은 호출한 에이전트가 담당한다.

## 5. 작성 예시

### 각 모듈 — B·C·D 및 A의 개별 모듈

```python
import logging

logger = logging.getLogger(__name__)

logger.info("기업 평가 시작: startup=%s", name)
logger.debug("출처 중복 제거: before=%d after=%d", before_count, after_count)
logger.warning("게시일 해석 실패, 조회일 사용: source_id=%s", source_id)
```

메시지와 값을 분리해 전달한다. 로그 호출에는 f-string 대신 `%s`, `%d` 인자를 사용한다. 로그가 비활성화된 경우 불필요한 문자열 포맷팅을 줄이고 형식을 통일하기 위함이다.

### 실행 진입점 — A

```python
import logging
import os

from dotenv import load_dotenv

from config import ROOT

logger = logging.getLogger(__name__)


def configure_logging():
    load_dotenv(ROOT / ".env")
    requested = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level = requested if requested in allowed else "INFO"
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 외부 HTTP 라이브러리의 요청 로그는 기본적으로 숨긴다.
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if requested not in allowed:
        logger.warning("지원하지 않는 LOG_LEVEL, INFO 사용")
```

외부 라이브러리 DEBUG 로그가 필요한 경우에만 A가 별도로 조정한다. 각 모듈은 `basicConfig()`나 `addHandler()`를 호출하지 않는다.

## 6. 기록하지 않을 정보

- API 키, 토큰, Authorization 헤더, `.env` 내용
- 프롬프트 전체, LLM 응답 원문, 검색 문서 본문
- 인증 정보나 서명 토큰이 포함될 수 있는 URL 쿼리 문자열# 로깅 컨벤션

> 상태: 팀 공유용 제안 — 합의 후 A·B·C·D 공통 적용
>
> 목적: 실행 흐름과 장애 원인을 같은 형식으로 확인하고, 중복 로그와 민감 정보 노출을 방지한다.

## 1. 공통 설정과 담당

- Python 표준 `logging`을 사용한다. 별도 로깅 패키지는 추가하지 않는다.
- 각 모듈은 `logger = logging.getLogger(__name__)`으로 로거를 선언한다.
- `basicConfig`, 핸들러, 출력 레벨 설정은 **A의 `app.py`에서만** 수행한다. 개별 모듈은 전역 설정을 변경하지 않는다.
- 기본 레벨은 `INFO`, 상세 진단 시 `LOG_LEVEL=DEBUG` 환경변수로 전환한다.
- 기본 출력은 콘솔이다. 별도 로그 파일·회전 정책은 이번 범위에 포함하지 않는다.
- 로그 메시지는 한국어로 작성하고, 기업명·문서 ID·건수 등 식별 값은 `key=value`로 표시한다.

출력 형식:

```text
시각 | 레벨 | 모듈명 | 메시지
2026-09-30 11:30:00 | INFO | agents.investment_judge | 투자 판단 완료: startup=Amperon decision=보류 total_score=65.0
```

시각은 실행 환경의 로컬 시간을 사용한다.

## 2. 레벨 기준

| 레벨 | 기록 대상 | 예시 |
|---|---|---|
| `DEBUG` | 상세 처리 정보와 중간 건수 | 검색 결과 수, 출처 필터 전후 건수, 중복 제거 건수 |
| `INFO` | 주요 단계의 시작·완료와 실행 결과 | 후보 선택, 에이전트 완료, 투자 판정, 보고서 저장 |
| `WARNING` | 정상 경로에서 벗어났지만 대체 처리로 계속 진행할 수 있는 상황 | 잘못된 게시일을 조회일로 대체, 외부 검색 실패 후 시드 후보 사용 |
| `ERROR` | 현재 실행을 완료하지 못하는 오류 | 필수 문서 레지스트리 누락, 유효하지 않은 최종 출력, PDF 생성 실패 |

정상적으로 예상되는 분기를 오류로 기록하지 않는다.

- RAG 관련 문서가 부족해 설계된 웹 보완 경로로 이동: `INFO`
- 투자 판단 결과가 보류: `INFO`
- 게시일이 처음부터 없는 검색 결과: 정상 대체 처리이므로 경고 생략
- 게시일이 있지만 해석할 수 없어 조회일로 대체: `WARNING`
- 출처 한 건 생성 성공: 로그 생략. 필요하면 호출 단계에서 전체 건수를 기록

## 3. 예외 처리와 중복 방지

하위 모듈은 필요한 식별 정보를 담은 예외를 발생시킨다. 다시 던질 예외를 같은 위치에서 `ERROR`로 기록하지 않는다.

```python
raise ValueError(f"docs.yaml에 문서 ID가 없습니다: {doc_id}")
```

처리되지 않은 예외는 A의 최상위 실행 경계에서 **한 번만** `logger.exception()`으로 기록한다. 이때 traceback을 포함하고 종료 코드는 실패를 나타내야 한다.

```python
if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except Exception:
        logger.exception("실행 실패")
        raise SystemExit(1)
```

하위 모듈에서 오류를 복구해 계속 실행하는 경우에는 해당 위치에서 `WARNING`을 기록한다. 같은 오류를 다시 `ERROR`로 남기지 않는다. 예외가 필요한 곳에서만 처리하고, 오류를 숨기기 위한 광범위한 `except Exception`은 사용하지 않는다.

## 4. State.log와 Python 로그 구분

| 구분 | 목적 | 기록 내용 |
|---|---|---|
| `State.log` | 실행 결과와 함께 저장할 주요 이력 | 기업 선택, 경로 전환, 최종 판정, 보고서 저장 |
| Python `logging` | 개발·운영 중 상세 진단 | 레벨별 이벤트, 처리 건수, 복구 상황, 최상위 예외 traceback |

- 주요 이벤트는 두 곳에 기록할 수 있다. 모든 DEBUG 메시지를 `State.log`에 복사하지 않는다.
- 노드는 새 이력만 `{"log": [message]}` 형태로 반환한다. 기존 이력 전체를 반환하면 reducer에 의해 중복 누적된다.
- 출처·검색 등 공통 도구는 Python 로거만 사용한다. State 갱신은 호출한 에이전트가 담당한다.

## 5. 작성 예시

### 각 모듈 — B·C·D 및 A의 개별 모듈

```python
import logging

logger = logging.getLogger(__name__)

logger.info("기업 평가 시작: startup=%s", name)
logger.debug("출처 중복 제거: before=%d after=%d", before_count, after_count)
logger.warning("게시일 해석 실패, 조회일 사용: source_id=%s", source_id)
```

메시지와 값을 분리해 전달한다. 로그 호출에는 f-string 대신 `%s`, `%d` 인자를 사용한다. 로그가 비활성화된 경우 불필요한 문자열 포맷팅을 줄이고 형식을 통일하기 위함이다.

### 실행 진입점 — A

```python
import logging
import os

from dotenv import load_dotenv

from config import ROOT

logger = logging.getLogger(__name__)


def configure_logging():
    load_dotenv(ROOT / ".env")
    requested = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level = requested if requested in allowed else "INFO"
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 외부 HTTP 라이브러리의 요청 로그는 기본적으로 숨긴다.
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if requested not in allowed:
        logger.warning("지원하지 않는 LOG_LEVEL, INFO 사용")
```

외부 라이브러리 DEBUG 로그가 필요한 경우에만 A가 별도로 조정한다. 각 모듈은 `basicConfig()`나 `addHandler()`를 호출하지 않는다.

## 6. 기록하지 않을 정보

- API 키, 토큰, Authorization 헤더, `.env` 내용
- 프롬프트 전체, LLM 응답 원문, 검색 문서 본문
- 인증 정보나 서명 토큰이 포함될 수 있는 URL 쿼리 문자열

출처 추적에는 전체 URL 대신 `source_id` 또는 `doc_id`를 우선 사용한다. 외부 오류 메시지에 민감 정보가 포함될 수 있으면 그대로 로그에 전달하지 않고 안전한 설명으로 변환한다.

## 7. 적용 순서와 확인

1. 팀이 이 제안을 합의한다.
2. A가 `app.py`에 공통 설정·최상위 예외 처리를 적용하고 `.env.example`에 `LOG_LEVEL=INFO`를 추가한다.
3. A·B·C·D가 각 소유 모듈에 레벨 기준을 적용한다.
4. D는 출처 모듈의 게시일 해석 실패에 WARNING, 필터·중복 제거 건수에 DEBUG를 추가한다. 문서 레지스트리 오류는 기존 예외로 전달한다.

검증은 pytest의 `caplog`로 레벨·핵심 식별 정보·건수를 확인한다. 기존 반환값과 예외 동작이 유지되는지 함께 검사한다. 통합 실행에서는 기본 INFO에서 DEBUG가 나오지 않는지, 실패 시 traceback이 한 번만 기록되고 실패 종료되는지 확인한다.


출처 추적에는 전체 URL 대신 `source_id` 또는 `doc_id`를 우선 사용한다. 외부 오류 메시지에 민감 정보가 포함될 수 있으면 그대로 로그에 전달하지 않고 안전한 설명으로 변환한다.

## 7. 적용 순서와 확인

1. 팀이 이 제안을 합의한다.
2. A가 `app.py`에 공통 설정·최상위 예외 처리를 적용하고 `.env.example`에 `LOG_LEVEL=INFO`를 추가한다.
3. A·B·C·D가 각 소유 모듈에 레벨 기준을 적용한다.
4. D는 출처 모듈의 게시일 해석 실패에 WARNING, 필터·중복 제거 건수에 DEBUG를 추가한다. 문서 레지스트리 오류는 기존 예외로 전달한다.

검증은 pytest의 `caplog`로 레벨·핵심 식별 정보·건수를 확인한다. 기존 반환값과 예외 동작이 유지되는지 함께 검사한다. 통합 실행에서는 기본 INFO에서 DEBUG가 나오지 않는지, 실패 시 traceback이 한 번만 기록되고 실패 종료되는지 확인한다.
