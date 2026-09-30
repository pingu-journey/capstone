# 로깅 컨벤션

> 상태: 리뷰 반영안 — D 모듈 적용, 공통 설정과 실행 진입점 연결은 담당자별 적용 예정
>
> 목적: 실행 흐름과 장애 원인을 같은 형식으로 확인하고, 중복 로그와 민감 정보 노출을 방지한다.

## 1. 공통 설정과 담당

- Python 표준 `logging`을 사용하며 별도 로깅 패키지는 추가하지 않는다.
- 각 모듈은 `logger = logging.getLogger(__name__)`으로 로거를 선언한다.
- A가 공통 `configure_logging()`을 `tools/logging_config.py`에 구현한다.
- **각 실행 진입점에서만** 공통 설정을 호출한다: `app.py`(A), `rag.ingest`(B), `eval.embedding_eval`(B). 다른 독립 실행 프로그램에도 같은 규칙을 적용한다.
- import만으로 로그를 설정하지 않는다. 일반 모듈은 `basicConfig()`, `addHandler()`, 전역 레벨 설정을 호출하지 않는다.
- `.env` 로딩은 기존 `config.py`가 담당한다. 진입점은 `config`가 로드된 뒤 로그를 설정하며, 로깅 설정 함수는 `os.getenv()`만 사용한다.
- 기본 레벨은 `INFO`, 상세 진단 시 `LOG_LEVEL=DEBUG`로 전환한다.
- 기본 출력은 콘솔이다. 별도 로그 파일·회전 정책은 이번 범위에 포함하지 않는다.
- Python logging 메시지는 한국어로 작성하고 식별 값은 `key=value`로 기록한다. **`State.log`는 기존 한 줄 실행 이력 형식을 유지한다.**

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
| `INFO` | 주요 단계의 완료 이벤트와 결과 중심 | 후보 선택, 투자 판정, 보고서 저장 |
| `WARNING` | 복구 또는 대체 처리로 계속 진행할 수 있는 비정상 상황 | 해석 불가능한 게시일을 조회일로 대체, 외부 검색 실패 후 시드 후보 사용 |
| `ERROR` | 복구 불가능한 오류를 하위 모듈에서 예외로 전달하고 최상위에서 한 번 traceback과 함께 기록 | 문서 레지스트리 누락, 유효하지 않은 최종 출력, PDF 생성 실패 |

- 모든 노드의 시작·완료를 일괄 기록하지 않는다. 장시간 작업이나 실패 위치 확인이 필요한 경우에만 시작을 INFO로 기록한다. 투자 판단의 외부 LLM 호출이 이에 해당한다.
- 설계된 RAG 웹 보완 경로 이동이나 투자 보류 판정은 정상 흐름이므로 INFO다.
- 게시일이 처음부터 없는 검색 결과는 정상 대체 처리이므로 경고를 생략한다. 게시일이 있지만 해석할 수 없을 때만 WARNING을 기록한다.
- 출처 한 건 생성 성공은 생략한다. 필요하면 호출 단계에서 전체 건수를 기록한다.

## 3. 예외 처리와 민감정보

### 실행 경계에서 한 번 기록

하위 모듈은 안전한 식별 정보를 담은 예외를 발생시킨다. 다시 던질 예외를 같은 위치에서 `logger.error()`나 `logger.exception()`으로 기록하지 않는다.

**처리되지 않은 `Exception`은 각 프로그램의 최상위 실행 경계에서 한 번 기록한다.** `logger.exception()`은 ERROR 레벨과 traceback을 함께 기록한다.

```python
if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except Exception:
        logger.exception("실행 실패")
        raise SystemExit(1)
```

- `SystemExit`와 `KeyboardInterrupt`는 `except Exception`의 대상이 아니다. 정상적인 CLI 종료와 사용자 중단을 실행 오류로 취급하지 않는다.
- 벡터스토어 누락 등 실행 실패로 취급할 오류는 하위 코드에서 `RuntimeError` 등으로 발생시킨다. `raise SystemExit("오류 내용")`으로 실행 경계를 우회하지 않는다.
- 실패 종료용 `SystemExit(1)`은 최상위에서 로그를 남긴 뒤 사용한다.
- 하위 모듈이 복구해 계속 진행할 때는 그 위치에서 WARNING을 기록한다. 복구한 동일 오류를 ERROR로 다시 남기지 않는다.
- 예외를 숨기기 위한 광범위한 `except Exception`은 사용하지 않는다. HTTP/API 경계에서는 예상되는 라이브러리 예외를 명시적으로 처리한다.

### 외부 오류는 안전한 예외로 변환

`logger.exception()`은 실제 예외 문자열과 연결된 예외도 출력한다. 로거에 안전한 메시지를 전달하는 것만으로는 외부 오류의 URL·토큰·응답 원문 노출을 막을 수 없다.

HTTP/API wrapper에서 외부 예외를 **고정된 설명의 자체 예외**로 변환한다. 원문을 메시지에 넣지 않고 `from None`으로 외부 예외 체인의 출력을 억제한다.

```python
from openai import APIError


class ExternalServiceError(RuntimeError):
    pass


# HTTP/API wrapper의 외부 호출 구간
try:
    result = client_call()
except APIError:
    raise ExternalServiceError("외부 API 호출 실패: operation=judge") from None
```

이 예시는 패턴 설명이며 `client_call()`은 실제 호출로 대체한다. HTTP 클라이언트 오류와 응답 파싱 오류도 사용하는 라이브러리의 예외 타입에 맞춰 같은 방식으로 처리한다. wrapper에서는 로그를 중복 기록하지 않는다. Pydantic 오류를 직접 전달할 경우 입력 원문이 예외 문자열에 포함되지 않도록 설정한다.

기록하지 않을 정보:

- API 키, 토큰, Authorization 헤더, `.env` 내용
- 프롬프트 전체, LLM 응답 원문, 검색 문서 본문
- 인증 정보나 서명 토큰이 포함될 수 있는 URL 쿼리 문자열

출처 추적에는 전체 URL 대신 `source_id` 또는 `doc_id`를 사용한다. 제3자 라이브러리 로그 레벨 제한만으로 traceback의 민감정보 문제가 해결되지는 않는다.

## 4. State.log와 Python 로그 구분

| 구분 | 목적 | 기록 내용 |
|---|---|---|
| `State.log` | 실행 결과와 함께 저장할 한 줄 주요 이력 | 기업 선택, 경로 전환, 최종 판정, 보고서 저장 |
| Python `logging` | 개발·운영 중 상세 진단 | 레벨별 이벤트, 처리 건수, 복구 상황, 최상위 예외 traceback |

- 주요 이벤트는 두 곳에 기록할 수 있다. 모든 DEBUG 메시지를 `State.log`에 복사하지 않는다.
- 노드는 반드시 **새 메시지만** `{"log": [message]}` 형태로 반환한다.
- 기존 `state["log"]`까지 다시 반환하면 `add` reducer가 기존 이력을 중복 누적한다. 입력 State를 직접 수정하지 않는다.
- 출처·검색 등 공통 도구는 Python 로거만 사용한다. State 갱신은 호출한 에이전트가 담당한다.

```python
message = f"[judge] {name}: {decision}, 총점 {total_score:.1f}점"
logger.info("투자 판단 완료: startup=%s decision=%s total_score=%.1f", name, decision, total_score)
return {"log": [message]}
```

위 예시는 로그 반환 부분만 보여준다. 실제 노드는 담당하는 다른 State 변경분도 함께 반환한다.

## 5. 설정과 작성 예시

### 개별 모듈

```python
import logging

logger = logging.getLogger(__name__)

logger.info("보고서 저장 완료: report_path=%s", report_path)
logger.debug("출처 중복 제거: before=%d after=%d", before_count, after_count)
logger.warning("게시일 해석 실패, 조회일 사용: source_id=%s", source_id)
```

Python 로그는 형식 통일과 lazy formatting을 위해 메시지와 값을 `%s`, `%d` 인자로 분리한다. `State.log`처럼 이미 완성된 문자열을 저장하는 곳에는 f-string을 사용할 수 있다.

### 공통 설정 모듈 — A 구현

```python
import logging
import os

logger = logging.getLogger(__name__)


def configure_logging():
    requested = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level = requested if requested in allowed else "INFO"
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if requested not in allowed:
        logger.warning("지원하지 않는 LOG_LEVEL, INFO 사용")
```

### 독립 실행 프로그램 연결 — 각 담당자

```python
import config  # 기존 config.py가 .env를 로드한다.
from tools.logging_config import configure_logging

# main() 등 프로그램 정의 후 __main__ 블록에서만 configure_logging() 호출
```

프로그램이 이미 `config`에서 필요한 값을 import하고 있으면 별도의 `import config`는 필요 없다. 공통 설정 모듈에서 `.env`를 다시 읽지 않는다.

## 6. 담당별 적용과 검증

| 담당 | 적용 범위 | 검증 책임 |
|---|---|---|
| A | 공통 `tools/logging_config.py`, `app.py` 실행 경계, `.env.example`의 `LOG_LEVEL=INFO`, 공통 HTTP/API wrapper | **A 소유 `tests/test_logging.py`**에 설정·실패 종료·민감정보 노출 방지 검증 |
| B | `rag.ingest`, `eval.embedding_eval` 실행 경계와 B 모듈 | 공통 설정 호출과 독립 실행 시 INFO 출력·실패 종료 확인 |
| C | 기술 요약·Agentic RAG·경쟁사 모듈 | INFO/WARNING 구분, 새 State 이력만 반환하는지 확인 |
| D | 출처·투자 판단·보고서 모듈 | 기존 `test_references.py`, `test_scoring.py`에서 caplog·예외 전달·이력 검증 |

공통 검증 기준:

- 기본 INFO에서 DEBUG는 출력하지 않고 `LOG_LEVEL=DEBUG`에서 처리 건수를 확인할 수 있다.
- 공통 설정을 import하는 것만으로 전역 로그 설정을 변경하지 않는다.
- 실제 각 진입점의 실패 실행에서 ERROR/traceback은 한 번, 종료 코드는 실패로 기록된다.
- 외부 예외에 테스트용 URL·토큰·응답 원문을 넣고 **최상위에서 렌더링된 traceback 전체**에 노출되지 않는지 확인한다.
- `SystemExit`·`KeyboardInterrupt`는 일반 실패로 변환하거나 ERROR로 기록하지 않는다.
- 기존 `State.log`가 있는 입력에서도 새 이력만 반환해 reducer 적용 후 기존 이력이 한 번만 남는다.

현재 범위: 이 문서는 공통 적용 기준이며 A·B 실행 진입점 구현 완료를 의미하지 않는다. D는 모듈 단위로 적용하고, 실행 진입점 통합 검증은 각 담당자가 수행한다.
