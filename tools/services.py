"""Classify failures before retrying; authentication/configuration never become evidence gaps."""

from httpx import HTTPError
from openai import APIError, AuthenticationError, PermissionDeniedError, BadRequestError
from requests.exceptions import RequestException
from tavily.errors import InvalidAPIKeyError, MissingAPIKeyError, ForbiddenError, BadRequestError as TavilyBadRequestError, TimeoutError as TavilyTimeoutError, UsageLimitExceededError


class ConfigurationError(RuntimeError):
    pass


class TransientServiceError(RuntimeError):
    pass


def check_service_error(error):
    if isinstance(error, (ConfigurationError, TransientServiceError)):
        raise error
    # Both HTTP clients and HfHubHTTPError expose a response. requests responses
    # are falsey on errors, so checking truthiness would hide auth failures.
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None) if response is not None else getattr(error, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        raise ConfigurationError("외부 서비스 인증 또는 요청 설정 오류") from None
    if isinstance(error, (AuthenticationError, PermissionDeniedError, BadRequestError,
                          InvalidAPIKeyError, MissingAPIKeyError, ForbiddenError, TavilyBadRequestError)):
        raise ConfigurationError("외부 서비스 인증 또는 요청 설정 오류") from None
    if isinstance(error, (APIError, HTTPError, RequestException, TavilyTimeoutError, UsageLimitExceededError)):
        raise TransientServiceError("외부 서비스 일시적 호출 실패") from None
