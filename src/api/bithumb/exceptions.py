"""
빗썸 API 예외 클래스 정의

API 호출 중 발생할 수 있는 다양한 예외 상황을 정의합니다.
"""

from typing import Optional, Dict, Any
import asyncio
import aiohttp


class BithumbAPIError(Exception):
    """빗썸 API 기본 예외 클래스"""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict[str, Any]] = None):
        """
        빗썸 API 예외 초기화

        Args:
            message: 오류 메시지
            status_code: HTTP 상태 코드
            response_data: API 응답 데이터
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}

    def __str__(self) -> str:
        if self.status_code:
            return f"BithumbAPIError [{self.status_code}]: {self.message}"
        return f"BithumbAPIError: {self.message}"


class BithumbAuthenticationError(BithumbAPIError):
    """인증 관련 오류"""

    def __init__(self, message: str = "인증 실패", **kwargs):
        super().__init__(message, **kwargs)


class BithumbRateLimitError(BithumbAPIError):
    """Rate Limit 초과 오류"""

    def __init__(self, message: str = "API 호출 제한 초과", **kwargs):
        super().__init__(message, **kwargs)


class BithumbNetworkError(BithumbAPIError):
    """네트워크 연결 오류"""

    def __init__(self, message: str = "네트워크 연결 실패", **kwargs):
        super().__init__(message, **kwargs)


class BithumbTimeoutError(BithumbAPIError):
    """타임아웃 오류"""

    def __init__(self, message: str = "요청 타임아웃", **kwargs):
        super().__init__(message, **kwargs)


class BithumbServerError(BithumbAPIError):
    """서버 오류 (5xx)"""

    def __init__(self, message: str = "서버 오류", **kwargs):
        super().__init__(message, **kwargs)


class BithumbClientError(BithumbAPIError):
    """클라이언트 오류 (4xx)"""

    def __init__(self, message: str = "클라이언트 오류", **kwargs):
        super().__init__(message, **kwargs)


class BithumbInsufficientBalanceError(BithumbAPIError):
    """잔고 부족 오류"""

    def __init__(self, message: str = "잔고 부족", **kwargs):
        super().__init__(message, **kwargs)


class BithumbOrderError(BithumbAPIError):
    """주문 관련 오류"""

    def __init__(self, message: str = "주문 처리 실패", **kwargs):
        super().__init__(message, **kwargs)


class BithumbMaintenanceError(BithumbAPIError):
    """서비스 점검 중 오류"""

    def __init__(self, message: str = "서비스 점검 중", **kwargs):
        super().__init__(message, **kwargs)


# 빗썸 API 에러 코드별 매핑
BITHUMB_ERROR_CODE_MAPPING = {
    # 인증 관련
    "5100": BithumbAuthenticationError,
    "5200": BithumbAuthenticationError,
    "5300": BithumbAuthenticationError,
    "5302": BithumbAuthenticationError,
    "5400": BithumbAuthenticationError,

    # 파라미터 관련
    "5500": BithumbClientError,
    "5600": BithumbClientError,

    # 잔고 관련
    "5900": BithumbInsufficientBalanceError,

    # 주문 관련
    "5800": BithumbOrderError,
    "5801": BithumbOrderError,
    "5802": BithumbOrderError,
    "5803": BithumbOrderError,
    "5804": BithumbOrderError,

    # 시스템 관련
    "5000": BithumbServerError,
    "5001": BithumbMaintenanceError,
    "5002": BithumbServerError,
}


def create_bithumb_exception(
    status_code: int,
    response_data: Dict[str, Any],
    default_message: str = "API 오류"
) -> BithumbAPIError:
    """
    HTTP 상태 코드와 응답 데이터를 기반으로 적절한 예외 생성

    Args:
        status_code: HTTP 상태 코드
        response_data: API 응답 데이터
        default_message: 기본 오류 메시지

    Returns:
        적절한 빗썸 API 예외 인스턴스
    """
    # 응답에서 에러 코드와 메시지 추출
    error_code = response_data.get("status", "")
    error_message = response_data.get("message", default_message)

    # 빗썸 에러 코드별 예외 매핑
    if error_code in BITHUMB_ERROR_CODE_MAPPING:
        exception_class = BITHUMB_ERROR_CODE_MAPPING[error_code]
        return exception_class(
            message=f"[{error_code}] {error_message}",
            status_code=status_code,
            response_data=response_data
        )

    # HTTP 상태 코드별 예외 매핑
    if status_code == 401:
        return BithumbAuthenticationError(
            message=error_message,
            status_code=status_code,
            response_data=response_data
        )
    elif status_code == 403:
        return BithumbAuthenticationError(
            message=f"접근 권한 없음: {error_message}",
            status_code=status_code,
            response_data=response_data
        )
    elif status_code == 429:
        return BithumbRateLimitError(
            message=f"Rate Limit 초과: {error_message}",
            status_code=status_code,
            response_data=response_data
        )
    elif 400 <= status_code < 500:
        return BithumbClientError(
            message=error_message,
            status_code=status_code,
            response_data=response_data
        )
    elif 500 <= status_code < 600:
        return BithumbServerError(
            message=error_message,
            status_code=status_code,
            response_data=response_data
        )

    # 기본 예외
    return BithumbAPIError(
        message=error_message,
        status_code=status_code,
        response_data=response_data
    )


def is_retryable_error(error: BaseException) -> bool:
    """
    재시도 가능한 오류인지 판단

    Args:
        error: 발생한 예외

    Returns:
        재시도 가능 여부
    """
    # 네트워크 관련 오류는 재시도 가능
    if isinstance(error, (BithumbNetworkError, BithumbTimeoutError)):
        return True

    # 서버 오류는 재시도 가능
    if isinstance(error, BithumbServerError):
        return True

    # Rate Limit 오류는 재시도 가능
    if isinstance(error, BithumbRateLimitError):
        return True

    # 유지보수 중 오류는 재시도 가능
    if isinstance(error, BithumbMaintenanceError):
        return True

    # HTTP 연결 오류 (aiohttp의 올바른 예외 클래스들)
    if isinstance(error, (aiohttp.ClientError, aiohttp.ServerTimeoutError)):
        return True

    # asyncio 타임아웃 오류
    if isinstance(error, asyncio.TimeoutError):
        return True

    return False


def get_retry_delay(error: BaseException, attempt: int) -> float:
    """
    재시도 대기 시간 계산 (지수 백오프)

    Args:
        error: 발생한 예외
        attempt: 시도 횟수

    Returns:
        대기 시간 (초)
    """
    # Rate Limit 오류의 경우 더 긴 대기
    if isinstance(error, BithumbRateLimitError):
        return min(60.0, 2 ** attempt)

    # 서버 오류의 경우 일반적인 지수 백오프
    if isinstance(error, (BithumbServerError, BithumbMaintenanceError)):
        return min(30.0, 1.0 * (2 ** attempt))

    # 네트워크 오류의 경우 짧은 대기
    return min(10.0, 0.5 * (2 ** attempt))