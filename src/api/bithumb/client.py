"""
빗썸 API 2.0 HTTP 클라이언트 구현

aiohttp 기반 비동기 HTTP 클라이언트로 빗썸 API와의 통신을 처리합니다.
"""

import asyncio
import logging
import random
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from config.config import get_settings
from .rate_limiter import get_rate_limiter, RateLimitedRequest
from .exceptions import (
    BithumbAPIError, BithumbNetworkError, BithumbTimeoutError,
    create_bithumb_exception, is_retryable_error, get_retry_delay
)

logger = logging.getLogger(__name__)


class BithumbHTTPClient:
    """
    빗썸 API 2.0 HTTP 클라이언트

    aiohttp.ClientSession 기반으로 빗썸 API와의 HTTP 통신을 처리합니다.
    연결 풀, 타임아웃, Rate Limiting 등을 관리합니다.
    """

    # 빗썸 API 기본 설정
    BASE_URL = "https://api.bithumb.com"
    API_VERSION = "1.2"  # 안정적인 1.2 버전 사용
    USER_AGENT = "BithumbTradingBot/1.0 (aiohttp)"

    def __init__(self):
        """HTTP 클라이언트 초기화"""
        self.settings = get_settings()
        self._session: Optional[ClientSession] = None
        self._closed = False

        # 연결 설정
        self._timeout = ClientTimeout(
            total=30,      # 전체 요청 타임아웃
            connect=10,    # 연결 타임아웃
            sock_read=20   # 소켓 읽기 타임아웃
        )

        # 연결 풀 설정
        self._connector = TCPConnector(
            limit=100,           # 전체 연결 풀 크기
            limit_per_host=20,   # 호스트당 최대 연결 수
            ttl_dns_cache=300,   # DNS 캐시 TTL (5분)
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )

        # 기본 헤더
        self._default_headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Connection": "keep-alive"
        }

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.close()

    async def _ensure_session(self):
        """ClientSession이 생성되어 있는지 확인하고, 없으면 생성"""
        if self._session is None or self._session.closed:
            self._session = ClientSession(
                base_url=self.BASE_URL,
                timeout=self._timeout,
                connector=self._connector,
                headers=self._default_headers,
                raise_for_status=False  # 수동으로 상태 코드 처리
            )
            self._closed = False
            logger.info("빗썸 HTTP 클라이언트 세션 생성 완료")

    async def close(self):
        """클라이언트 세션 종료"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._closed = True
            logger.info("빗썸 HTTP 클라이언트 세션 종료 완료")

    def _build_url(self, endpoint: str) -> str:
        """
        API 엔드포인트 URL 생성

        Args:
            endpoint: API 엔드포인트 경로 (예: "/public/ticker/BTC_KRW" 또는 "/v1/accounts")

        Returns:
            완전한 API URL
        """
        # endpoint가 '/'로 시작하지 않으면 추가
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint

        # v1, v2로 시작하는 엔드포인트는 그대로 사용 (API 2.0)
        # 그 외는 기본 API 1.2 방식
        return f"{self.BASE_URL}{endpoint}"

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_rate_limit: bool = True,
        max_retries: int = 3,
        retry_backoff: float = 1.0
    ) -> Dict[str, Any]:
        """
        HTTP 요청 실행 (Rate Limiting 및 재시도 포함)

        Args:
            method: HTTP 메서드 (GET, POST 등)
            endpoint: API 엔드포인트
            params: URL 파라미터
            data: 요청 본문 데이터
            headers: 추가 헤더
            use_rate_limit: Rate Limiting 사용 여부
            max_retries: 최대 재시도 횟수
            retry_backoff: 재시도 백오프 배율

        Returns:
            API 응답 데이터

        Raises:
            BithumbAPIError: API 오류
            BithumbNetworkError: 네트워크 오류
            BithumbTimeoutError: 타임아웃 오류
        """
        await self._ensure_session()

        url = self._build_url(endpoint)

        # 헤더 병합
        request_headers = self._default_headers.copy()
        if headers:
            request_headers.update(headers)

        logger.debug(f"빗썸 API 요청: {method} {url}")

        # 재시도 로직
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                # Rate Limiting 적용
                if use_rate_limit:
                    async with RateLimitedRequest(endpoint, timeout=30.0):
                        return await self._execute_request(
                            method, url, params, data, request_headers
                        )
                else:
                    return await self._execute_request(
                        method, url, params, data, request_headers
                    )

            except Exception as e:
                last_exception = e

                # 재시도 가능한 오류인지 확인 (안전한 방식으로)
                try:
                    should_retry = is_retryable_error(e)
                except Exception:
                    should_retry = False  # 에러 발생 시 재시도하지 않음

                if not should_retry or attempt >= max_retries:
                    # 재시도 불가능하거나 최대 시도 횟수 도달
                    logger.error(f"빗썸 API 요청 최종 실패 (시도 {attempt + 1}/{max_retries + 1}): {str(e)}")
                    raise e

                # 재시도 대기 시간 계산 (안전한 방식으로)
                try:
                    retry_delay = get_retry_delay(e, attempt) * retry_backoff
                except Exception:
                    retry_delay = 1.0 * (2 ** attempt)  # 기본 지수 백오프

                # 지터 추가 (동시 재시도 방지)
                jitter = random.uniform(0.1, 0.5)
                total_delay = retry_delay + jitter

                logger.warning(
                    f"빗썸 API 요청 실패 (시도 {attempt + 1}/{max_retries + 1}): {str(e)}. "
                    f"{total_delay:.2f}초 후 재시도..."
                )

                await asyncio.sleep(total_delay)

        # 모든 재시도 실패
        if last_exception:
            raise last_exception
        else:
            raise BithumbAPIError("알 수 없는 오류로 인한 요청 실패")

    async def _execute_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]],
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        실제 HTTP 요청 실행

        Args:
            method: HTTP 메서드
            url: 완전한 URL
            params: URL 파라미터
            data: 요청 본문 데이터
            headers: 요청 헤더

        Returns:
            API 응답 데이터

        Raises:
            BithumbAPIError: API 오류
            BithumbNetworkError: 네트워크 오류
            BithumbTimeoutError: 타임아웃 오류
        """
        try:
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers
            ) as response:

                # 응답 상태 로깅
                logger.debug(f"빗썸 API 응답: {response.status} {url}")

                # 응답 본문 읽기
                try:
                    response_text = await response.text()
                except Exception as e:
                    raise BithumbNetworkError(f"응답 본문 읽기 실패: {str(e)}")

                # JSON 파싱 시도
                try:
                    response_data = await response.json()
                except Exception as e:
                    logger.error(f"JSON 파싱 실패: {response_text[:200]}...")
                    raise BithumbAPIError(f"잘못된 JSON 응답: {str(e)}")

                # 빗썸 API 에러 체크
                if response.status != 200:
                    # 적절한 예외 생성
                    exception = create_bithumb_exception(
                        status_code=response.status,
                        response_data=response_data,
                        default_message="API 요청 실패"
                    )
                    logger.error(f"빗썸 API 오류 [{response.status}]: {exception.message}")
                    raise exception

                # 빗썸 API 응답에서 status 필드 체크
                api_status = response_data.get("status")
                if api_status and api_status != "0000":
                    # API 레벨에서의 오류
                    exception = create_bithumb_exception(
                        status_code=200,  # HTTP는 성공이지만 API 레벨 오류
                        response_data=response_data,
                        default_message="API 응답 오류"
                    )
                    logger.error(f"빗썸 API 응답 오류 [{api_status}]: {exception.message}")
                    raise exception

                return response_data

        except asyncio.TimeoutError as e:
            logger.error(f"빗썸 API 타임아웃: {url}")
            raise BithumbTimeoutError(f"요청 타임아웃: {url}")


        except aiohttp.ClientConnectorError as e:
            logger.error(f"빗썸 API 연결 실패: {str(e)}")
            raise BithumbNetworkError(f"서버 연결 실패: {str(e)}")

        except aiohttp.ClientError as e:
            logger.error(f"빗썸 API 클라이언트 오류: {str(e)}")
            raise BithumbNetworkError(f"네트워크 오류: {str(e)}")

        except BithumbAPIError:
            # 이미 처리된 빗썸 API 오류는 그대로 전파
            raise
        except BithumbNetworkError:
            # 이미 처리된 네트워크 오류는 그대로 전파
            raise
        except BithumbTimeoutError:
            # 이미 처리된 타임아웃 오류는 그대로 전파
            raise

        except Exception as e:
            logger.error(f"빗썸 API 예상치 못한 오류: {str(e)}")
            raise BithumbAPIError(f"예상치 못한 오류: {str(e)}")

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_rate_limit: bool = True,
        max_retries: int = 3,
        retry_backoff: float = 1.0
    ) -> Dict[str, Any]:
        """
        GET 요청 실행

        Args:
            endpoint: API 엔드포인트
            params: URL 파라미터
            headers: 추가 헤더
            use_rate_limit: Rate Limiting 사용 여부
            max_retries: 최대 재시도 횟수
            retry_backoff: 재시도 백오프 배율

        Returns:
            API 응답 데이터
        """
        return await self._make_request(
            "GET", endpoint, params=params, headers=headers,
            use_rate_limit=use_rate_limit, max_retries=max_retries,
            retry_backoff=retry_backoff
        )

    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_rate_limit: bool = True,
        max_retries: int = 3,
        retry_backoff: float = 1.0
    ) -> Dict[str, Any]:
        """
        POST 요청 실행

        Args:
            endpoint: API 엔드포인트
            data: 요청 본문 데이터
            headers: 추가 헤더
            use_rate_limit: Rate Limiting 사용 여부
            max_retries: 최대 재시도 횟수
            retry_backoff: 재시도 백오프 배율

        Returns:
            API 응답 데이터
        """
        return await self._make_request(
            "POST", endpoint, data=data, headers=headers,
            use_rate_limit=use_rate_limit, max_retries=max_retries,
            retry_backoff=retry_backoff
        )

    async def health_check(self) -> bool:
        """
        빗썸 API 연결 상태 확인 (Rate Limiting 없음)

        Returns:
            연결 상태 (True: 정상, False: 오류)
        """
        try:
            # 빗썸 공개 API 중 가장 가벼운 엔드포인트 호출
            # health check는 Rate Limiting을 적용하지 않음
            response = await self.get("/public/ticker/BTC_KRW", use_rate_limit=False)

            # 기본적인 응답 구조 확인
            if "status" in response and response["status"] == "0000":
                logger.info("빗썸 API 연결 상태: 정상")
                return True
            else:
                logger.warning(f"빗썸 API 응답 이상: {response}")
                return False

        except Exception as e:
            logger.error(f"빗썸 API 연결 확인 실패: {str(e)}")
            return False

    def get_rate_limit_status(self) -> Dict[str, Dict]:
        """
        Rate Limit 상태 정보 반환

        Returns:
            API 종류별 Rate Limit 상태
        """
        rate_limiter = get_rate_limiter()
        return rate_limiter.get_rate_limit_status()

    async def wait_for_rate_limit(self, endpoint: str, max_wait: float = 60.0) -> bool:
        """
        Rate Limit 여유가 생길 때까지 대기

        Args:
            endpoint: API 엔드포인트
            max_wait: 최대 대기 시간 (초)

        Returns:
            대기 성공 여부
        """
        rate_limiter = get_rate_limiter()
        return await rate_limiter.wait_for_available_slot(endpoint, max_wait)

    def can_make_request_now(self, endpoint: str) -> bool:
        """
        현재 즉시 요청 가능한지 확인

        Args:
            endpoint: API 엔드포인트

        Returns:
            즉시 요청 가능 여부
        """
        rate_limiter = get_rate_limiter()
        return rate_limiter.can_make_request(endpoint)

    @property
    def is_closed(self) -> bool:
        """클라이언트 세션이 닫혔는지 확인"""
        return self._closed or (self._session is not None and self._session.closed)

    def __del__(self):
        """소멸자에서 세션 정리"""
        if not self.is_closed:
            # 이벤트 루프가 있는 경우에만 정리
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.create_task(self.close())
            except RuntimeError:
                # 이벤트 루프가 없거나 닫힌 경우 무시
                pass


# 전역 클라이언트 인스턴스 (싱글톤 패턴)
_global_client: Optional[BithumbHTTPClient] = None


async def get_http_client() -> BithumbHTTPClient:
    """
    전역 HTTP 클라이언트 인스턴스 반환

    싱글톤 패턴으로 하나의 클라이언트 인스턴스만 사용하여
    연결 풀을 효율적으로 관리합니다.

    Returns:
        BithumbHTTPClient 인스턴스
    """
    global _global_client

    if _global_client is None or _global_client.is_closed:
        _global_client = BithumbHTTPClient()
        await _global_client._ensure_session()

    return _global_client


async def close_http_client():
    """전역 HTTP 클라이언트 정리"""
    global _global_client

    if _global_client is not None:
        await _global_client.close()
        _global_client = None