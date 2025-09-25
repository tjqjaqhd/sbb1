"""
빗썸 API Rate Limiting 시스템

Token Bucket 알고리즘을 사용하여 API 호출 제한을 관리합니다.
빗썸 API는 초당 10회 호출 제한이 있습니다.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class APIEndpointType(Enum):
    """API 엔드포인트 타입"""
    PUBLIC = "public"           # 공개 API (시세, 호가 등)
    PRIVATE = "private"         # 개인 API (잔고, 주문 등)
    ORDER = "order"            # 주문 관련 API (별도 제한)
    WEBSOCKET = "websocket"    # WebSocket 연결


@dataclass
class RateLimitConfig:
    """Rate Limit 설정"""
    max_requests: int           # 최대 요청 수
    time_window: float         # 시간 윈도우 (초)
    burst_allowance: int       # 버스트 허용량

    @property
    def requests_per_second(self) -> float:
        """초당 요청 수"""
        return self.max_requests / self.time_window


class TokenBucket:
    """
    Token Bucket 알고리즘 구현

    일정한 속도로 토큰을 채우고, 요청 시마다 토큰을 소모합니다.
    토큰이 없으면 대기하거나 요청을 거부합니다.
    """

    def __init__(self, config: RateLimitConfig):
        """
        Token Bucket 초기화

        Args:
            config: Rate Limit 설정
        """
        self.config = config
        self.tokens = float(config.max_requests)  # 현재 토큰 수
        self.last_refill = time.time()           # 마지막 토큰 충전 시간
        self._lock = asyncio.Lock()              # 동시성 제어

        logger.debug(
            f"TokenBucket 초기화: "
            f"max_requests={config.max_requests}, "
            f"time_window={config.time_window}s, "
            f"rate={config.requests_per_second:.2f}/s"
        )

    def _refill_tokens(self) -> None:
        """토큰 충전"""
        now = time.time()
        time_elapsed = now - self.last_refill

        if time_elapsed > 0:
            # 경과 시간에 따른 토큰 충전
            tokens_to_add = time_elapsed * self.config.requests_per_second
            self.tokens = min(
                self.config.max_requests,
                self.tokens + tokens_to_add
            )
            self.last_refill = now

            if tokens_to_add > 0:
                logger.debug(f"토큰 {tokens_to_add:.2f}개 충전, 현재: {self.tokens:.2f}")

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        토큰 획득 (비동기)

        Args:
            tokens: 필요한 토큰 수
            timeout: 타임아웃 (초, None이면 무한 대기)

        Returns:
            토큰 획득 성공 여부
        """
        async with self._lock:
            start_time = time.time()

            while True:
                self._refill_tokens()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(f"토큰 {tokens}개 소모, 잔여: {self.tokens:.2f}")
                    return True

                # 타임아웃 체크
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        logger.warning(f"토큰 획득 타임아웃: {timeout}s")
                        return False

                # 다음 토큰까지 대기 시간 계산
                tokens_needed = tokens - self.tokens
                wait_time = min(
                    tokens_needed / self.config.requests_per_second,
                    1.0  # 최대 1초 대기
                )

                logger.debug(f"토큰 부족, {wait_time:.2f}초 대기")
                await asyncio.sleep(wait_time)

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        토큰 즉시 획득 시도 (대기 없음)

        Args:
            tokens: 필요한 토큰 수

        Returns:
            토큰 획득 성공 여부
        """
        self._refill_tokens()

        if self.tokens >= tokens:
            self.tokens -= tokens
            logger.debug(f"토큰 {tokens}개 즉시 획득, 잔여: {self.tokens:.2f}")
            return True

        logger.debug(f"토큰 부족으로 즉시 획득 실패, 필요: {tokens}, 현재: {self.tokens:.2f}")
        return False

    @property
    def available_tokens(self) -> float:
        """현재 사용 가능한 토큰 수"""
        self._refill_tokens()
        return self.tokens

    @property
    def is_full(self) -> bool:
        """토큰 버킷이 가득 찬 상태인지 확인"""
        return self.available_tokens >= self.config.max_requests


class BithumbRateLimiter:
    """
    빗썸 API Rate Limiter

    API 종류별로 별도의 Token Bucket을 관리하여
    빗썸의 API 호출 제한을 준수합니다.
    """

    # API 종류별 기본 설정
    DEFAULT_CONFIGS = {
        APIEndpointType.PUBLIC: RateLimitConfig(
            max_requests=20,      # 공개 API는 좀 더 여유
            time_window=1.0,      # 1초
            burst_allowance=5     # 버스트 허용
        ),
        APIEndpointType.PRIVATE: RateLimitConfig(
            max_requests=10,      # 빗썸 기본 제한
            time_window=1.0,      # 1초
            burst_allowance=2     # 개인 API는 보수적
        ),
        APIEndpointType.ORDER: RateLimitConfig(
            max_requests=5,       # 주문 API는 더 엄격
            time_window=1.0,      # 1초
            burst_allowance=1     # 버스트 최소화
        ),
        APIEndpointType.WEBSOCKET: RateLimitConfig(
            max_requests=1,       # WebSocket 연결은 제한적
            time_window=5.0,      # 5초마다 1회
            burst_allowance=0     # 버스트 없음
        )
    }

    def __init__(self, custom_configs: Optional[Dict[APIEndpointType, RateLimitConfig]] = None):
        """
        Rate Limiter 초기화

        Args:
            custom_configs: 사용자 정의 설정 (없으면 기본값 사용)
        """
        self.configs = self.DEFAULT_CONFIGS.copy()
        if custom_configs:
            self.configs.update(custom_configs)

        # API 타입별 Token Bucket 생성
        self.buckets: Dict[APIEndpointType, TokenBucket] = {}
        for api_type, config in self.configs.items():
            self.buckets[api_type] = TokenBucket(config)

        # 전역 세마포어 (전체 API 호출 제한)
        self.global_semaphore = asyncio.Semaphore(50)  # 동시 호출 최대 50개

        logger.info(f"빗썸 Rate Limiter 초기화 완료: {len(self.buckets)}개 API 타입")

    def _get_api_type(self, endpoint: str) -> APIEndpointType:
        """
        엔드포인트에서 API 타입 추출

        Args:
            endpoint: API 엔드포인트

        Returns:
            API 타입
        """
        endpoint = endpoint.lower().strip('/')

        if endpoint.startswith('public/'):
            return APIEndpointType.PUBLIC
        elif any(keyword in endpoint for keyword in ['order', 'trade']):
            return APIEndpointType.ORDER
        elif endpoint.startswith('info/') or 'balance' in endpoint:
            return APIEndpointType.PRIVATE
        else:
            # 기본적으로 개인 API로 분류 (안전)
            return APIEndpointType.PRIVATE

    async def acquire_permission(
        self,
        endpoint: str,
        timeout: Optional[float] = 30.0
    ) -> bool:
        """
        API 호출 허가 획득

        Args:
            endpoint: API 엔드포인트
            timeout: 타임아웃 (초)

        Returns:
            호출 허가 획득 성공 여부
        """
        api_type = self._get_api_type(endpoint)
        bucket = self.buckets[api_type]

        logger.debug(f"API 호출 허가 요청: {endpoint} ({api_type.value})")

        try:
            # 전역 세마포어 획득
            await asyncio.wait_for(
                self.global_semaphore.acquire(),
                timeout=timeout
            )

            try:
                # Token Bucket에서 토큰 획득
                success = await bucket.acquire(tokens=1, timeout=timeout)

                if success:
                    logger.debug(f"API 호출 허가 획득: {endpoint}")
                    return True
                else:
                    # 토큰 획득 실패 시 세마포어 해제
                    self.global_semaphore.release()
                    logger.warning(f"Rate limit 초과: {endpoint}")
                    return False

            except Exception:
                # 예외 발생 시 세마포어 해제
                self.global_semaphore.release()
                raise

        except asyncio.TimeoutError:
            logger.error(f"API 호출 허가 타임아웃: {endpoint}")
            return False

    def release_permission(self, endpoint: str) -> None:
        """
        API 호출 허가 해제

        Args:
            endpoint: API 엔드포인트
        """
        self.global_semaphore.release()
        logger.debug(f"API 호출 허가 해제: {endpoint}")

    def can_make_request(self, endpoint: str) -> bool:
        """
        즉시 요청 가능한지 확인 (대기 없음)

        Args:
            endpoint: API 엔드포인트

        Returns:
            즉시 요청 가능 여부
        """
        api_type = self._get_api_type(endpoint)
        bucket = self.buckets[api_type]

        # 전역 세마포어와 토큰 버킷 모두 확인
        return (
            self.global_semaphore._value > 0 and
            bucket.try_acquire(tokens=0)  # 토큰 소모 없이 확인만
        )

    def get_rate_limit_status(self) -> Dict[str, Dict]:
        """
        Rate Limit 상태 정보 반환

        Returns:
            API 타입별 상태 정보
        """
        status = {}

        for api_type, bucket in self.buckets.items():
            config = bucket.config
            status[api_type.value] = {
                "available_tokens": round(bucket.available_tokens, 2),
                "max_tokens": config.max_requests,
                "requests_per_second": config.requests_per_second,
                "is_full": bucket.is_full,
                "utilization": round(
                    (1 - bucket.available_tokens / config.max_requests) * 100, 1
                )
            }

        status["global"] = {
            "available_permits": self.global_semaphore._value,
            "max_permits": 50
        }

        return status

    async def wait_for_available_slot(
        self,
        endpoint: str,
        max_wait: float = 60.0
    ) -> bool:
        """
        사용 가능한 슬롯까지 대기

        Args:
            endpoint: API 엔드포인트
            max_wait: 최대 대기 시간 (초)

        Returns:
            대기 성공 여부
        """
        api_type = self._get_api_type(endpoint)
        bucket = self.buckets[api_type]

        start_time = time.time()

        while time.time() - start_time < max_wait:
            if self.can_make_request(endpoint):
                return True

            # 다음 토큰까지 예상 대기 시간
            tokens_needed = 1 - bucket.available_tokens
            if tokens_needed > 0:
                wait_time = tokens_needed / bucket.config.requests_per_second
                wait_time = min(wait_time, 1.0)  # 최대 1초씩 대기
            else:
                wait_time = 0.1

            await asyncio.sleep(wait_time)

        logger.warning(f"사용 가능한 슬롯 대기 타임아웃: {endpoint}")
        return False


# 전역 Rate Limiter 인스턴스
_global_rate_limiter: Optional[BithumbRateLimiter] = None


def get_rate_limiter() -> BithumbRateLimiter:
    """
    전역 Rate Limiter 인스턴스 반환

    Returns:
        BithumbRateLimiter 인스턴스
    """
    global _global_rate_limiter

    if _global_rate_limiter is None:
        _global_rate_limiter = BithumbRateLimiter()

    return _global_rate_limiter


def reset_rate_limiter(custom_configs: Optional[Dict[APIEndpointType, RateLimitConfig]] = None) -> None:
    """
    전역 Rate Limiter 리셋

    Args:
        custom_configs: 새로운 설정
    """
    global _global_rate_limiter
    _global_rate_limiter = BithumbRateLimiter(custom_configs)


# Rate Limiter 컨텍스트 매니저
class RateLimitedRequest:
    """Rate Limit이 적용된 요청 컨텍스트 매니저"""

    def __init__(self, endpoint: str, timeout: float = 30.0):
        """
        초기화

        Args:
            endpoint: API 엔드포인트
            timeout: 타임아웃
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.rate_limiter = get_rate_limiter()
        self.acquired = False

    async def __aenter__(self):
        """컨텍스트 진입"""
        self.acquired = await self.rate_limiter.acquire_permission(
            self.endpoint, self.timeout
        )

        if not self.acquired:
            raise RuntimeError(f"Rate limit 초과로 요청 실패: {self.endpoint}")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료"""
        if self.acquired:
            self.rate_limiter.release_permission(self.endpoint)