"""
WebSocket 자동 재연결 메커니즘 구현

네트워크 불안정이나 서버 문제로 인한 연결 끊김을 자동으로 복구하는
재연결 메커니즘을 제공합니다. 지수 백오프, 연결 상태 모니터링,
하트비트 등의 기능을 포함합니다.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class ReconnectStrategy(Enum):
    """재연결 전략"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_INTERVAL = "fixed_interval"
    LINEAR_BACKOFF = "linear_backoff"


class ReconnectState(Enum):
    """재연결 상태"""
    IDLE = "idle"
    RECONNECTING = "reconnecting"
    WAITING = "waiting"
    FAILED = "failed"
    DISABLED = "disabled"


class WebSocketReconnectManager:
    """
    WebSocket 자동 재연결 관리자

    연결 끊김 감지, 자동 재연결 시도, 백오프 전략 적용,
    하트비트 모니터링 등의 기능을 제공합니다.
    """

    def __init__(
        self,
        websocket_url: str,
        max_retry_attempts: int = 10,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 300.0,
        backoff_multiplier: float = 2.0,
        jitter_range: float = 0.2,
        strategy: ReconnectStrategy = ReconnectStrategy.EXPONENTIAL_BACKOFF,
        heartbeat_interval: float = 30.0,
        heartbeat_timeout: float = 10.0,
        connection_timeout: float = 10.0
    ):
        """
        재연결 관리자 초기화

        Args:
            websocket_url: WebSocket 서버 URL
            max_retry_attempts: 최대 재시도 횟수 (0은 무제한)
            initial_retry_delay: 초기 재시도 지연 시간 (초)
            max_retry_delay: 최대 재시도 지연 시간 (초)
            backoff_multiplier: 백오프 배율
            jitter_range: 지터 범위 (0.0 ~ 1.0)
            strategy: 재연결 전략
            heartbeat_interval: 하트비트 간격 (초)
            heartbeat_timeout: 하트비트 타임아웃 (초)
            connection_timeout: 연결 타임아웃 (초)
        """
        self._websocket_url = websocket_url
        self._max_retry_attempts = max_retry_attempts
        self._initial_retry_delay = initial_retry_delay
        self._max_retry_delay = max_retry_delay
        self._backoff_multiplier = backoff_multiplier
        self._jitter_range = jitter_range
        self._strategy = strategy
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._connection_timeout = connection_timeout

        # 상태 관리
        self._reconnect_state = ReconnectState.IDLE
        self._current_attempt = 0
        self._last_connection_time: Optional[datetime] = None
        self._last_disconnect_time: Optional[datetime] = None
        self._total_disconnects = 0
        self._total_reconnects = 0

        # 콜백 함수들
        self._on_connected: List[Callable] = []
        self._on_disconnected: List[Callable] = []
        self._on_reconnect_attempt: List[Callable] = []
        self._on_reconnect_failed: List[Callable] = []

        # 현재 WebSocket 연결
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None

        # 하트비트 관리
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_heartbeat: Optional[datetime] = None
        self._heartbeat_enabled = True

        # 재연결 태스크
        self._reconnect_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._should_reconnect = True

        # 통계 정보
        self._stats = {
            'total_connections': 0,
            'total_disconnections': 0,
            'total_reconnect_attempts': 0,
            'successful_reconnects': 0,
            'failed_reconnects': 0,
            'current_uptime': 0.0,
            'total_uptime': 0.0,
            'average_connection_duration': 0.0,
            'last_error': None,
            'connection_errors': []
        }

    @property
    def reconnect_state(self) -> ReconnectState:
        """현재 재연결 상태"""
        return self._reconnect_state

    @property
    def is_connected(self) -> bool:
        """연결 여부 확인"""
        return self._websocket is not None and not self._websocket.closed

    @property
    def current_attempt(self) -> int:
        """현재 재연결 시도 횟수"""
        return self._current_attempt

    @property
    def stats(self) -> Dict[str, Any]:
        """재연결 통계 정보"""
        stats = self._stats.copy()

        # 현재 업타임 계산
        if self._last_connection_time and self.is_connected:
            stats['current_uptime'] = (datetime.now() - self._last_connection_time).total_seconds()

        return stats

    def add_connection_callback(self, callback: Callable):
        """연결 성공 시 호출될 콜백 함수 추가"""
        self._on_connected.append(callback)

    def add_disconnection_callback(self, callback: Callable):
        """연결 끊김 시 호출될 콜백 함수 추가"""
        self._on_disconnected.append(callback)

    def add_reconnect_attempt_callback(self, callback: Callable):
        """재연결 시도 시 호출될 콜백 함수 추가"""
        self._on_reconnect_attempt.append(callback)

    def add_reconnect_failed_callback(self, callback: Callable):
        """재연결 실패 시 호출될 콜백 함수 추가"""
        self._on_reconnect_failed.append(callback)

    def _calculate_retry_delay(self) -> float:
        """재연결 지연 시간 계산"""
        if self._strategy == ReconnectStrategy.FIXED_INTERVAL:
            delay = self._initial_retry_delay
        elif self._strategy == ReconnectStrategy.LINEAR_BACKOFF:
            delay = self._initial_retry_delay * (self._current_attempt + 1)
        else:  # EXPONENTIAL_BACKOFF
            delay = self._initial_retry_delay * (self._backoff_multiplier ** self._current_attempt)

        # 최대 지연 시간 제한
        delay = min(delay, self._max_retry_delay)

        # 지터 추가 (동시 재연결 방지)
        if self._jitter_range > 0:
            jitter = delay * self._jitter_range * (2 * random.random() - 1)
            delay += jitter

        return max(0.1, delay)  # 최소 0.1초

    async def _execute_callbacks(self, callbacks: List[Callable], *args, **kwargs):
        """콜백 함수들을 비동기적으로 실행"""
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"콜백 실행 중 오류: {str(e)}")

    async def _create_connection(self) -> Optional[websockets.WebSocketServerProtocol]:
        """WebSocket 연결 생성"""
        try:
            logger.info(f"WebSocket 연결 시도: {self._websocket_url}")

            websocket = await asyncio.wait_for(
                websockets.connect(
                    self._websocket_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                    max_size=10**7,
                    max_queue=32,
                    compression=None
                ),
                timeout=self._connection_timeout
            )

            logger.info("WebSocket 연결 성공")
            return websocket

        except asyncio.TimeoutError:
            logger.error("WebSocket 연결 타임아웃")
            return None
        except Exception as e:
            logger.error(f"WebSocket 연결 실패: {str(e)}")
            return None

    async def _start_heartbeat(self):
        """하트비트 시작"""
        if not self._heartbeat_enabled:
            return

        async def heartbeat_worker():
            while self._is_running and self.is_connected:
                try:
                    # 하트비트 전송
                    if self._websocket:
                        pong_waiter = await self._websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=self._heartbeat_timeout)
                        self._last_heartbeat = datetime.now()

                    await asyncio.sleep(self._heartbeat_interval)

                except asyncio.TimeoutError:
                    logger.warning("하트비트 타임아웃")
                    await self._handle_disconnection("heartbeat_timeout")
                    break
                except Exception as e:
                    logger.error(f"하트비트 오류: {str(e)}")
                    await self._handle_disconnection("heartbeat_error")
                    break

        self._heartbeat_task = asyncio.create_task(heartbeat_worker())

    async def _stop_heartbeat(self):
        """하트비트 중지"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _handle_connection(self):
        """연결 성공 처리"""
        self._last_connection_time = datetime.now()
        self._current_attempt = 0
        self._reconnect_state = ReconnectState.IDLE

        # 통계 업데이트
        self._stats['total_connections'] += 1
        if self._current_attempt > 0:
            self._stats['successful_reconnects'] += 1

        # 하트비트 시작
        await self._start_heartbeat()

        # 연결 성공 콜백 실행
        await self._execute_callbacks(self._on_connected)

        logger.info("WebSocket 연결 완료")

    async def _handle_disconnection(self, reason: str = "unknown"):
        """연결 끊김 처리"""
        self._last_disconnect_time = datetime.now()
        self._total_disconnects += 1

        # 업타임 계산
        if self._last_connection_time:
            uptime = (self._last_disconnect_time - self._last_connection_time).total_seconds()
            self._stats['total_uptime'] += uptime

            # 평균 연결 지속 시간 계산
            if self._stats['total_connections'] > 0:
                self._stats['average_connection_duration'] = \
                    self._stats['total_uptime'] / self._stats['total_connections']

        # 통계 업데이트
        self._stats['total_disconnections'] += 1
        self._stats['last_error'] = reason
        self._stats['connection_errors'].append({
            'timestamp': self._last_disconnect_time,
            'reason': reason
        })

        # 하트비트 중지
        await self._stop_heartbeat()

        # 연결 끊김 콜백 실행
        await self._execute_callbacks(self._on_disconnected, reason)

        # WebSocket 정리
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

        logger.warning(f"WebSocket 연결 끊김: {reason}")

        # 재연결 시도
        if self._should_reconnect and self._is_running:
            await self._start_reconnect()

    async def _start_reconnect(self):
        """재연결 프로세스 시작"""
        if self._reconnect_state in (ReconnectState.RECONNECTING, ReconnectState.WAITING):
            logger.debug("이미 재연결 시도 중")
            return

        # 최대 재시도 횟수 확인
        if self._max_retry_attempts > 0 and self._current_attempt >= self._max_retry_attempts:
            logger.error(f"최대 재연결 시도 횟수 초과 ({self._max_retry_attempts})")
            self._reconnect_state = ReconnectState.FAILED
            await self._execute_callbacks(self._on_reconnect_failed)
            return

        self._reconnect_state = ReconnectState.RECONNECTING

        async def reconnect_worker():
            while self._should_reconnect and self._is_running:
                self._current_attempt += 1
                self._stats['total_reconnect_attempts'] += 1

                logger.info(f"재연결 시도 {self._current_attempt}/{self._max_retry_attempts}")

                # 재연결 시도 콜백 실행
                await self._execute_callbacks(self._on_reconnect_attempt, self._current_attempt)

                # 연결 시도
                websocket = await self._create_connection()
                if websocket:
                    self._websocket = websocket
                    await self._handle_connection()
                    return

                # 연결 실패 시 대기
                if self._current_attempt < self._max_retry_attempts or self._max_retry_attempts == 0:
                    delay = self._calculate_retry_delay()
                    self._reconnect_state = ReconnectState.WAITING

                    logger.info(f"다음 재연결 시도까지 {delay:.1f}초 대기")
                    await asyncio.sleep(delay)

                    self._reconnect_state = ReconnectState.RECONNECTING
                else:
                    # 최대 시도 횟수 도달
                    self._reconnect_state = ReconnectState.FAILED
                    self._stats['failed_reconnects'] += 1
                    await self._execute_callbacks(self._on_reconnect_failed)
                    break

        self._reconnect_task = asyncio.create_task(reconnect_worker())

    async def connect(self) -> bool:
        """초기 연결 수행"""
        if self.is_connected:
            logger.warning("이미 연결되어 있음")
            return True

        self._is_running = True
        self._should_reconnect = True
        self._current_attempt = 0

        websocket = await self._create_connection()
        if websocket:
            self._websocket = websocket
            await self._handle_connection()
            return True

        return False

    async def disconnect(self):
        """연결 종료"""
        self._should_reconnect = False
        self._is_running = False

        # 재연결 태스크 취소
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        # 하트비트 중지
        await self._stop_heartbeat()

        # WebSocket 연결 종료
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

        self._reconnect_state = ReconnectState.IDLE
        logger.info("WebSocket 연결 종료 완료")

    async def send_message(self, message: str) -> bool:
        """메시지 전송 (재연결 지원)"""
        if not self.is_connected:
            logger.error("WebSocket이 연결되지 않음")
            return False

        try:
            await self._websocket.send(message)
            return True
        except Exception as e:
            logger.error(f"메시지 전송 실패: {str(e)}")
            await self._handle_disconnection("send_error")
            return False

    async def receive_message(self) -> Optional[str]:
        """메시지 수신 (재연결 지원)"""
        if not self.is_connected:
            return None

        try:
            message = await self._websocket.recv()
            return message
        except ConnectionClosed:
            await self._handle_disconnection("connection_closed")
            return None
        except WebSocketException as e:
            await self._handle_disconnection(f"websocket_error: {str(e)}")
            return None
        except Exception as e:
            await self._handle_disconnection(f"receive_error: {str(e)}")
            return None

    def enable_heartbeat(self):
        """하트비트 활성화"""
        self._heartbeat_enabled = True

    def disable_heartbeat(self):
        """하트비트 비활성화"""
        self._heartbeat_enabled = False

    def set_max_retry_attempts(self, max_attempts: int):
        """최대 재시도 횟수 설정 (0은 무제한)"""
        self._max_retry_attempts = max_attempts

    def reset_retry_counter(self):
        """재시도 카운터 초기화"""
        self._current_attempt = 0
        if self._reconnect_state == ReconnectState.FAILED:
            self._reconnect_state = ReconnectState.IDLE

    async def health_check(self) -> Dict[str, Any]:
        """건강 상태 확인"""
        return {
            'connected': self.is_connected,
            'reconnect_state': self._reconnect_state.value,
            'current_attempt': self._current_attempt,
            'max_attempts': self._max_retry_attempts,
            'last_connection': self._last_connection_time.isoformat() if self._last_connection_time else None,
            'last_disconnect': self._last_disconnect_time.isoformat() if self._last_disconnect_time else None,
            'last_heartbeat': self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            'heartbeat_enabled': self._heartbeat_enabled,
            'stats': self.stats
        }