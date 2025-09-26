"""
데이터 스트림 처리기

빗썸 WebSocket의 Ticker, Orderbook, Trade 데이터 스트림을 처리하는 통합 시스템입니다.
앞서 구현한 WebSocket 클라이언트, 재연결, 파싱, 버퍼링, 백프레셔 시스템을 모두 통합합니다.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
import copy

from .websocket_client import BithumbWebSocketClient, SubscriptionType, get_websocket_client
from .message_parser import TickerData, OrderBookData, TransactionData, WebSocketMessage, MessageType
from .redis_buffer import RedisQueueBuffer, get_redis_buffer
from .backpressure_handler import BackpressureHandler, BackpressureLevel

logger = logging.getLogger(__name__)


@dataclass
class TickerSnapshot:
    """Ticker 데이터 스냅샷"""
    symbol: str
    timestamp: datetime

    # 가격 정보
    opening_price: Optional[Decimal] = None
    closing_price: Optional[Decimal] = None
    min_price: Optional[Decimal] = None
    max_price: Optional[Decimal] = None

    # 거래량 정보
    volume_24h: Optional[Decimal] = None
    volume_7d: Optional[Decimal] = None

    # 호가 정보
    best_bid: Optional[Decimal] = None
    best_ask: Optional[Decimal] = None

    # 변동률 정보
    price_change_24h: Optional[Decimal] = None
    price_change_rate_24h: Optional[Decimal] = None

    # 메타 정보
    last_updated: Optional[datetime] = None
    sequence_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화 가능)"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'opening_price': str(self.opening_price) if self.opening_price else None,
            'closing_price': str(self.closing_price) if self.closing_price else None,
            'min_price': str(self.min_price) if self.min_price else None,
            'max_price': str(self.max_price) if self.max_price else None,
            'volume_24h': str(self.volume_24h) if self.volume_24h else None,
            'volume_7d': str(self.volume_7d) if self.volume_7d else None,
            'best_bid': str(self.best_bid) if self.best_bid else None,
            'best_ask': str(self.best_ask) if self.best_ask else None,
            'price_change_24h': str(self.price_change_24h) if self.price_change_24h else None,
            'price_change_rate_24h': str(self.price_change_rate_24h) if self.price_change_rate_24h else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'sequence_number': self.sequence_number
        }


class StreamStatus(str, Enum):
    """스트림 상태"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class TickerStreamProcessor:
    """
    Ticker 데이터 스트림 처리기

    실시간 시세 데이터를 구독, 파싱, 캐싱, 버퍼링하는 완전한 처리 시스템입니다.
    """

    def __init__(
        self,
        symbols: List[str],
        websocket_client: Optional[BithumbWebSocketClient] = None,
        redis_buffer: Optional[RedisQueueBuffer] = None,
        backpressure_handler: Optional[BackpressureHandler] = None,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 300,
        buffer_queue_name: str = "ticker_stream"
    ):
        """
        Ticker 스트림 처리기 초기화

        Args:
            symbols: 구독할 심볼 리스트
            websocket_client: WebSocket 클라이언트 (None이면 전역 인스턴스 사용)
            redis_buffer: Redis 버퍼 (None이면 전역 인스턴스 사용)
            backpressure_handler: 백프레셔 핸들러 (None이면 자동 생성)
            enable_caching: 메모리 캐싱 활성화
            cache_ttl_seconds: 캐시 TTL (초)
            buffer_queue_name: 버퍼 큐 이름
        """
        self.symbols = [symbol.upper() for symbol in symbols]
        self._websocket_client = websocket_client
        self._redis_buffer = redis_buffer
        self._backpressure_handler = backpressure_handler
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.buffer_queue_name = buffer_queue_name

        # 상태 관리
        self._status = StreamStatus.STOPPED
        self._last_error: Optional[str] = None

        # 메모리 캐시
        self._ticker_cache: Dict[str, TickerSnapshot] = {}
        self._cache_lock = asyncio.Lock()

        # 통계
        self._stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_cached': 0,
            'messages_buffered': 0,
            'parse_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'backpressure_events': 0,
            'start_time': None,
            'last_message_time': None
        }

        # 콜백 함수들
        self._data_callbacks: List[Callable[[str, TickerSnapshot], None]] = []
        self._error_callbacks: List[Callable[[str, Exception], None]] = []

        # 백그라운드 태스크
        self._background_tasks: List[asyncio.Task] = []

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.stop()

    async def start(self):
        """스트림 처리기 시작"""
        if self._status != StreamStatus.STOPPED:
            logger.warning("스트림 처리기가 이미 실행 중입니다")
            return

        try:
            self._status = StreamStatus.STARTING
            logger.info(f"Ticker 스트림 처리기 시작: {self.symbols}")

            # 컴포넌트 초기화
            await self._initialize_components()

            # 백프레셔 핸들러 시작
            if self._backpressure_handler:
                await self._backpressure_handler.start()

            # WebSocket 연결 및 구독
            await self._setup_websocket_subscription()

            # 백그라운드 태스크 시작
            await self._start_background_tasks()

            self._status = StreamStatus.RUNNING
            self._stats['start_time'] = datetime.now()

            logger.info("Ticker 스트림 처리기 시작 완료")

        except Exception as e:
            self._status = StreamStatus.ERROR
            self._last_error = str(e)
            logger.error(f"Ticker 스트림 처리기 시작 실패: {str(e)}")
            raise

    async def stop(self):
        """스트림 처리기 중지"""
        if self._status == StreamStatus.STOPPED:
            return

        logger.info("Ticker 스트림 처리기 중지 중...")

        # 백그라운드 태스크 정리
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._background_tasks.clear()

        # 백프레셔 핸들러 중지
        if self._backpressure_handler:
            await self._backpressure_handler.stop()

        # WebSocket 구독 해제
        if self._websocket_client:
            try:
                await self._websocket_client.unsubscribe(SubscriptionType.TICKER, self.symbols)
            except Exception as e:
                logger.warning(f"구독 해제 실패: {str(e)}")

        self._status = StreamStatus.STOPPED
        logger.info("Ticker 스트림 처리기 중지 완료")

    async def _initialize_components(self):
        """컴포넌트 초기화"""
        # WebSocket 클라이언트
        if self._websocket_client is None:
            self._websocket_client = await get_websocket_client()

        # Redis 버퍼
        if self._redis_buffer is None:
            self._redis_buffer = await get_redis_buffer()

        # Redis 큐 생성
        await self._redis_buffer.create_queue(self.buffer_queue_name, max_size=50000)

        # 백프레셔 핸들러
        if self._backpressure_handler is None:
            self._backpressure_handler = BackpressureHandler(
                memory_threshold_mb=500.0,
                cpu_threshold_percent=80.0
            )

            # 백프레셔 콜백 등록
            self._backpressure_handler.add_level_change_callback(self._on_backpressure_change)

    async def _setup_websocket_subscription(self):
        """WebSocket 구독 설정"""
        if not self._websocket_client:
            raise RuntimeError("WebSocket 클라이언트가 초기화되지 않음")

        # 메시지 핸들러 등록
        self._websocket_client.add_message_handler(
            SubscriptionType.TICKER,
            self._handle_ticker_message
        )

        # WebSocket 연결 확인
        if not self._websocket_client.is_connected:
            success = await self._websocket_client.connect()
            if not success:
                raise RuntimeError("WebSocket 연결 실패")

        # Ticker 구독
        success = await self._websocket_client.subscribe(SubscriptionType.TICKER, self.symbols)
        if not success:
            raise RuntimeError("Ticker 구독 실패")

        logger.info(f"Ticker 구독 완료: {self.symbols}")

    async def _start_background_tasks(self):
        """백그라운드 태스크 시작"""
        # 캐시 정리 태스크
        if self.enable_caching:
            cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
            self._background_tasks.append(cache_cleanup_task)

        # 백프레셔 모니터링 태스크
        if self._backpressure_handler and self._redis_buffer:
            backpressure_monitor_task = asyncio.create_task(self._backpressure_monitor_loop())
            self._background_tasks.append(backpressure_monitor_task)

    async def _handle_ticker_message(self, ticker_data: TickerData, raw_message: WebSocketMessage):
        """
        Ticker 메시지 처리

        Args:
            ticker_data: 파싱된 Ticker 데이터
            raw_message: 원본 메시지
        """
        try:
            self._stats['messages_received'] += 1
            self._stats['last_message_time'] = datetime.now()

            # 백프레셔 상태 확인
            if self._backpressure_handler and self._backpressure_handler.should_drop_data():
                logger.debug("백프레셔로 인한 데이터 드롭")
                return

            # 스로틀링 적용
            if self._backpressure_handler and self._backpressure_handler.should_throttle():
                throttle_delay = await self._backpressure_handler.get_throttle_delay()
                if throttle_delay > 0:
                    await asyncio.sleep(throttle_delay)

            # Ticker 스냅샷 생성
            ticker_snapshot = await self._create_ticker_snapshot(ticker_data)

            # 메모리 캐싱
            if self.enable_caching:
                await self._update_cache(ticker_snapshot)

            # Redis 버퍼링
            if self._redis_buffer:
                await self._buffer_ticker_data(ticker_snapshot)

            # 백프레셔 메트릭 기록
            if self._backpressure_handler:
                self._backpressure_handler.record_ingestion()
                self._backpressure_handler.record_processing()

            # 콜백 실행
            await self._execute_data_callbacks(ticker_snapshot.symbol, ticker_snapshot)

            self._stats['messages_processed'] += 1

        except Exception as e:
            self._stats['parse_errors'] += 1
            logger.error(f"Ticker 메시지 처리 오류: {str(e)}")
            await self._execute_error_callbacks(ticker_data.symbol if ticker_data else "UNKNOWN", e)

    async def _create_ticker_snapshot(self, ticker_data: TickerData) -> TickerSnapshot:
        """Ticker 데이터로부터 스냅샷 생성"""
        now = datetime.now()

        # 기존 캐시에서 sequence number 가져오기
        sequence_number = 0
        if self.enable_caching and ticker_data.symbol in self._ticker_cache:
            sequence_number = self._ticker_cache[ticker_data.symbol].sequence_number + 1

        snapshot = TickerSnapshot(
            symbol=ticker_data.symbol,
            timestamp=now,
            opening_price=ticker_data.opening_price,
            closing_price=ticker_data.closing_price,
            min_price=ticker_data.min_price,
            max_price=ticker_data.max_price,
            volume_24h=ticker_data.volume_1day,
            volume_7d=ticker_data.volume_7day,
            best_bid=ticker_data.buy_price,
            best_ask=ticker_data.sell_price,
            price_change_24h=ticker_data.fluctate_24h,
            price_change_rate_24h=ticker_data.fluctate_rate_24h,
            last_updated=now,
            sequence_number=sequence_number
        )

        return snapshot

    async def _update_cache(self, ticker_snapshot: TickerSnapshot):
        """메모리 캐시 업데이트"""
        async with self._cache_lock:
            self._ticker_cache[ticker_snapshot.symbol] = ticker_snapshot
            self._stats['messages_cached'] += 1

    async def _buffer_ticker_data(self, ticker_snapshot: TickerSnapshot):
        """Redis 버퍼에 데이터 저장"""
        try:
            data_dict = ticker_snapshot.to_dict()
            success = await self._redis_buffer.enqueue(
                self.buffer_queue_name,
                data_dict,
                priority=1  # Ticker는 일반 우선순위
            )

            if success:
                self._stats['messages_buffered'] += 1
            else:
                logger.warning("Redis 버퍼링 실패")

        except Exception as e:
            logger.error(f"Redis 버퍼링 오류: {str(e)}")

    async def _cache_cleanup_loop(self):
        """캐시 정리 루프"""
        while self._status == StreamStatus.RUNNING:
            try:
                now = datetime.now()
                cutoff_time = now - timedelta(seconds=self.cache_ttl_seconds)

                async with self._cache_lock:
                    expired_symbols = [
                        symbol for symbol, snapshot in self._ticker_cache.items()
                        if snapshot.timestamp < cutoff_time
                    ]

                    for symbol in expired_symbols:
                        del self._ticker_cache[symbol]

                if expired_symbols:
                    logger.debug(f"만료된 캐시 제거: {len(expired_symbols)}개")

                await asyncio.sleep(60)  # 1분마다 정리

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"캐시 정리 오류: {str(e)}")
                await asyncio.sleep(10)

    async def _backpressure_monitor_loop(self):
        """백프레셔 모니터링 루프"""
        while self._status == StreamStatus.RUNNING:
            try:
                if self._redis_buffer and self._backpressure_handler:
                    queue_size = await self._redis_buffer.get_queue_size(self.buffer_queue_name)
                    queue_capacity = 50000  # 기본 큐 용량

                    # 백프레셔 핸들러에 큐 메트릭 업데이트
                    self._backpressure_handler.update_queue_metrics(queue_size, queue_capacity)

                await asyncio.sleep(1)  # 1초마다 모니터링

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"백프레셔 모니터링 오류: {str(e)}")
                await asyncio.sleep(5)

    async def _on_backpressure_change(self, old_level: BackpressureLevel, new_level: BackpressureLevel):
        """백프레셔 레벨 변경 콜백"""
        self._stats['backpressure_events'] += 1
        logger.info(f"백프레셔 레벨 변경: {old_level.value} -> {new_level.value}")

        if new_level in [BackpressureLevel.HIGH, BackpressureLevel.CRITICAL]:
            logger.warning(f"높은 백프레셔 감지: {new_level.value}")

    async def _execute_data_callbacks(self, symbol: str, ticker_snapshot: TickerSnapshot):
        """데이터 콜백 실행"""
        for callback in self._data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, ticker_snapshot)
                else:
                    callback(symbol, ticker_snapshot)
            except Exception as e:
                logger.error(f"데이터 콜백 실행 오류: {str(e)}")

    async def _execute_error_callbacks(self, symbol: str, error: Exception):
        """에러 콜백 실행"""
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, error)
                else:
                    callback(symbol, error)
            except Exception as e:
                logger.error(f"에러 콜백 실행 오류: {str(e)}")

    def add_data_callback(self, callback: Callable[[str, TickerSnapshot], None]):
        """데이터 수신 콜백 추가"""
        self._data_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """에러 콜백 추가"""
        self._error_callbacks.append(callback)

    async def get_latest_ticker(self, symbol: str) -> Optional[TickerSnapshot]:
        """최신 Ticker 데이터 조회"""
        symbol = symbol.upper()

        if not self.enable_caching:
            logger.warning("캐싱이 비활성화되어 있어 최신 데이터를 제공할 수 없습니다")
            return None

        async with self._cache_lock:
            if symbol in self._ticker_cache:
                self._stats['cache_hits'] += 1
                return copy.deepcopy(self._ticker_cache[symbol])
            else:
                self._stats['cache_misses'] += 1
                return None

    async def get_all_tickers(self) -> Dict[str, TickerSnapshot]:
        """모든 Ticker 데이터 조회"""
        if not self.enable_caching:
            return {}

        async with self._cache_lock:
            return {symbol: copy.deepcopy(snapshot)
                   for symbol, snapshot in self._ticker_cache.items()}

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = self._stats.copy()
        stats.update({
            'status': self._status.value,
            'subscribed_symbols': self.symbols,
            'cached_symbols': len(self._ticker_cache) if self.enable_caching else 0,
            'last_error': self._last_error,
            'uptime_seconds': (datetime.now() - self._stats['start_time']).total_seconds()
                             if self._stats['start_time'] else 0
        })

        if self._backpressure_handler:
            stats['backpressure'] = self._backpressure_handler.get_stats()

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """건강 상태 확인"""
        return {
            'status': self._status.value,
            'websocket_connected': self._websocket_client.is_connected if self._websocket_client else False,
            'redis_connected': await self._redis_buffer.health_check() if self._redis_buffer else {'status': 'unknown'},
            'subscribed_symbols_count': len(self.symbols),
            'cached_data_count': len(self._ticker_cache) if self.enable_caching else 0,
            'last_message_time': self._stats.get('last_message_time'),
            'total_messages_processed': self._stats.get('messages_processed', 0),
            'error_count': self._stats.get('parse_errors', 0)
        }

    @property
    def status(self) -> StreamStatus:
        """현재 상태"""
        return self._status

    @property
    def is_running(self) -> bool:
        """실행 중인지 확인"""
        return self._status == StreamStatus.RUNNING


@dataclass
class OrderBookSnapshot:
    """Orderbook 데이터 스냅샷"""
    symbol: str
    timestamp: datetime

    # 매수/매도 호가
    bids: List[Tuple[Decimal, Decimal]]  # [(가격, 수량), ...]
    asks: List[Tuple[Decimal, Decimal]]  # [(가격, 수량), ...]

    # 베스트 호가
    best_bid_price: Optional[Decimal] = None
    best_bid_qty: Optional[Decimal] = None
    best_ask_price: Optional[Decimal] = None
    best_ask_qty: Optional[Decimal] = None

    # 스프레드 정보
    spread_amount: Optional[Decimal] = None
    spread_rate: Optional[Decimal] = None

    # 메타 정보
    last_updated: Optional[datetime] = None
    sequence_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화 가능)"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'bids': [[str(price), str(qty)] for price, qty in self.bids],
            'asks': [[str(price), str(qty)] for price, qty in self.asks],
            'best_bid_price': str(self.best_bid_price) if self.best_bid_price else None,
            'best_bid_qty': str(self.best_bid_qty) if self.best_bid_qty else None,
            'best_ask_price': str(self.best_ask_price) if self.best_ask_price else None,
            'best_ask_qty': str(self.best_ask_qty) if self.best_ask_qty else None,
            'spread_amount': str(self.spread_amount) if self.spread_amount else None,
            'spread_rate': str(self.spread_rate) if self.spread_rate else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'sequence_number': self.sequence_number
        }


class OrderBookStreamProcessor:
    """
    Orderbook 데이터 스트림 처리기

    실시간 호가 데이터를 구독, 파싱, 캐싱, 버퍼링하는 완전한 처리 시스템입니다.
    """

    def __init__(
        self,
        symbols: List[str],
        websocket_client: Optional[BithumbWebSocketClient] = None,
        redis_buffer: Optional[RedisQueueBuffer] = None,
        backpressure_handler: Optional[BackpressureHandler] = None,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 60,  # Orderbook은 더 짧은 TTL
        buffer_queue_name: str = "orderbook_stream",
        max_depth: int = 20  # 최대 호가 깊이
    ):
        """
        Orderbook 스트림 처리기 초기화

        Args:
            symbols: 구독할 심볼 리스트
            websocket_client: WebSocket 클라이언트
            redis_buffer: Redis 버퍼
            backpressure_handler: 백프레셔 핸들러
            enable_caching: 메모리 캐싱 활성화
            cache_ttl_seconds: 캐시 TTL (초)
            buffer_queue_name: 버퍼 큐 이름
            max_depth: 최대 호가 깊이
        """
        self.symbols = [symbol.upper() for symbol in symbols]
        self._websocket_client = websocket_client
        self._redis_buffer = redis_buffer
        self._backpressure_handler = backpressure_handler
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.buffer_queue_name = buffer_queue_name
        self.max_depth = max_depth

        # 상태 관리
        self._status = StreamStatus.STOPPED
        self._last_error: Optional[str] = None

        # 메모리 캐시
        self._orderbook_cache: Dict[str, OrderBookSnapshot] = {}
        self._cache_lock = asyncio.Lock()

        # 통계
        self._stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_cached': 0,
            'messages_buffered': 0,
            'parse_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'backpressure_events': 0,
            'start_time': None,
            'last_message_time': None
        }

        # 콜백 함수들
        self._data_callbacks: List[Callable[[str, OrderBookSnapshot], None]] = []
        self._error_callbacks: List[Callable[[str, Exception], None]] = []

        # 백그라운드 태스크
        self._background_tasks: List[asyncio.Task] = []

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.stop()

    async def start(self):
        """스트림 처리기 시작"""
        if self._status != StreamStatus.STOPPED:
            logger.warning("Orderbook 스트림 처리기가 이미 실행 중입니다")
            return

        try:
            self._status = StreamStatus.STARTING
            logger.info(f"Orderbook 스트림 처리기 시작: {self.symbols}")

            # 컴포넌트 초기화
            await self._initialize_components()

            # 백프레셔 핸들러 시작
            if self._backpressure_handler:
                await self._backpressure_handler.start()

            # WebSocket 연결 및 구독
            await self._setup_websocket_subscription()

            # 백그라운드 태스크 시작
            await self._start_background_tasks()

            self._status = StreamStatus.RUNNING
            self._stats['start_time'] = datetime.now()

            logger.info("Orderbook 스트림 처리기 시작 완료")

        except Exception as e:
            self._status = StreamStatus.ERROR
            self._last_error = str(e)
            logger.error(f"Orderbook 스트림 처리기 시작 실패: {str(e)}")
            raise

    async def stop(self):
        """스트림 처리기 중지"""
        if self._status == StreamStatus.STOPPED:
            return

        logger.info("Orderbook 스트림 처리기 중지 중...")

        # 백그라운드 태스크 정리
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._background_tasks.clear()

        # 백프레셔 핸들러 중지
        if self._backpressure_handler:
            await self._backpressure_handler.stop()

        # WebSocket 구독 해제
        if self._websocket_client:
            try:
                await self._websocket_client.unsubscribe(SubscriptionType.ORDERBOOK, self.symbols)
            except Exception as e:
                logger.warning(f"구독 해제 실패: {str(e)}")

        self._status = StreamStatus.STOPPED
        logger.info("Orderbook 스트림 처리기 중지 완료")

    async def _initialize_components(self):
        """컴포넌트 초기화"""
        # WebSocket 클라이언트
        if self._websocket_client is None:
            self._websocket_client = await get_websocket_client()

        # Redis 버퍼
        if self._redis_buffer is None:
            self._redis_buffer = await get_redis_buffer()

        # Redis 큐 생성
        await self._redis_buffer.create_queue(self.buffer_queue_name, max_size=30000)

        # 백프레셔 핸들러
        if self._backpressure_handler is None:
            self._backpressure_handler = BackpressureHandler(
                memory_threshold_mb=500.0,
                cpu_threshold_percent=80.0
            )

            # 백프레셔 콜백 등록
            self._backpressure_handler.add_level_change_callback(self._on_backpressure_change)

    async def _setup_websocket_subscription(self):
        """WebSocket 구독 설정"""
        if not self._websocket_client:
            raise RuntimeError("WebSocket 클라이언트가 초기화되지 않음")

        # 메시지 핸들러 등록
        self._websocket_client.add_message_handler(
            SubscriptionType.ORDERBOOK,
            self._handle_orderbook_message
        )

        # WebSocket 연결 확인
        if not self._websocket_client.is_connected:
            success = await self._websocket_client.connect()
            if not success:
                raise RuntimeError("WebSocket 연결 실패")

        # Orderbook 구독
        success = await self._websocket_client.subscribe(SubscriptionType.ORDERBOOK, self.symbols)
        if not success:
            raise RuntimeError("Orderbook 구독 실패")

        logger.info(f"Orderbook 구독 완료: {self.symbols}")

    async def _start_background_tasks(self):
        """백그라운드 태스크 시작"""
        # 캐시 정리 태스크
        if self.enable_caching:
            cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
            self._background_tasks.append(cache_cleanup_task)

        # 백프레셔 모니터링 태스크
        if self._backpressure_handler and self._redis_buffer:
            backpressure_monitor_task = asyncio.create_task(self._backpressure_monitor_loop())
            self._background_tasks.append(backpressure_monitor_task)

    async def _handle_orderbook_message(self, orderbook_data: OrderBookData, raw_message: WebSocketMessage):
        """
        Orderbook 메시지 처리

        Args:
            orderbook_data: 파싱된 Orderbook 데이터
            raw_message: 원본 메시지
        """
        try:
            self._stats['messages_received'] += 1
            self._stats['last_message_time'] = datetime.now()

            # 백프레셔 상태 확인
            if self._backpressure_handler and self._backpressure_handler.should_drop_data():
                logger.debug("백프레셔로 인한 데이터 드롭")
                return

            # 스로틀링 적용
            if self._backpressure_handler and self._backpressure_handler.should_throttle():
                throttle_delay = await self._backpressure_handler.get_throttle_delay()
                if throttle_delay > 0:
                    await asyncio.sleep(throttle_delay)

            # Orderbook 스냅샷 생성
            orderbook_snapshot = await self._create_orderbook_snapshot(orderbook_data)

            # 메모리 캐싱
            if self.enable_caching:
                await self._update_cache(orderbook_snapshot)

            # Redis 버퍼링
            if self._redis_buffer:
                await self._buffer_orderbook_data(orderbook_snapshot)

            # 백프레셔 메트릭 기록
            if self._backpressure_handler:
                self._backpressure_handler.record_ingestion()
                self._backpressure_handler.record_processing()

            # 콜백 실행
            await self._execute_data_callbacks(orderbook_snapshot.symbol, orderbook_snapshot)

            self._stats['messages_processed'] += 1

        except Exception as e:
            self._stats['parse_errors'] += 1
            logger.error(f"Orderbook 메시지 처리 오류: {str(e)}")
            await self._execute_error_callbacks(orderbook_data.symbol if orderbook_data else "UNKNOWN", e)

    async def _create_orderbook_snapshot(self, orderbook_data: OrderBookData) -> OrderBookSnapshot:
        """Orderbook 데이터로부터 스냅샷 생성"""
        now = datetime.now()

        # 기존 캐시에서 sequence number 가져오기
        sequence_number = 0
        if self.enable_caching and orderbook_data.symbol in self._orderbook_cache:
            sequence_number = self._orderbook_cache[orderbook_data.symbol].sequence_number + 1

        # 호가 데이터 변환 및 제한
        bids = [(entry.price, entry.quantity) for entry in orderbook_data.bids[:self.max_depth]]
        asks = [(entry.price, entry.quantity) for entry in orderbook_data.asks[:self.max_depth]]

        # 베스트 호가 계산
        best_bid_price = bids[0][0] if bids else None
        best_bid_qty = bids[0][1] if bids else None
        best_ask_price = asks[0][0] if asks else None
        best_ask_qty = asks[0][1] if asks else None

        # 스프레드 계산
        spread_amount = None
        spread_rate = None
        if best_bid_price and best_ask_price:
            spread_amount = best_ask_price - best_bid_price
            spread_rate = (spread_amount / best_bid_price) * 100

        snapshot = OrderBookSnapshot(
            symbol=orderbook_data.symbol,
            timestamp=now,
            bids=bids,
            asks=asks,
            best_bid_price=best_bid_price,
            best_bid_qty=best_bid_qty,
            best_ask_price=best_ask_price,
            best_ask_qty=best_ask_qty,
            spread_amount=spread_amount,
            spread_rate=spread_rate,
            last_updated=now,
            sequence_number=sequence_number
        )

        return snapshot

    async def _update_cache(self, orderbook_snapshot: OrderBookSnapshot):
        """메모리 캐시 업데이트"""
        async with self._cache_lock:
            self._orderbook_cache[orderbook_snapshot.symbol] = orderbook_snapshot
            self._stats['messages_cached'] += 1

    async def _buffer_orderbook_data(self, orderbook_snapshot: OrderBookSnapshot):
        """Redis 버퍼에 데이터 저장"""
        try:
            data_dict = orderbook_snapshot.to_dict()
            success = await self._redis_buffer.enqueue(
                self.buffer_queue_name,
                data_dict,
                priority=2  # Orderbook은 높은 우선순위
            )

            if success:
                self._stats['messages_buffered'] += 1
            else:
                logger.warning("Redis 버퍼링 실패")

        except Exception as e:
            logger.error(f"Redis 버퍼링 오류: {str(e)}")

    async def _cache_cleanup_loop(self):
        """캐시 정리 루프"""
        while self._status == StreamStatus.RUNNING:
            try:
                now = datetime.now()
                cutoff_time = now - timedelta(seconds=self.cache_ttl_seconds)

                async with self._cache_lock:
                    expired_symbols = [
                        symbol for symbol, snapshot in self._orderbook_cache.items()
                        if snapshot.timestamp < cutoff_time
                    ]

                    for symbol in expired_symbols:
                        del self._orderbook_cache[symbol]

                if expired_symbols:
                    logger.debug(f"만료된 Orderbook 캐시 제거: {len(expired_symbols)}개")

                await asyncio.sleep(30)  # 30초마다 정리

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"캐시 정리 오류: {str(e)}")
                await asyncio.sleep(10)

    async def _backpressure_monitor_loop(self):
        """백프레셔 모니터링 루프"""
        while self._status == StreamStatus.RUNNING:
            try:
                if self._redis_buffer and self._backpressure_handler:
                    queue_size = await self._redis_buffer.get_queue_size(self.buffer_queue_name)
                    queue_capacity = 30000

                    # 백프레셔 핸들러에 큐 메트릭 업데이트
                    self._backpressure_handler.update_queue_metrics(queue_size, queue_capacity)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"백프레셔 모니터링 오류: {str(e)}")
                await asyncio.sleep(5)

    async def _on_backpressure_change(self, old_level: BackpressureLevel, new_level: BackpressureLevel):
        """백프레셔 레벨 변경 콜백"""
        self._stats['backpressure_events'] += 1
        logger.info(f"Orderbook 백프레셔 레벨 변경: {old_level.value} -> {new_level.value}")

    async def _execute_data_callbacks(self, symbol: str, orderbook_snapshot: OrderBookSnapshot):
        """데이터 콜백 실행"""
        for callback in self._data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, orderbook_snapshot)
                else:
                    callback(symbol, orderbook_snapshot)
            except Exception as e:
                logger.error(f"데이터 콜백 실행 오류: {str(e)}")

    async def _execute_error_callbacks(self, symbol: str, error: Exception):
        """에러 콜백 실행"""
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, error)
                else:
                    callback(symbol, error)
            except Exception as e:
                logger.error(f"에러 콜백 실행 오류: {str(e)}")

    def add_data_callback(self, callback: Callable[[str, OrderBookSnapshot], None]):
        """데이터 수신 콜백 추가"""
        self._data_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """에러 콜백 추가"""
        self._error_callbacks.append(callback)

    async def get_latest_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """최신 Orderbook 데이터 조회"""
        symbol = symbol.upper()

        if not self.enable_caching:
            logger.warning("캐싱이 비활성화되어 있어 최신 데이터를 제공할 수 없습니다")
            return None

        async with self._cache_lock:
            if symbol in self._orderbook_cache:
                self._stats['cache_hits'] += 1
                return copy.deepcopy(self._orderbook_cache[symbol])
            else:
                self._stats['cache_misses'] += 1
                return None

    async def get_all_orderbooks(self) -> Dict[str, OrderBookSnapshot]:
        """모든 Orderbook 데이터 조회"""
        if not self.enable_caching:
            return {}

        async with self._cache_lock:
            return {symbol: copy.deepcopy(snapshot)
                   for symbol, snapshot in self._orderbook_cache.items()}

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = self._stats.copy()
        stats.update({
            'status': self._status.value,
            'subscribed_symbols': self.symbols,
            'cached_symbols': len(self._orderbook_cache) if self.enable_caching else 0,
            'last_error': self._last_error,
            'uptime_seconds': (datetime.now() - self._stats['start_time']).total_seconds()
                             if self._stats['start_time'] else 0
        })

        if self._backpressure_handler:
            stats['backpressure'] = self._backpressure_handler.get_stats()

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """건강 상태 확인"""
        return {
            'status': self._status.value,
            'websocket_connected': self._websocket_client.is_connected if self._websocket_client else False,
            'redis_connected': await self._redis_buffer.health_check() if self._redis_buffer else {'status': 'unknown'},
            'subscribed_symbols_count': len(self.symbols),
            'cached_data_count': len(self._orderbook_cache) if self.enable_caching else 0,
            'last_message_time': self._stats.get('last_message_time'),
            'total_messages_processed': self._stats.get('messages_processed', 0),
            'error_count': self._stats.get('parse_errors', 0)
        }

    @property
    def status(self) -> StreamStatus:
        """현재 상태"""
        return self._status

    @property
    def is_running(self) -> bool:
        """실행 중인지 확인"""
        return self._status == StreamStatus.RUNNING


@dataclass
class TradeSnapshot:
    """Trade 데이터 스냅샷"""
    symbol: str
    timestamp: datetime

    # 체결 정보
    price: Decimal
    quantity: Decimal
    trade_type: str  # "bid" 또는 "ask"

    # 누적 정보 (최근 1시간 기준)
    total_volume_1h: Optional[Decimal] = None
    total_trades_1h: int = 0
    avg_price_1h: Optional[Decimal] = None

    # 메타 정보
    trade_id: Optional[str] = None
    last_updated: Optional[datetime] = None
    sequence_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화 가능)"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': str(self.price),
            'quantity': str(self.quantity),
            'trade_type': self.trade_type,
            'total_volume_1h': str(self.total_volume_1h) if self.total_volume_1h else None,
            'total_trades_1h': self.total_trades_1h,
            'avg_price_1h': str(self.avg_price_1h) if self.avg_price_1h else None,
            'trade_id': self.trade_id,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'sequence_number': self.sequence_number
        }


class TradeStreamProcessor:
    """
    Trade 데이터 스트림 처리기

    실시간 체결 데이터를 구독, 파싱, 캐싱, 버퍼링하는 완전한 처리 시스템입니다.
    중복 체결 데이터 필터링 및 시간 기반 정렬을 제공합니다.
    """

    def __init__(
        self,
        symbols: List[str],
        websocket_client: Optional[BithumbWebSocketClient] = None,
        redis_buffer: Optional[RedisQueueBuffer] = None,
        backpressure_handler: Optional[BackpressureHandler] = None,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 120,  # Trade는 중간 길이 TTL
        buffer_queue_name: str = "trade_stream",
        max_trades_cache: int = 100  # 심볼당 최대 캐시 체결 수
    ):
        """
        Trade 스트림 처리기 초기화

        Args:
            symbols: 구독할 심볼 리스트
            websocket_client: WebSocket 클라이언트
            redis_buffer: Redis 버퍼
            backpressure_handler: 백프레셔 핸들러
            enable_caching: 메모리 캐싱 활성화
            cache_ttl_seconds: 캐시 TTL (초)
            buffer_queue_name: 버퍼 큐 이름
            max_trades_cache: 심볼당 최대 캐시 체결 수
        """
        self.symbols = [symbol.upper() for symbol in symbols]
        self._websocket_client = websocket_client
        self._redis_buffer = redis_buffer
        self._backpressure_handler = backpressure_handler
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.buffer_queue_name = buffer_queue_name
        self.max_trades_cache = max_trades_cache

        # 상태 관리
        self._status = StreamStatus.STOPPED
        self._last_error: Optional[str] = None

        # 메모리 캐시 (심볼별로 최근 거래 리스트 저장)
        self._trade_cache: Dict[str, List[TradeSnapshot]] = {}
        self._cache_lock = asyncio.Lock()

        # 중복 체결 필터링을 위한 trade_id 세트
        self._processed_trade_ids: Dict[str, Set[str]] = {}

        # 통계
        self._stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_cached': 0,
            'messages_buffered': 0,
            'duplicate_trades_filtered': 0,
            'parse_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'backpressure_events': 0,
            'start_time': None,
            'last_message_time': None
        }

        # 콜백 함수들
        self._data_callbacks: List[Callable[[str, TradeSnapshot], None]] = []
        self._error_callbacks: List[Callable[[str, Exception], None]] = []

        # 백그라운드 태스크
        self._background_tasks: List[asyncio.Task] = []

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.stop()

    async def start(self):
        """스트림 처리기 시작"""
        if self._status != StreamStatus.STOPPED:
            logger.warning("Trade 스트림 처리기가 이미 실행 중입니다")
            return

        try:
            self._status = StreamStatus.STARTING
            logger.info(f"Trade 스트림 처리기 시작: {self.symbols}")

            # 컴포넌트 초기화
            await self._initialize_components()

            # 백프레셔 핸들러 시작
            if self._backpressure_handler:
                await self._backpressure_handler.start()

            # WebSocket 연결 및 구독
            await self._setup_websocket_subscription()

            # 백그라운드 태스크 시작
            await self._start_background_tasks()

            self._status = StreamStatus.RUNNING
            self._stats['start_time'] = datetime.now()

            logger.info("Trade 스트림 처리기 시작 완료")

        except Exception as e:
            self._status = StreamStatus.ERROR
            self._last_error = str(e)
            logger.error(f"Trade 스트림 처리기 시작 실패: {str(e)}")
            raise

    async def stop(self):
        """스트림 처리기 중지"""
        if self._status == StreamStatus.STOPPED:
            return

        logger.info("Trade 스트림 처리기 중지 중...")

        # 백그라운드 태스크 정리
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._background_tasks.clear()

        # 백프레셔 핸들러 중지
        if self._backpressure_handler:
            await self._backpressure_handler.stop()

        # WebSocket 구독 해제
        if self._websocket_client:
            try:
                await self._websocket_client.unsubscribe(SubscriptionType.TRANSACTION, self.symbols)
            except Exception as e:
                logger.warning(f"구독 해제 실패: {str(e)}")

        self._status = StreamStatus.STOPPED
        logger.info("Trade 스트림 처리기 중지 완료")

    async def _initialize_components(self):
        """컴포넌트 초기화"""
        # WebSocket 클라이언트
        if self._websocket_client is None:
            self._websocket_client = await get_websocket_client()

        # Redis 버퍼
        if self._redis_buffer is None:
            self._redis_buffer = await get_redis_buffer()

        # Redis 큐 생성
        await self._redis_buffer.create_queue(self.buffer_queue_name, max_size=40000)

        # 백프레셔 핸들러
        if self._backpressure_handler is None:
            self._backpressure_handler = BackpressureHandler(
                memory_threshold_mb=500.0,
                cpu_threshold_percent=80.0
            )

            # 백프레셔 콜백 등록
            self._backpressure_handler.add_level_change_callback(self._on_backpressure_change)

    async def _setup_websocket_subscription(self):
        """WebSocket 구독 설정"""
        if not self._websocket_client:
            raise RuntimeError("WebSocket 클라이언트가 초기화되지 않음")

        # 메시지 핸들러 등록
        self._websocket_client.add_message_handler(
            SubscriptionType.TRANSACTION,
            self._handle_trade_message
        )

        # WebSocket 연결 확인
        if not self._websocket_client.is_connected:
            success = await self._websocket_client.connect()
            if not success:
                raise RuntimeError("WebSocket 연결 실패")

        # Trade 구독
        success = await self._websocket_client.subscribe(SubscriptionType.TRANSACTION, self.symbols)
        if not success:
            raise RuntimeError("Trade 구독 실패")

        logger.info(f"Trade 구독 완료: {self.symbols}")

    async def _start_background_tasks(self):
        """백그라운드 태스크 시작"""
        # 캐시 정리 태스크
        if self.enable_caching:
            cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
            self._background_tasks.append(cache_cleanup_task)

        # 백프레셔 모니터링 태스크
        if self._backpressure_handler and self._redis_buffer:
            backpressure_monitor_task = asyncio.create_task(self._backpressure_monitor_loop())
            self._background_tasks.append(backpressure_monitor_task)

        # trade_id 정리 태스크
        trade_id_cleanup_task = asyncio.create_task(self._trade_id_cleanup_loop())
        self._background_tasks.append(trade_id_cleanup_task)

    async def _handle_trade_message(self, trade_data: TransactionData, raw_message: WebSocketMessage):
        """
        Trade 메시지 처리

        Args:
            trade_data: 파싱된 Trade 데이터
            raw_message: 원본 메시지
        """
        try:
            self._stats['messages_received'] += 1
            self._stats['last_message_time'] = datetime.now()

            # 백프레셔 상태 확인
            if self._backpressure_handler and self._backpressure_handler.should_drop_data():
                logger.debug("백프레셔로 인한 데이터 드롭")
                return

            # 중복 체결 필터링
            if await self._is_duplicate_trade(trade_data):
                self._stats['duplicate_trades_filtered'] += 1
                logger.debug(f"중복 체결 데이터 필터링: {trade_data.symbol}")
                return

            # 스로틀링 적용
            if self._backpressure_handler and self._backpressure_handler.should_throttle():
                throttle_delay = await self._backpressure_handler.get_throttle_delay()
                if throttle_delay > 0:
                    await asyncio.sleep(throttle_delay)

            # Trade 스냅샷 생성
            trade_snapshot = await self._create_trade_snapshot(trade_data)

            # 메모리 캐싱
            if self.enable_caching:
                await self._update_cache(trade_snapshot)

            # Redis 버퍼링
            if self._redis_buffer:
                await self._buffer_trade_data(trade_snapshot)

            # 백프레셔 메트릭 기록
            if self._backpressure_handler:
                self._backpressure_handler.record_ingestion()
                self._backpressure_handler.record_processing()

            # 콜백 실행
            await self._execute_data_callbacks(trade_snapshot.symbol, trade_snapshot)

            self._stats['messages_processed'] += 1

        except Exception as e:
            self._stats['parse_errors'] += 1
            logger.error(f"Trade 메시지 처리 오류: {str(e)}")
            await self._execute_error_callbacks(trade_data.symbol if trade_data else "UNKNOWN", e)

    async def _is_duplicate_trade(self, trade_data: TransactionData) -> bool:
        """중복 체결 데이터 확인"""
        symbol = trade_data.symbol

        # trade_id가 없으면 중복 체크 불가
        if not hasattr(trade_data, 'trade_id') or not trade_data.trade_id:
            return False

        # 심볼별 processed_trade_ids 초기화
        if symbol not in self._processed_trade_ids:
            self._processed_trade_ids[symbol] = set()

        # 이미 처리된 trade_id인지 확인
        if trade_data.trade_id in self._processed_trade_ids[symbol]:
            return True

        # trade_id 추가 (최대 1000개 유지)
        self._processed_trade_ids[symbol].add(trade_data.trade_id)
        if len(self._processed_trade_ids[symbol]) > 1000:
            # 오래된 trade_id 제거 (FIFO)
            old_ids = list(self._processed_trade_ids[symbol])[:100]
            for old_id in old_ids:
                self._processed_trade_ids[symbol].discard(old_id)

        return False

    async def _create_trade_snapshot(self, trade_data: TransactionData) -> TradeSnapshot:
        """Trade 데이터로부터 스냅샷 생성"""
        now = datetime.now()

        # 기존 캐시에서 sequence number 가져오기
        sequence_number = 0
        if self.enable_caching and trade_data.symbol in self._trade_cache:
            last_trade = self._trade_cache[trade_data.symbol][-1] if self._trade_cache[trade_data.symbol] else None
            if last_trade:
                sequence_number = last_trade.sequence_number + 1

        # 1시간 집계 정보 계산 (캐시에서)
        total_volume_1h = None
        total_trades_1h = 0
        avg_price_1h = None

        if self.enable_caching and trade_data.symbol in self._trade_cache:
            one_hour_ago = now - timedelta(hours=1)
            recent_trades = [
                trade for trade in self._trade_cache[trade_data.symbol]
                if trade.timestamp >= one_hour_ago
            ]

            if recent_trades:
                total_volume_1h = sum(trade.quantity for trade in recent_trades)
                total_trades_1h = len(recent_trades)
                total_value = sum(trade.price * trade.quantity for trade in recent_trades)
                avg_price_1h = total_value / total_volume_1h if total_volume_1h > 0 else None

        snapshot = TradeSnapshot(
            symbol=trade_data.symbol,
            timestamp=now,
            price=trade_data.price,
            quantity=trade_data.quantity,
            trade_type=trade_data.trade_type,
            total_volume_1h=total_volume_1h,
            total_trades_1h=total_trades_1h,
            avg_price_1h=avg_price_1h,
            trade_id=getattr(trade_data, 'trade_id', None),
            last_updated=now,
            sequence_number=sequence_number
        )

        return snapshot

    async def _update_cache(self, trade_snapshot: TradeSnapshot):
        """메모리 캐시 업데이트"""
        async with self._cache_lock:
            symbol = trade_snapshot.symbol

            # 심볼별 캐시 초기화
            if symbol not in self._trade_cache:
                self._trade_cache[symbol] = []

            # 새 체결 데이터 추가
            self._trade_cache[symbol].append(trade_snapshot)

            # 최대 캐시 크기 유지 (FIFO)
            if len(self._trade_cache[symbol]) > self.max_trades_cache:
                self._trade_cache[symbol] = self._trade_cache[symbol][-self.max_trades_cache:]

            # 시간 순 정렬 보장
            self._trade_cache[symbol].sort(key=lambda x: x.timestamp, reverse=True)

            self._stats['messages_cached'] += 1

    async def _buffer_trade_data(self, trade_snapshot: TradeSnapshot):
        """Redis 버퍼에 데이터 저장"""
        try:
            data_dict = trade_snapshot.to_dict()
            success = await self._redis_buffer.enqueue(
                self.buffer_queue_name,
                data_dict,
                priority=3  # Trade는 최고 우선순위
            )

            if success:
                self._stats['messages_buffered'] += 1
            else:
                logger.warning("Redis 버퍼링 실패")

        except Exception as e:
            logger.error(f"Redis 버퍼링 오류: {str(e)}")

    async def _cache_cleanup_loop(self):
        """캐시 정리 루프"""
        while self._status == StreamStatus.RUNNING:
            try:
                now = datetime.now()
                cutoff_time = now - timedelta(seconds=self.cache_ttl_seconds)

                async with self._cache_lock:
                    for symbol in list(self._trade_cache.keys()):
                        # 만료된 체결 데이터 제거
                        self._trade_cache[symbol] = [
                            trade for trade in self._trade_cache[symbol]
                            if trade.timestamp >= cutoff_time
                        ]

                        # 빈 리스트면 제거
                        if not self._trade_cache[symbol]:
                            del self._trade_cache[symbol]

                await asyncio.sleep(45)  # 45초마다 정리

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"캐시 정리 오류: {str(e)}")
                await asyncio.sleep(10)

    async def _trade_id_cleanup_loop(self):
        """trade_id 정리 루프"""
        while self._status == StreamStatus.RUNNING:
            try:
                # processed_trade_ids 크기 제한 (심볼당 최대 500개 유지)
                for symbol in list(self._processed_trade_ids.keys()):
                    if len(self._processed_trade_ids[symbol]) > 500:
                        # 절반을 제거
                        trade_ids = list(self._processed_trade_ids[symbol])
                        keep_ids = trade_ids[-250:]  # 최신 250개 유지
                        self._processed_trade_ids[symbol] = set(keep_ids)

                await asyncio.sleep(300)  # 5분마다 정리

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trade ID 정리 오류: {str(e)}")
                await asyncio.sleep(30)

    async def _backpressure_monitor_loop(self):
        """백프레셔 모니터링 루프"""
        while self._status == StreamStatus.RUNNING:
            try:
                if self._redis_buffer and self._backpressure_handler:
                    queue_size = await self._redis_buffer.get_queue_size(self.buffer_queue_name)
                    queue_capacity = 40000

                    # 백프레셔 핸들러에 큐 메트릭 업데이트
                    self._backpressure_handler.update_queue_metrics(queue_size, queue_capacity)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"백프레셔 모니터링 오류: {str(e)}")
                await asyncio.sleep(5)

    async def _on_backpressure_change(self, old_level: BackpressureLevel, new_level: BackpressureLevel):
        """백프레셔 레벨 변경 콜백"""
        self._stats['backpressure_events'] += 1
        logger.info(f"Trade 백프레셔 레벨 변경: {old_level.value} -> {new_level.value}")

    async def _execute_data_callbacks(self, symbol: str, trade_snapshot: TradeSnapshot):
        """데이터 콜백 실행"""
        for callback in self._data_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, trade_snapshot)
                else:
                    callback(symbol, trade_snapshot)
            except Exception as e:
                logger.error(f"데이터 콜백 실행 오류: {str(e)}")

    async def _execute_error_callbacks(self, symbol: str, error: Exception):
        """에러 콜백 실행"""
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, error)
                else:
                    callback(symbol, error)
            except Exception as e:
                logger.error(f"에러 콜백 실행 오류: {str(e)}")

    def add_data_callback(self, callback: Callable[[str, TradeSnapshot], None]):
        """데이터 수신 콜백 추가"""
        self._data_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """에러 콜백 추가"""
        self._error_callbacks.append(callback)

    async def get_recent_trades(self, symbol: str, limit: int = 10) -> List[TradeSnapshot]:
        """최근 체결 데이터 조회"""
        symbol = symbol.upper()

        if not self.enable_caching:
            logger.warning("캐싱이 비활성화되어 있어 최신 데이터를 제공할 수 없습니다")
            return []

        async with self._cache_lock:
            if symbol in self._trade_cache:
                self._stats['cache_hits'] += 1
                return copy.deepcopy(self._trade_cache[symbol][:limit])
            else:
                self._stats['cache_misses'] += 1
                return []

    async def get_trade_volume_1h(self, symbol: str) -> Optional[Decimal]:
        """1시간 체결량 조회"""
        symbol = symbol.upper()

        if not self.enable_caching or symbol not in self._trade_cache:
            return None

        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)

        async with self._cache_lock:
            recent_trades = [
                trade for trade in self._trade_cache[symbol]
                if trade.timestamp >= one_hour_ago
            ]

            if not recent_trades:
                return None

            return sum(trade.quantity for trade in recent_trades)

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = self._stats.copy()
        stats.update({
            'status': self._status.value,
            'subscribed_symbols': self.symbols,
            'cached_symbols': len(self._trade_cache) if self.enable_caching else 0,
            'processed_trade_ids_count': sum(len(trade_ids) for trade_ids in self._processed_trade_ids.values()),
            'last_error': self._last_error,
            'uptime_seconds': (datetime.now() - self._stats['start_time']).total_seconds()
                             if self._stats['start_time'] else 0
        })

        if self._backpressure_handler:
            stats['backpressure'] = self._backpressure_handler.get_stats()

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """건강 상태 확인"""
        return {
            'status': self._status.value,
            'websocket_connected': self._websocket_client.is_connected if self._websocket_client else False,
            'redis_connected': await self._redis_buffer.health_check() if self._redis_buffer else {'status': 'unknown'},
            'subscribed_symbols_count': len(self.symbols),
            'cached_trades_count': sum(len(trades) for trades in self._trade_cache.values()) if self.enable_caching else 0,
            'duplicate_filter_count': sum(len(trade_ids) for trade_ids in self._processed_trade_ids.values()),
            'last_message_time': self._stats.get('last_message_time'),
            'total_messages_processed': self._stats.get('messages_processed', 0),
            'error_count': self._stats.get('parse_errors', 0)
        }

    @property
    def status(self) -> StreamStatus:
        """현재 상태"""
        return self._status

    @property
    def is_running(self) -> bool:
        """실행 중인지 확인"""
        return self._status == StreamStatus.RUNNING