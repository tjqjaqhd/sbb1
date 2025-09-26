"""
빗썸 WebSocket 클라이언트 구현

실시간 시세, 호가, 체결 데이터를 수집하는 WebSocket 클라이언트입니다.
websockets 라이브러리 기반의 비동기 WebSocket 연결을 관리합니다.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from .websocket_reconnect import WebSocketReconnectManager, ReconnectStrategy
from .message_parser import (
    MessageParser, WebSocketMessage, MessageType, TickerData,
    OrderBookData, TransactionData, MessageParsingError, get_message_parser
)

logger = logging.getLogger(__name__)


class SubscriptionType(Enum):
    """WebSocket 구독 데이터 타입"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRANSACTION = "transaction"


class ConnectionState(Enum):
    """WebSocket 연결 상태"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


class BithumbWebSocketClient:
    """
    빗썸 WebSocket 클라이언트

    실시간 데이터 스트림을 수신하고 처리하는 WebSocket 클라이언트입니다.
    자동 재연결, 메시지 파싱, 상태 관리 기능을 제공합니다.
    """

    # 빗썸 WebSocket 엔드포인트
    WEBSOCKET_URL = "wss://pubwss.bithumb.com/pub/ws"

    # 연결 설정
    DEFAULT_PING_INTERVAL = 20  # 핑 간격 (초)
    DEFAULT_PING_TIMEOUT = 10   # 핑 타임아웃 (초)
    DEFAULT_CLOSE_TIMEOUT = 10  # 연결 종료 타임아웃 (초)

    def __init__(
        self,
        ping_interval: int = DEFAULT_PING_INTERVAL,
        ping_timeout: int = DEFAULT_PING_TIMEOUT,
        close_timeout: int = DEFAULT_CLOSE_TIMEOUT,
        enable_auto_reconnect: bool = True,
        max_retry_attempts: int = 10,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 300.0,
        reconnect_strategy: ReconnectStrategy = ReconnectStrategy.EXPONENTIAL_BACKOFF,
        strict_parsing: bool = False
    ):
        """
        WebSocket 클라이언트 초기화

        Args:
            ping_interval: 핑 메시지 전송 간격 (초)
            ping_timeout: 핑 응답 타임아웃 (초)
            close_timeout: 연결 종료 타임아웃 (초)
            enable_auto_reconnect: 자동 재연결 활성화 여부
            max_retry_attempts: 최대 재연결 시도 횟수 (0은 무제한)
            initial_retry_delay: 초기 재연결 지연 시간 (초)
            max_retry_delay: 최대 재연결 지연 시간 (초)
            reconnect_strategy: 재연결 전략
            strict_parsing: 엄격한 메시지 파싱 모드 (True시 파싱 오류시 예외 발생)
        """
        self._subscriptions: Dict[str, SubscriptionType] = {}
        self._message_handlers: Dict[SubscriptionType, List[Callable]] = {
            SubscriptionType.TICKER: [],
            SubscriptionType.ORDERBOOK: [],
            SubscriptionType.TRANSACTION: []
        }

        # 메시지 파서 초기화
        self._message_parser = get_message_parser(strict_mode=strict_parsing)

        # 재연결 매니저 초기화
        if enable_auto_reconnect:
            self._reconnect_manager = WebSocketReconnectManager(
                websocket_url=self.WEBSOCKET_URL,
                max_retry_attempts=max_retry_attempts,
                initial_retry_delay=initial_retry_delay,
                max_retry_delay=max_retry_delay,
                strategy=reconnect_strategy,
                heartbeat_interval=ping_interval,
                heartbeat_timeout=ping_timeout,
                connection_timeout=close_timeout
            )

            # 재연결 이벤트 콜백 등록
            self._reconnect_manager.add_connection_callback(self._on_reconnect_connected)
            self._reconnect_manager.add_disconnection_callback(self._on_reconnect_disconnected)
            self._reconnect_manager.add_reconnect_attempt_callback(self._on_reconnect_attempt)
            self._reconnect_manager.add_reconnect_failed_callback(self._on_reconnect_failed)
        else:
            self._reconnect_manager = None

        # 레거시 속성 (호환성 유지)
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._connection_state = ConnectionState.DISCONNECTED
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._close_timeout = close_timeout

        # 통계 정보
        self._stats = {
            'connected_at': None,
            'messages_received': 0,
            'messages_sent': 0,
            'reconnect_count': 0,
            'last_message_time': None,
            'errors': []
        }

        # 이벤트 루프 및 태스크 관리
        self._connection_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._is_running = False

    @property
    def connection_state(self) -> ConnectionState:
        """현재 연결 상태 반환"""
        return self._connection_state

    @property
    def is_connected(self) -> bool:
        """연결 여부 확인"""
        if self._reconnect_manager:
            return self._reconnect_manager.is_connected
        return self._connection_state == ConnectionState.CONNECTED and \
               self._websocket is not None and self._websocket.state.name == 'OPEN'

    @property
    def stats(self) -> Dict[str, Any]:
        """연결 및 메시지 통계 정보 반환"""
        return self._stats.copy()

    def add_message_handler(
        self,
        subscription_type: SubscriptionType,
        handler: Callable[[Dict[str, Any]], None]
    ):
        """
        메시지 핸들러 추가

        Args:
            subscription_type: 구독 데이터 타입
            handler: 메시지 처리 함수 (message dict를 인자로 받음)
        """
        self._message_handlers[subscription_type].append(handler)
        logger.info(f"{subscription_type.value} 메시지 핸들러 추가됨")

    def remove_message_handler(
        self,
        subscription_type: SubscriptionType,
        handler: Callable[[Dict[str, Any]], None]
    ):
        """
        메시지 핸들러 제거

        Args:
            subscription_type: 구독 데이터 타입
            handler: 제거할 메시지 처리 함수
        """
        if handler in self._message_handlers[subscription_type]:
            self._message_handlers[subscription_type].remove(handler)
            logger.info(f"{subscription_type.value} 메시지 핸들러 제거됨")

    # 재연결 이벤트 콜백 메서드들
    async def _on_reconnect_connected(self):
        """재연결 성공 시 호출되는 콜백"""
        self._connection_state = ConnectionState.CONNECTED
        self._stats['connected_at'] = datetime.now()
        self._stats['reconnect_count'] += 1

        # WebSocket 참조 업데이트 (재연결 매니저에서 가져오기)
        if self._reconnect_manager:
            self._websocket = self._reconnect_manager._websocket

        # 기존 구독 복구
        await self._restore_subscriptions()

        logger.info("재연결 성공 및 구독 복구 완료")

    async def _on_reconnect_disconnected(self, reason: str):
        """재연결 관리자의 연결 끊김 콜백"""
        self._connection_state = ConnectionState.DISCONNECTED
        error_info = {
            'timestamp': datetime.now(),
            'error': f"재연결 관리자 연결 끊김: {reason}"
        }
        self._stats['errors'].append(error_info)
        logger.warning(f"재연결 관리자 연결 끊김: {reason}")

    async def _on_reconnect_attempt(self, attempt_number: int):
        """재연결 시도 시 호출되는 콜백"""
        self._connection_state = ConnectionState.RECONNECTING
        logger.info(f"재연결 시도 중: {attempt_number}회")

    async def _on_reconnect_failed(self):
        """재연결 최종 실패 시 호출되는 콜백"""
        self._connection_state = ConnectionState.DISCONNECTED
        error_info = {
            'timestamp': datetime.now(),
            'error': "재연결 최종 실패"
        }
        self._stats['errors'].append(error_info)
        logger.error("재연결이 최종적으로 실패했습니다")

    async def _restore_subscriptions(self):
        """재연결 후 기존 구독 복구"""
        if not self._subscriptions:
            return

        logger.info("기존 구독 복구 시작")

        # 구독 타입별로 그룹화
        subscriptions_by_type: Dict[SubscriptionType, List[str]] = {}

        for subscription_key, subscription_type in self._subscriptions.items():
            # subscription_key는 "{type}_{symbol}" 형태
            symbol = subscription_key.split('_', 1)[1]

            if subscription_type not in subscriptions_by_type:
                subscriptions_by_type[subscription_type] = []
            subscriptions_by_type[subscription_type].append(symbol)

        # 타입별로 구독 복구
        for subscription_type, symbols in subscriptions_by_type.items():
            success = await self._send_subscription_message(subscription_type, symbols)
            if success:
                logger.info(f"{subscription_type.value} 구독 복구 성공: {symbols}")
            else:
                logger.error(f"{subscription_type.value} 구독 복구 실패: {symbols}")

    async def connect(self) -> bool:
        """
        WebSocket 서버에 연결

        Returns:
            연결 성공 여부
        """
        if self._reconnect_manager:
            # 재연결 매니저 사용
            success = await self._reconnect_manager.connect()
            if success:
                self._websocket = self._reconnect_manager._websocket
                self._connection_state = ConnectionState.CONNECTED
                self._stats['connected_at'] = datetime.now()
                self._stats['messages_received'] = 0
                self._stats['messages_sent'] = 0
            return success
        else:
            # 기존 방식 (재연결 없음)
            if self._connection_state in (ConnectionState.CONNECTING, ConnectionState.CONNECTED):
                logger.warning("이미 연결되어 있거나 연결 시도 중입니다")
                return self.is_connected

            try:
                self._connection_state = ConnectionState.CONNECTING
                logger.info(f"빗썸 WebSocket 서버 연결 시도: {self.WEBSOCKET_URL}")

                # WebSocket 연결 설정
                self._websocket = await websockets.connect(
                    self.WEBSOCKET_URL,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    close_timeout=self._close_timeout,
                    max_size=10**7,  # 10MB 최대 메시지 크기
                    max_queue=32,    # 메시지 큐 크기
                    compression=None  # 압축 비활성화 (성능상 이유)
                )

                self._connection_state = ConnectionState.CONNECTED
                self._stats['connected_at'] = datetime.now()
                self._stats['messages_received'] = 0
                self._stats['messages_sent'] = 0

                logger.info("빗썸 WebSocket 연결 성공")
                return True

            except Exception as e:
                self._connection_state = ConnectionState.DISCONNECTED
                error_msg = f"WebSocket 연결 실패: {str(e)}"
                logger.error(error_msg)
                self._stats['errors'].append({
                    'timestamp': datetime.now(),
                    'error': error_msg
                })
                return False

    async def disconnect(self):
        """WebSocket 연결 종료"""
        if self._reconnect_manager:
            # 재연결 매니저 사용
            await self._reconnect_manager.disconnect()
            self._connection_state = ConnectionState.CLOSED
            self._websocket = None
        else:
            # 기존 방식
            if self._websocket:
                try:
                    self._connection_state = ConnectionState.CLOSED
                    await self._websocket.close()
                    logger.info("WebSocket 연결 종료됨")
                except Exception as e:
                    logger.error(f"WebSocket 연결 종료 중 오류: {str(e)}")
                finally:
                    self._websocket = None

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """
        WebSocket 메시지 전송

        Args:
            message: 전송할 메시지 (dict 형태)

        Returns:
            전송 성공 여부
        """
        if not self.is_connected:
            logger.error("WebSocket이 연결되지 않음")
            return False

        try:
            message_str = json.dumps(message, ensure_ascii=False)

            if self._reconnect_manager:
                # 재연결 매니저 사용 (자동 재연결 지원)
                success = await self._reconnect_manager.send_message(message_str)
            else:
                # 기존 방식
                await self._websocket.send(message_str)
                success = True

            if success:
                self._stats['messages_sent'] += 1
                logger.debug(f"메시지 전송: {message_str}")

            return success

        except Exception as e:
            logger.error(f"메시지 전송 실패: {str(e)}")
            return False

    async def _send_subscription_message(
        self,
        subscription_type: SubscriptionType,
        symbols: List[str]
    ) -> bool:
        """
        구독 메시지 전송 (내부 메서드)

        Args:
            subscription_type: 구독할 데이터 타입
            symbols: 구독할 심볼 리스트

        Returns:
            전송 성공 여부
        """
        # 빗썸 WebSocket API 올바른 구독 형태 적용
        if subscription_type == SubscriptionType.TICKER:
            subscription_message = {
                "type": "ticker",
                "symbols": symbols,
                "tickTypes": ["24H"]  # 필수 파라미터
            }
        elif subscription_type == SubscriptionType.ORDERBOOK:
            subscription_message = {
                "type": "orderbookdepth",  # orderbook 대신 orderbookdepth 사용
                "symbols": symbols
            }
        elif subscription_type == SubscriptionType.TRANSACTION:
            subscription_message = {
                "type": "transaction",
                "symbols": symbols
            }
        else:
            # 기본 형태
            subscription_message = {
                "type": subscription_type.value,
                "symbols": symbols
            }

        return await self.send_message(subscription_message)

    async def subscribe(
        self,
        subscription_type: SubscriptionType,
        symbols: List[str]
    ) -> bool:
        """
        데이터 스트림 구독

        Args:
            subscription_type: 구독할 데이터 타입
            symbols: 구독할 심볼 리스트 (예: ["BTC_KRW", "ETH_KRW"])

        Returns:
            구독 성공 여부
        """
        if not self.is_connected:
            logger.error("WebSocket이 연결되지 않아 구독할 수 없음")
            return False

        success = await self._send_subscription_message(subscription_type, symbols)
        if success:
            # 구독 정보 저장 (재연결 시 복구용)
            for symbol in symbols:
                subscription_key = f"{subscription_type.value}_{symbol}"
                self._subscriptions[subscription_key] = subscription_type

            logger.info(f"{subscription_type.value} 구독 완료: {symbols}")
        else:
            logger.error(f"{subscription_type.value} 구독 실패: {symbols}")

        return success

    async def unsubscribe(
        self,
        subscription_type: SubscriptionType,
        symbols: List[str]
    ) -> bool:
        """
        데이터 스트림 구독 해제

        Args:
            subscription_type: 구독 해제할 데이터 타입
            symbols: 구독 해제할 심볼 리스트

        Returns:
            구독 해제 성공 여부
        """
        if not self.is_connected:
            logger.error("WebSocket이 연결되지 않아 구독 해제할 수 없음")
            return False

        # 구독 해제 메시지는 구독 메시지와 동일한 형태를 사용
        # (빗썸 API 특성에 따라 조정 필요할 수 있음)
        unsubscribe_message = {
            "type": subscription_type.value,
            "symbols": symbols,
            "action": "unsubscribe"  # 필요시 추가
        }

        success = await self.send_message(unsubscribe_message)
        if success:
            # 구독 정보에서 제거
            for symbol in symbols:
                subscription_key = f"{subscription_type.value}_{symbol}"
                self._subscriptions.pop(subscription_key, None)

            logger.info(f"{subscription_type.value} 구독 해제 완료: {symbols}")
        else:
            logger.error(f"{subscription_type.value} 구독 해제 실패: {symbols}")

        return success

    async def _handle_message(self, message_str: str):
        """
        수신된 메시지 처리 (개선된 파싱 시스템 사용)

        Args:
            message_str: 수신된 JSON 메시지 문자열
        """
        try:
            # 통계 업데이트
            self._stats['messages_received'] += 1
            self._stats['last_message_time'] = datetime.now()

            # 메시지 파싱 (새로운 파싱 시스템 사용)
            parsed_message = self._message_parser.parse_message(message_str)
            if parsed_message is None:
                logger.warning("메시지 파싱 실패")
                return

            # 타입별 메시지 핸들러 실행
            await self._dispatch_message_to_handlers(parsed_message)

        except MessageParsingError as e:
            logger.error(f"메시지 파싱 오류: {e.message}")
            if e.raw_data:
                logger.debug(f"원본 데이터: {str(e.raw_data)[:200]}...")
        except Exception as e:
            logger.error(f"메시지 처리 중 예외 발생: {str(e)}")

    async def _dispatch_message_to_handlers(self, parsed_message: WebSocketMessage):
        """
        파싱된 메시지를 적절한 핸들러들에게 전달

        Args:
            parsed_message: 파싱된 WebSocket 메시지
        """
        try:
            # 메시지 타입에 따른 SubscriptionType 매핑
            message_type_mapping = {
                MessageType.TICKER: SubscriptionType.TICKER,
                MessageType.ORDERBOOK: SubscriptionType.ORDERBOOK,
                MessageType.TRANSACTION: SubscriptionType.TRANSACTION,
            }

            subscription_type = message_type_mapping.get(parsed_message.type)
            if subscription_type is None:
                logger.debug(f"핸들러가 없는 메시지 타입: {parsed_message.type}")
                return

            handlers = self._message_handlers.get(subscription_type, [])
            if not handlers:
                logger.debug(f"등록된 핸들러가 없음: {subscription_type}")
                return

            # 각 핸들러 실행
            for handler in handlers:
                try:
                    # 핸들러 인수 결정 (구조화된 데이터 또는 원본 데이터)
                    handler_args = self._prepare_handler_arguments(parsed_message, subscription_type)

                    # 비동기 핸들러인지 확인하여 적절히 호출
                    if asyncio.iscoroutinefunction(handler):
                        await handler(*handler_args)
                    else:
                        handler(*handler_args)

                except Exception as e:
                    logger.error(f"메시지 핸들러 실행 오류: {str(e)}")

        except Exception as e:
            logger.error(f"메시지 디스패치 중 오류: {str(e)}")

    def _prepare_handler_arguments(self, parsed_message: WebSocketMessage, subscription_type: SubscriptionType):
        """
        핸들러 호출을 위한 인수 준비

        Args:
            parsed_message: 파싱된 메시지
            subscription_type: 구독 타입

        Returns:
            핸들러에 전달할 인수 튜플
        """
        # 구조화된 데이터와 원본 메시지 모두 제공
        if subscription_type == SubscriptionType.TICKER and parsed_message.ticker_data:
            return (parsed_message.ticker_data, parsed_message)
        elif subscription_type == SubscriptionType.ORDERBOOK and parsed_message.orderbook_data:
            return (parsed_message.orderbook_data, parsed_message)
        elif subscription_type == SubscriptionType.TRANSACTION and parsed_message.transaction_data:
            return (parsed_message.transaction_data, parsed_message)
        else:
            # 구조화된 데이터가 없는 경우 원본 데이터만 전달
            return (parsed_message.raw_data or {}, parsed_message)

    async def start_receiving(self):
        """메시지 수신 시작"""
        if not self.is_connected:
            logger.error("WebSocket이 연결되지 않아 메시지 수신을 시작할 수 없음")
            return

        logger.info("WebSocket 메시지 수신 시작")

        try:
            async for message in self._websocket:
                await self._handle_message(message)

        except ConnectionClosed as e:
            logger.warning(f"WebSocket 연결이 종료됨: {e}")
            self._connection_state = ConnectionState.DISCONNECTED
        except WebSocketException as e:
            logger.error(f"WebSocket 오류: {str(e)}")
            self._connection_state = ConnectionState.DISCONNECTED
        except Exception as e:
            logger.error(f"메시지 수신 중 예외 발생: {str(e)}")
            self._connection_state = ConnectionState.DISCONNECTED

    async def health_check(self) -> Dict[str, Any]:
        """
        WebSocket 연결 상태 확인

        Returns:
            상태 정보가 담긴 딕셔너리
        """
        return {
            'connected': self.is_connected,
            'connection_state': self._connection_state.value,
            'subscriptions': len(self._subscriptions),
            'active_subscriptions': list(self._subscriptions.keys()),
            'stats': self.stats
        }

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.disconnect()

    def __del__(self):
        """소멸자에서 연결 정리"""
        if self._websocket and self._websocket.state.name == 'OPEN':
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.create_task(self.disconnect())
            except RuntimeError:
                # 이벤트 루프가 없거나 닫힌 경우 무시
                pass


# 전역 WebSocket 클라이언트 인스턴스 (싱글톤 패턴)
_global_websocket_client: Optional[BithumbWebSocketClient] = None


async def get_websocket_client() -> BithumbWebSocketClient:
    """
    전역 WebSocket 클라이언트 인스턴스 반환

    싱글톤 패턴으로 하나의 클라이언트 인스턴스만 사용합니다.

    Returns:
        BithumbWebSocketClient 인스턴스
    """
    global _global_websocket_client

    if _global_websocket_client is None:
        _global_websocket_client = BithumbWebSocketClient()

    return _global_websocket_client


async def close_websocket_client():
    """전역 WebSocket 클라이언트 정리"""
    global _global_websocket_client

    if _global_websocket_client is not None:
        await _global_websocket_client.disconnect()
        _global_websocket_client = None