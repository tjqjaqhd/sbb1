"""
Redis Queue 데이터 버퍼링 시스템

실시간 WebSocket 데이터를 Redis Queue에 버퍼링하여 안정적인 데이터 처리를 지원합니다.
높은 처리량의 데이터 스트림을 효율적으로 관리하고, 백프레셔 상황을 대비합니다.
"""

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable, AsyncGenerator
from enum import Enum
import uuid

import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from .message_parser import WebSocketMessage, MessageType

logger = logging.getLogger(__name__)


class SerializationFormat(str, Enum):
    """직렬화 포맷"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"  # 향후 지원 예정


class BufferStrategy(str, Enum):
    """버퍼링 전략"""
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    PRIORITY = "priority"  # 우선순위 기반


class QueueStatus(str, Enum):
    """큐 상태"""
    ACTIVE = "active"
    PAUSED = "paused"
    DRAINING = "draining"
    ERROR = "error"


class RedisQueueBuffer:
    """
    Redis 기반 큐 버퍼 시스템

    실시간 데이터를 Redis 큐에 안전하게 버퍼링하고,
    소비자가 안정적으로 데이터를 처리할 수 있도록 지원합니다.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        queue_prefix: str = "bithumb_ws",
        max_queue_size: int = 10000,
        serialization_format: SerializationFormat = SerializationFormat.JSON,
        buffer_strategy: BufferStrategy = BufferStrategy.FIFO,
        ttl_seconds: int = 3600,
        batch_size: int = 100,
        connection_pool_size: int = 20
    ):
        """
        Redis 큐 버퍼 초기화

        Args:
            redis_url: Redis 서버 URL
            queue_prefix: 큐 이름 접두사
            max_queue_size: 최대 큐 크기
            serialization_format: 직렬화 포맷
            buffer_strategy: 버퍼링 전략
            ttl_seconds: 데이터 TTL (초)
            batch_size: 배치 처리 크기
            connection_pool_size: 연결 풀 크기
        """
        self._redis_url = redis_url
        self._queue_prefix = queue_prefix
        self._max_queue_size = max_queue_size
        self._serialization_format = serialization_format
        self._buffer_strategy = buffer_strategy
        self._ttl_seconds = ttl_seconds
        self._batch_size = batch_size

        # Redis 연결 풀
        self._connection_pool = ConnectionPool.from_url(
            redis_url,
            max_connections=connection_pool_size,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30
        )

        # Redis 클라이언트
        self._redis: Optional[Redis] = None

        # 큐 관리
        self._active_queues: Dict[str, QueueStatus] = {}
        self._queue_stats: Dict[str, Dict[str, Any]] = {}

        # 모니터링 및 통계
        self._global_stats = {
            'messages_enqueued': 0,
            'messages_dequeued': 0,
            'bytes_written': 0,
            'bytes_read': 0,
            'serialization_errors': 0,
            'queue_full_events': 0,
            'last_activity': None
        }

        # 이벤트 콜백
        self._on_queue_full_callbacks: List[Callable] = []
        self._on_queue_empty_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []

        # 실행 상태
        self._is_running = False
        self._background_tasks: List[asyncio.Task] = []

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.disconnect()

    async def connect(self) -> bool:
        """Redis에 연결"""
        try:
            self._redis = Redis(connection_pool=self._connection_pool, decode_responses=False)

            # 연결 테스트
            await self._redis.ping()

            logger.info(f"Redis에 연결됨: {self._redis_url}")
            self._is_running = True

            # 백그라운드 모니터링 시작
            await self._start_background_monitoring()

            return True

        except Exception as e:
            logger.error(f"Redis 연결 실패: {str(e)}")
            return False

    async def disconnect(self):
        """Redis 연결 종료"""
        self._is_running = False

        # 백그라운드 태스크 정리
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._background_tasks.clear()

        # Redis 연결 종료
        if self._redis:
            await self._redis.close()
            self._redis = None

        # 연결 풀 종료
        if self._connection_pool:
            await self._connection_pool.disconnect()

        logger.info("Redis 연결 종료됨")

    def _get_queue_key(self, queue_name: str) -> str:
        """큐 키 생성"""
        return f"{self._queue_prefix}:{queue_name}"

    def _get_queue_info_key(self, queue_name: str) -> str:
        """큐 정보 키 생성"""
        return f"{self._queue_prefix}:info:{queue_name}"

    async def _serialize_data(self, data: Any) -> bytes:
        """데이터 직렬화"""
        try:
            if self._serialization_format == SerializationFormat.JSON:
                if isinstance(data, WebSocketMessage):
                    # Pydantic 모델인 경우 dict로 변환
                    serialized = json.dumps(data.dict(), ensure_ascii=False, default=str)
                else:
                    serialized = json.dumps(data, ensure_ascii=False, default=str)
                return serialized.encode('utf-8')

            elif self._serialization_format == SerializationFormat.PICKLE:
                return pickle.dumps(data)

            else:
                raise ValueError(f"지원하지 않는 직렬화 포맷: {self._serialization_format}")

        except Exception as e:
            self._global_stats['serialization_errors'] += 1
            logger.error(f"데이터 직렬화 실패: {str(e)}")
            raise

    async def _deserialize_data(self, data: bytes) -> Any:
        """데이터 역직렬화"""
        try:
            if self._serialization_format == SerializationFormat.JSON:
                return json.loads(data.decode('utf-8'))
            elif self._serialization_format == SerializationFormat.PICKLE:
                return pickle.loads(data)
            else:
                raise ValueError(f"지원하지 않는 직렬화 포맷: {self._serialization_format}")

        except Exception as e:
            self._global_stats['serialization_errors'] += 1
            logger.error(f"데이터 역직렬화 실패: {str(e)}")
            raise

    async def create_queue(self, queue_name: str, max_size: Optional[int] = None) -> bool:
        """
        큐 생성

        Args:
            queue_name: 큐 이름
            max_size: 최대 크기 (None이면 기본값 사용)

        Returns:
            생성 성공 여부
        """
        if not self._redis:
            logger.error("Redis가 연결되지 않음")
            return False

        try:
            queue_key = self._get_queue_key(queue_name)
            info_key = self._get_queue_info_key(queue_name)

            # 큐 정보 저장
            queue_info = {
                'created_at': datetime.now().isoformat(),
                'max_size': max_size or self._max_queue_size,
                'strategy': self._buffer_strategy.value,
                'ttl_seconds': self._ttl_seconds
            }

            await self._redis.hset(info_key, mapping={
                k: json.dumps(v) if not isinstance(v, (str, int, float)) else str(v)
                for k, v in queue_info.items()
            })

            # TTL 설정
            if self._ttl_seconds > 0:
                await self._redis.expire(info_key, self._ttl_seconds)

            # 상태 관리에 추가
            self._active_queues[queue_name] = QueueStatus.ACTIVE
            self._queue_stats[queue_name] = {
                'enqueued': 0,
                'dequeued': 0,
                'current_size': 0,
                'max_size_reached': 0,
                'last_enqueue': None,
                'last_dequeue': None
            }

            logger.info(f"큐 생성됨: {queue_name}")
            return True

        except Exception as e:
            logger.error(f"큐 생성 실패 {queue_name}: {str(e)}")
            return False

    async def enqueue(
        self,
        queue_name: str,
        data: Any,
        priority: int = 0,
        expire_at: Optional[datetime] = None
    ) -> bool:
        """
        큐에 데이터 추가

        Args:
            queue_name: 큐 이름
            data: 추가할 데이터
            priority: 우선순위 (PRIORITY 전략에서만 사용)
            expire_at: 만료 시간

        Returns:
            추가 성공 여부
        """
        if not self._redis:
            logger.error("Redis가 연결되지 않음")
            return False

        if queue_name not in self._active_queues:
            # 큐가 없으면 자동 생성
            await self.create_queue(queue_name)

        try:
            queue_key = self._get_queue_key(queue_name)

            # 큐 크기 확인
            current_size = await self._redis.llen(queue_key)
            max_size = self._queue_stats[queue_name].get('max_size', self._max_queue_size)

            if current_size >= max_size:
                logger.warning(f"큐가 가득참: {queue_name} ({current_size}/{max_size})")
                self._global_stats['queue_full_events'] += 1
                self._queue_stats[queue_name]['max_size_reached'] += 1

                # 큐 가득참 콜백 실행
                await self._execute_callbacks(self._on_queue_full_callbacks, queue_name, current_size)

                # 전략에 따른 처리
                if self._buffer_strategy == BufferStrategy.FIFO:
                    # 오래된 데이터 제거
                    await self._redis.lpop(queue_key)
                elif self._buffer_strategy == BufferStrategy.LIFO:
                    # 최신 데이터 제거
                    await self._redis.rpop(queue_key)
                else:
                    # 큐가 가득참을 알리고 실패
                    return False

            # 데이터 직렬화
            serialized_data = await self._serialize_data(data)

            # 메타데이터 추가
            message_envelope = {
                'id': str(uuid.uuid4()),
                'enqueued_at': datetime.now().isoformat(),
                'priority': priority,
                'expire_at': expire_at.isoformat() if expire_at else None,
                'data': serialized_data.decode('utf-8') if self._serialization_format == SerializationFormat.JSON else None
            }

            if self._serialization_format == SerializationFormat.PICKLE:
                message_envelope['data_pickle'] = serialized_data

            envelope_data = await self._serialize_data(message_envelope)

            # 전략에 따른 큐 삽입
            if self._buffer_strategy == BufferStrategy.FIFO:
                await self._redis.rpush(queue_key, envelope_data)
            elif self._buffer_strategy == BufferStrategy.LIFO:
                await self._redis.lpush(queue_key, envelope_data)
            elif self._buffer_strategy == BufferStrategy.PRIORITY:
                # 우선순위 큐 (SORTED SET 사용)
                priority_key = f"{queue_key}:priority"
                await self._redis.zadd(priority_key, {envelope_data: -priority})  # 높은 우선순위가 먼저
            else:
                # 기본값은 FIFO
                await self._redis.rpush(queue_key, envelope_data)

            # TTL 설정
            if self._ttl_seconds > 0:
                await self._redis.expire(queue_key, self._ttl_seconds)

            # 통계 업데이트
            self._global_stats['messages_enqueued'] += 1
            self._global_stats['bytes_written'] += len(envelope_data)
            self._global_stats['last_activity'] = datetime.now()

            self._queue_stats[queue_name]['enqueued'] += 1
            self._queue_stats[queue_name]['current_size'] = await self._redis.llen(queue_key)
            self._queue_stats[queue_name]['last_enqueue'] = datetime.now()

            return True

        except Exception as e:
            logger.error(f"큐 추가 실패 {queue_name}: {str(e)}")
            await self._execute_callbacks(self._on_error_callbacks, queue_name, str(e))
            return False

    async def dequeue(self, queue_name: str, timeout: float = 0.0) -> Optional[Any]:
        """
        큐에서 데이터 제거 및 반환

        Args:
            queue_name: 큐 이름
            timeout: 대기 시간 (초, 0이면 즉시 반환)

        Returns:
            제거된 데이터 또는 None
        """
        if not self._redis:
            logger.error("Redis가 연결되지 않음")
            return None

        if queue_name not in self._active_queues:
            logger.error(f"존재하지 않는 큐: {queue_name}")
            return None

        try:
            queue_key = self._get_queue_key(queue_name)

            # 전략에 따른 큐에서 데이터 가져오기
            envelope_data = None

            if self._buffer_strategy == BufferStrategy.PRIORITY:
                priority_key = f"{queue_key}:priority"
                if timeout > 0:
                    # 블로킹 방식은 priority queue에서 지원하지 않음
                    result = await self._redis.zpopmax(priority_key)
                else:
                    result = await self._redis.zpopmax(priority_key)

                if result:
                    envelope_data = result[0][0]  # (value, score) 튜플에서 value 추출
            else:
                if timeout > 0:
                    # 블로킹 방식
                    result = await self._redis.blpop(queue_key, timeout=timeout)
                    if result:
                        envelope_data = result[1]  # (key, value) 튜플에서 value 추출
                else:
                    # 논블로킹 방식
                    if self._buffer_strategy == BufferStrategy.LIFO:
                        envelope_data = await self._redis.rpop(queue_key)
                    else:  # FIFO
                        envelope_data = await self._redis.lpop(queue_key)

            if envelope_data is None:
                return None

            # 봉투 데이터 역직렬화
            message_envelope = await self._deserialize_data(envelope_data)

            # 만료 시간 확인
            if message_envelope.get('expire_at'):
                expire_at = datetime.fromisoformat(message_envelope['expire_at'])
                if datetime.now() > expire_at:
                    logger.debug("만료된 메시지 건너뜀")
                    return await self.dequeue(queue_name, timeout)  # 재귀적으로 다음 메시지 시도

            # 실제 데이터 추출
            if self._serialization_format == SerializationFormat.JSON:
                data = json.loads(message_envelope['data'])
            elif self._serialization_format == SerializationFormat.PICKLE:
                data = pickle.loads(message_envelope['data_pickle'])
            else:
                data = message_envelope['data']

            # 통계 업데이트
            self._global_stats['messages_dequeued'] += 1
            self._global_stats['bytes_read'] += len(envelope_data)
            self._global_stats['last_activity'] = datetime.now()

            if queue_name in self._queue_stats:
                self._queue_stats[queue_name]['dequeued'] += 1
                self._queue_stats[queue_name]['current_size'] = await self._redis.llen(queue_key)
                self._queue_stats[queue_name]['last_dequeue'] = datetime.now()

            return data

        except Exception as e:
            logger.error(f"큐 제거 실패 {queue_name}: {str(e)}")
            await self._execute_callbacks(self._on_error_callbacks, queue_name, str(e))
            return None

    async def dequeue_batch(self, queue_name: str, batch_size: Optional[int] = None) -> List[Any]:
        """
        배치로 데이터 제거

        Args:
            queue_name: 큐 이름
            batch_size: 배치 크기

        Returns:
            제거된 데이터 리스트
        """
        batch_size = batch_size or self._batch_size
        result = []

        for _ in range(batch_size):
            data = await self.dequeue(queue_name, timeout=0.0)
            if data is None:
                break
            result.append(data)

        return result

    async def get_queue_size(self, queue_name: str) -> int:
        """큐 크기 반환"""
        if not self._redis or queue_name not in self._active_queues:
            return 0

        try:
            queue_key = self._get_queue_key(queue_name)
            if self._buffer_strategy == BufferStrategy.PRIORITY:
                priority_key = f"{queue_key}:priority"
                return await self._redis.zcard(priority_key)
            else:
                return await self._redis.llen(queue_key)
        except Exception as e:
            logger.error(f"큐 크기 조회 실패 {queue_name}: {str(e)}")
            return 0

    async def clear_queue(self, queue_name: str) -> bool:
        """큐 비우기"""
        if not self._redis or queue_name not in self._active_queues:
            return False

        try:
            queue_key = self._get_queue_key(queue_name)
            info_key = self._get_queue_info_key(queue_name)

            if self._buffer_strategy == BufferStrategy.PRIORITY:
                priority_key = f"{queue_key}:priority"
                await self._redis.delete(priority_key)
            else:
                await self._redis.delete(queue_key)

            # 통계 초기화
            if queue_name in self._queue_stats:
                self._queue_stats[queue_name]['current_size'] = 0

            logger.info(f"큐 비움: {queue_name}")
            return True

        except Exception as e:
            logger.error(f"큐 비우기 실패 {queue_name}: {str(e)}")
            return False

    async def delete_queue(self, queue_name: str) -> bool:
        """큐 삭제"""
        if not self._redis:
            return False

        try:
            queue_key = self._get_queue_key(queue_name)
            info_key = self._get_queue_info_key(queue_name)

            # 모든 관련 키 삭제
            keys_to_delete = [queue_key, info_key]
            if self._buffer_strategy == BufferStrategy.PRIORITY:
                priority_key = f"{queue_key}:priority"
                keys_to_delete.append(priority_key)

            await self._redis.delete(*keys_to_delete)

            # 상태에서 제거
            self._active_queues.pop(queue_name, None)
            self._queue_stats.pop(queue_name, None)

            logger.info(f"큐 삭제됨: {queue_name}")
            return True

        except Exception as e:
            logger.error(f"큐 삭제 실패 {queue_name}: {str(e)}")
            return False

    async def _start_background_monitoring(self):
        """백그라운드 모니터링 시작"""
        monitor_task = asyncio.create_task(self._monitor_queues())
        self._background_tasks.append(monitor_task)

    async def _monitor_queues(self):
        """큐 모니터링 루프"""
        while self._is_running:
            try:
                for queue_name in list(self._active_queues.keys()):
                    current_size = await self.get_queue_size(queue_name)

                    # 큐가 비어있으면 콜백 실행
                    if current_size == 0:
                        await self._execute_callbacks(self._on_queue_empty_callbacks, queue_name)

                    # 통계 업데이트
                    if queue_name in self._queue_stats:
                        self._queue_stats[queue_name]['current_size'] = current_size

                await asyncio.sleep(30)  # 30초마다 모니터링

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"큐 모니터링 오류: {str(e)}")
                await asyncio.sleep(5)

    async def _execute_callbacks(self, callbacks: List[Callable], *args, **kwargs):
        """콜백 함수들 실행"""
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"콜백 실행 오류: {str(e)}")

    def add_queue_full_callback(self, callback: Callable):
        """큐 가득참 콜백 추가"""
        self._on_queue_full_callbacks.append(callback)

    def add_queue_empty_callback(self, callback: Callable):
        """큐 비어있음 콜백 추가"""
        self._on_queue_empty_callbacks.append(callback)

    def add_error_callback(self, callback: Callable):
        """에러 콜백 추가"""
        self._on_error_callbacks.append(callback)

    async def get_stats(self) -> Dict[str, Any]:
        """전체 통계 정보 반환"""
        return {
            'global_stats': self._global_stats.copy(),
            'queue_stats': self._queue_stats.copy(),
            'active_queues': dict(self._active_queues),
            'redis_info': await self._redis.info() if self._redis else None
        }

    async def health_check(self) -> Dict[str, Any]:
        """건강 상태 확인"""
        try:
            if not self._redis:
                return {'status': 'error', 'message': 'Redis 연결되지 않음'}

            # Redis 핑 테스트
            await self._redis.ping()

            return {
                'status': 'healthy',
                'redis_connected': True,
                'active_queues': len(self._active_queues),
                'total_messages_processed': self._global_stats['messages_enqueued'] + self._global_stats['messages_dequeued'],
                'last_activity': self._global_stats.get('last_activity')
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}


# 전역 버퍼 인스턴스
_global_redis_buffer: Optional[RedisQueueBuffer] = None


async def get_redis_buffer(**kwargs) -> RedisQueueBuffer:
    """
    전역 Redis 버퍼 인스턴스 반환

    Returns:
        RedisQueueBuffer 인스턴스
    """
    global _global_redis_buffer

    if _global_redis_buffer is None:
        _global_redis_buffer = RedisQueueBuffer(**kwargs)
        await _global_redis_buffer.connect()

    return _global_redis_buffer


async def close_redis_buffer():
    """전역 Redis 버퍼 정리"""
    global _global_redis_buffer

    if _global_redis_buffer is not None:
        await _global_redis_buffer.disconnect()
        _global_redis_buffer = None