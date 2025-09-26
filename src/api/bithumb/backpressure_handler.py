"""
백프레셔 핸들링 메커니즘

높은 데이터 처리량 상황에서 시스템 안정성을 보장하는 백프레셔 제어 시스템입니다.
큐 크기 모니터링, 동적 버퍼 조정, 처리 속도 제어 등의 기능을 제공합니다.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, NamedTuple
from enum import Enum
import threading
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


class BackpressureLevel(str, Enum):
    """백프레셔 수준"""
    NONE = "none"           # 정상 상태
    LOW = "low"             # 낮은 백프레셔
    MEDIUM = "medium"       # 중간 백프레셔
    HIGH = "high"           # 높은 백프레셔
    CRITICAL = "critical"   # 임계 상태


class DropPolicy(str, Enum):
    """데이터 드롭 정책"""
    NONE = "none"           # 드롭 없음
    OLDEST = "oldest"       # 오래된 데이터 드롭
    NEWEST = "newest"       # 최신 데이터 드롭
    RANDOM = "random"       # 랜덤 드롭
    PRIORITY_LOW = "priority_low"  # 낮은 우선순위 드롭


class ThrottleStrategy(str, Enum):
    """스로틀링 전략"""
    ADAPTIVE = "adaptive"   # 적응형 스로틀링
    LINEAR = "linear"       # 선형 스로틀링
    EXPONENTIAL = "exponential"  # 지수적 스로틀링
    CIRCUIT_BREAKER = "circuit_breaker"  # 서킷 브레이커


@dataclass
class BackpressureMetrics:
    """백프레셔 메트릭"""
    timestamp: datetime
    queue_size: int
    queue_capacity: int
    memory_usage_mb: float
    cpu_usage_percent: float
    processing_rate: float  # 메시지/초
    ingestion_rate: float   # 메시지/초
    backpressure_level: BackpressureLevel
    throttle_factor: float  # 0.0 ~ 1.0 (1.0 = 정상 속도)


class SystemResourceMonitor:
    """시스템 리소스 모니터"""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._cpu_usage = 0.0
        self._memory_usage_mb = 0.0
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self):
        """모니터링 시작"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """모니터링 중지"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitoring_loop(self):
        """모니터링 루프"""
        while self._monitoring:
            try:
                # CPU 사용률 (논블로킹)
                self._cpu_usage = psutil.cpu_percent(interval=None)

                # 메모리 사용률
                process = psutil.Process()
                memory_info = process.memory_info()
                self._memory_usage_mb = memory_info.rss / 1024 / 1024

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"리소스 모니터링 오류: {str(e)}")
                await asyncio.sleep(self.update_interval)

    @property
    def cpu_usage(self) -> float:
        """현재 CPU 사용률"""
        return self._cpu_usage

    @property
    def memory_usage_mb(self) -> float:
        """현재 메모리 사용량 (MB)"""
        return self._memory_usage_mb


class RateCalculator:
    """처리율 계산기"""

    def __init__(self, window_size: int = 60):
        """
        Args:
            window_size: 계산 윈도우 크기 (초)
        """
        self.window_size = window_size
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def record_event(self):
        """이벤트 기록"""
        current_time = time.time()
        with self._lock:
            self._timestamps.append(current_time)
            # 윈도우 크기를 넘는 오래된 이벤트 제거
            cutoff_time = current_time - self.window_size
            self._timestamps = [ts for ts in self._timestamps if ts >= cutoff_time]

    def get_rate(self) -> float:
        """현재 처리율 반환 (이벤트/초)"""
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0

            current_time = time.time()
            cutoff_time = current_time - self.window_size
            recent_events = [ts for ts in self._timestamps if ts >= cutoff_time]

            if len(recent_events) < 2:
                return 0.0

            time_span = recent_events[-1] - recent_events[0]
            if time_span <= 0:
                return 0.0

            return len(recent_events) / time_span

    def reset(self):
        """통계 초기화"""
        with self._lock:
            self._timestamps.clear()


class BackpressureHandler:
    """
    백프레셔 핸들링 시스템

    큐 크기, 시스템 리소스, 처리율 등을 모니터링하여
    동적으로 백프레셔를 제어하는 시스템입니다.
    """

    def __init__(
        self,
        queue_capacity_threshold: Dict[BackpressureLevel, float] = None,
        memory_threshold_mb: float = 1000.0,
        cpu_threshold_percent: float = 80.0,
        drop_policy: DropPolicy = DropPolicy.OLDEST,
        throttle_strategy: ThrottleStrategy = ThrottleStrategy.ADAPTIVE,
        monitoring_interval: float = 1.0,
        rate_window_size: int = 60
    ):
        """
        백프레셔 핸들러 초기화

        Args:
            queue_capacity_threshold: 백프레셔 수준별 큐 용량 임계값 (0.0 ~ 1.0)
            memory_threshold_mb: 메모리 사용량 임계값 (MB)
            cpu_threshold_percent: CPU 사용률 임계값 (%)
            drop_policy: 데이터 드롭 정책
            throttle_strategy: 스로틀링 전략
            monitoring_interval: 모니터링 간격 (초)
            rate_window_size: 처리율 계산 윈도우 크기 (초)
        """
        # 임계값 설정
        self.queue_thresholds = queue_capacity_threshold or {
            BackpressureLevel.NONE: 0.5,      # 50% 이하
            BackpressureLevel.LOW: 0.7,       # 70% 이하
            BackpressureLevel.MEDIUM: 0.85,   # 85% 이하
            BackpressureLevel.HIGH: 0.95,     # 95% 이하
            BackpressureLevel.CRITICAL: 1.0   # 100%
        }

        self.memory_threshold_mb = memory_threshold_mb
        self.cpu_threshold_percent = cpu_threshold_percent
        self.drop_policy = drop_policy
        self.throttle_strategy = throttle_strategy
        self.monitoring_interval = monitoring_interval

        # 모니터링 컴포넌트
        self._resource_monitor = SystemResourceMonitor(monitoring_interval)
        self._ingestion_rate_calc = RateCalculator(rate_window_size)
        self._processing_rate_calc = RateCalculator(rate_window_size)

        # 상태 관리
        self._current_level = BackpressureLevel.NONE
        self._throttle_factor = 1.0
        self._metrics_history: List[BackpressureMetrics] = []
        self._max_history_size = 1000

        # 콜백 함수들
        self._level_change_callbacks: List[Callable] = []
        self._metrics_callbacks: List[Callable] = []

        # 실행 상태
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # 통계
        self._stats = {
            'total_events_ingested': 0,
            'total_events_processed': 0,
            'total_events_dropped': 0,
            'backpressure_events': {level.value: 0 for level in BackpressureLevel},
            'throttle_activations': 0,
            'last_level_change': None
        }

    async def start(self):
        """백프레셔 핸들러 시작"""
        if self._is_running:
            return

        self._is_running = True

        # 리소스 모니터링 시작
        await self._resource_monitor.start_monitoring()

        # 백프레셔 모니터링 시작
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        logger.info("백프레셔 핸들러 시작됨")

    async def stop(self):
        """백프레셔 핸들러 중지"""
        self._is_running = False

        # 모니터링 태스크 중지
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        # 리소스 모니터링 중지
        await self._resource_monitor.stop_monitoring()

        logger.info("백프레셔 핸들러 중지됨")

    async def _monitoring_loop(self):
        """백프레셔 모니터링 루프"""
        while self._is_running:
            try:
                await self._update_backpressure_metrics()
                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"백프레셔 모니터링 오류: {str(e)}")
                await asyncio.sleep(self.monitoring_interval)

    async def _update_backpressure_metrics(self):
        """백프레셔 메트릭 업데이트"""
        # 기본 메트릭 수집은 외부에서 제공받음
        # 여기서는 레벨 계산 및 스로틀 팩터 조정만 수행
        pass

    def record_ingestion(self, count: int = 1):
        """데이터 수신 기록"""
        self._stats['total_events_ingested'] += count
        for _ in range(count):
            self._ingestion_rate_calc.record_event()

    def record_processing(self, count: int = 1):
        """데이터 처리 기록"""
        self._stats['total_events_processed'] += count
        for _ in range(count):
            self._processing_rate_calc.record_event()

    def record_drop(self, count: int = 1):
        """데이터 드롭 기록"""
        self._stats['total_events_dropped'] += count

    def update_queue_metrics(self, queue_size: int, queue_capacity: int):
        """큐 메트릭 업데이트"""
        # 백프레셔 레벨 계산
        queue_utilization = queue_size / max(queue_capacity, 1)
        new_level = self._calculate_backpressure_level(
            queue_utilization,
            self._resource_monitor.memory_usage_mb,
            self._resource_monitor.cpu_usage
        )

        # 레벨 변경 시 처리
        if new_level != self._current_level:
            old_level = self._current_level
            self._current_level = new_level
            self._stats['backpressure_events'][new_level.value] += 1
            self._stats['last_level_change'] = datetime.now()

            # 스로틀 팩터 조정
            self._update_throttle_factor(new_level)

            # 콜백 실행
            asyncio.create_task(self._execute_level_change_callbacks(old_level, new_level))

        # 메트릭 생성 및 저장
        metrics = BackpressureMetrics(
            timestamp=datetime.now(),
            queue_size=queue_size,
            queue_capacity=queue_capacity,
            memory_usage_mb=self._resource_monitor.memory_usage_mb,
            cpu_usage_percent=self._resource_monitor.cpu_usage,
            processing_rate=self._processing_rate_calc.get_rate(),
            ingestion_rate=self._ingestion_rate_calc.get_rate(),
            backpressure_level=self._current_level,
            throttle_factor=self._throttle_factor
        )

        self._add_metrics(metrics)

    def _calculate_backpressure_level(
        self,
        queue_utilization: float,
        memory_usage_mb: float,
        cpu_usage_percent: float
    ) -> BackpressureLevel:
        """백프레셔 레벨 계산"""
        # 큐 사용률 기반 레벨 결정
        for level in [BackpressureLevel.CRITICAL, BackpressureLevel.HIGH,
                      BackpressureLevel.MEDIUM, BackpressureLevel.LOW, BackpressureLevel.NONE]:
            if queue_utilization >= self.queue_thresholds[level]:
                queue_level = level
                break
        else:
            queue_level = BackpressureLevel.NONE

        # 시스템 리소스 기반 레벨 결정
        resource_level = BackpressureLevel.NONE
        if memory_usage_mb > self.memory_threshold_mb:
            resource_level = BackpressureLevel.HIGH
        elif cpu_usage_percent > self.cpu_threshold_percent:
            resource_level = BackpressureLevel.MEDIUM

        # 두 레벨 중 더 높은 것 선택
        levels_priority = {
            BackpressureLevel.NONE: 0,
            BackpressureLevel.LOW: 1,
            BackpressureLevel.MEDIUM: 2,
            BackpressureLevel.HIGH: 3,
            BackpressureLevel.CRITICAL: 4
        }

        queue_priority = levels_priority[queue_level]
        resource_priority = levels_priority[resource_level]

        final_priority = max(queue_priority, resource_priority)
        final_level = [level for level, priority in levels_priority.items()
                       if priority == final_priority][0]

        return final_level

    def _update_throttle_factor(self, level: BackpressureLevel):
        """스로틀 팩터 업데이트"""
        if self.throttle_strategy == ThrottleStrategy.ADAPTIVE:
            # 적응형 스로틀링
            throttle_map = {
                BackpressureLevel.NONE: 1.0,
                BackpressureLevel.LOW: 0.9,
                BackpressureLevel.MEDIUM: 0.7,
                BackpressureLevel.HIGH: 0.5,
                BackpressureLevel.CRITICAL: 0.2
            }
            self._throttle_factor = throttle_map[level]

        elif self.throttle_strategy == ThrottleStrategy.LINEAR:
            # 선형 스로틀링
            level_values = {
                BackpressureLevel.NONE: 0,
                BackpressureLevel.LOW: 1,
                BackpressureLevel.MEDIUM: 2,
                BackpressureLevel.HIGH: 3,
                BackpressureLevel.CRITICAL: 4
            }
            factor = 1.0 - (level_values[level] * 0.2)
            self._throttle_factor = max(0.1, factor)

        elif self.throttle_strategy == ThrottleStrategy.EXPONENTIAL:
            # 지수적 스로틀링
            level_values = {
                BackpressureLevel.NONE: 0,
                BackpressureLevel.LOW: 1,
                BackpressureLevel.MEDIUM: 2,
                BackpressureLevel.HIGH: 3,
                BackpressureLevel.CRITICAL: 4
            }
            factor = 1.0 / (2 ** level_values[level])
            self._throttle_factor = max(0.1, factor)

        elif self.throttle_strategy == ThrottleStrategy.CIRCUIT_BREAKER:
            # 서킷 브레이커
            if level in [BackpressureLevel.HIGH, BackpressureLevel.CRITICAL]:
                self._throttle_factor = 0.0  # 완전 차단
            else:
                self._throttle_factor = 1.0

        if self._throttle_factor < 1.0:
            self._stats['throttle_activations'] += 1

    def _add_metrics(self, metrics: BackpressureMetrics):
        """메트릭 히스토리에 추가"""
        self._metrics_history.append(metrics)

        # 히스토리 크기 제한
        if len(self._metrics_history) > self._max_history_size:
            self._metrics_history = self._metrics_history[-self._max_history_size//2:]

        # 메트릭 콜백 실행
        asyncio.create_task(self._execute_metrics_callbacks(metrics))

    async def _execute_level_change_callbacks(self, old_level: BackpressureLevel, new_level: BackpressureLevel):
        """레벨 변경 콜백 실행"""
        for callback in self._level_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_level, new_level)
                else:
                    callback(old_level, new_level)
            except Exception as e:
                logger.error(f"레벨 변경 콜백 실행 오류: {str(e)}")

    async def _execute_metrics_callbacks(self, metrics: BackpressureMetrics):
        """메트릭 콜백 실행"""
        for callback in self._metrics_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                logger.error(f"메트릭 콜백 실행 오류: {str(e)}")

    def should_throttle(self) -> bool:
        """스로틀링이 필요한지 확인"""
        return self._throttle_factor < 1.0

    def should_drop_data(self) -> bool:
        """데이터 드롭이 필요한지 확인"""
        return self._current_level in [BackpressureLevel.HIGH, BackpressureLevel.CRITICAL]

    async def get_throttle_delay(self) -> float:
        """스로틀링 지연 시간 계산"""
        if not self.should_throttle():
            return 0.0

        base_delay = 0.001  # 기본 1ms
        return base_delay * (1.0 - self._throttle_factor)

    def add_level_change_callback(self, callback: Callable):
        """레벨 변경 콜백 추가"""
        self._level_change_callbacks.append(callback)

    def add_metrics_callback(self, callback: Callable):
        """메트릭 콜백 추가"""
        self._metrics_callbacks.append(callback)

    def get_current_metrics(self) -> Optional[BackpressureMetrics]:
        """현재 메트릭 반환"""
        return self._metrics_history[-1] if self._metrics_history else None

    def get_metrics_history(self, duration_seconds: Optional[int] = None) -> List[BackpressureMetrics]:
        """메트릭 히스토리 반환"""
        if duration_seconds is None:
            return self._metrics_history.copy()

        cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)
        return [m for m in self._metrics_history if m.timestamp >= cutoff_time]

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        stats = self._stats.copy()
        stats.update({
            'current_level': self._current_level.value,
            'throttle_factor': self._throttle_factor,
            'ingestion_rate': self._ingestion_rate_calc.get_rate(),
            'processing_rate': self._processing_rate_calc.get_rate(),
            'memory_usage_mb': self._resource_monitor.memory_usage_mb,
            'cpu_usage_percent': self._resource_monitor.cpu_usage,
            'metrics_history_size': len(self._metrics_history)
        })
        return stats

    async def reset_stats(self):
        """통계 초기화"""
        self._stats = {
            'total_events_ingested': 0,
            'total_events_processed': 0,
            'total_events_dropped': 0,
            'backpressure_events': {level.value: 0 for level in BackpressureLevel},
            'throttle_activations': 0,
            'last_level_change': None
        }
        self._ingestion_rate_calc.reset()
        self._processing_rate_calc.reset()
        self._metrics_history.clear()

    @property
    def current_level(self) -> BackpressureLevel:
        """현재 백프레셔 레벨"""
        return self._current_level

    @property
    def throttle_factor(self) -> float:
        """현재 스로틀 팩터"""
        return self._throttle_factor