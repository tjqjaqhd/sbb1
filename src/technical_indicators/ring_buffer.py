"""
링 버퍼 기반 메모리 최적화 구조 구현

고정 크기 순환 버퍼를 활용한 효율적인 데이터 관리와 지표 계산을 제공합니다.
메모리 사용량을 최적화하고 가비지 컬렉션 부담을 최소화합니다.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Union, List, Callable
from abc import ABC, abstractmethod
import threading
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class RingBuffer:
    """
    고정 크기 순환 버퍼 구현

    OHLCV 데이터를 효율적으로 저장하고 관리합니다.
    메모리 재할당 없이 새 데이터를 추가할 수 있습니다.

    특징:
    - O(1) 시간 복잡도로 데이터 추가
    - 고정 메모리 사용량
    - 스레드 안전성 보장
    - numpy 배열 기반 효율적인 메모리 관리
    """

    def __init__(self, capacity: int = 1000, data_type: str = 'float64'):
        """
        링 버퍼 초기화

        Args:
            capacity (int): 버퍼의 최대 크기 (기본값: 1000)
            data_type (str): 데이터 타입 (기본값: 'float64')

        Raises:
            ValueError: capacity가 1보다 작은 경우
        """
        if capacity < 1:
            raise ValueError("버퍼 크기는 1 이상이어야 합니다")

        self._capacity = capacity
        self._data_type = data_type
        self._buffer = np.zeros(capacity, dtype=data_type)
        self._head = 0  # 다음 쓰기 위치
        self._size = 0  # 현재 저장된 데이터 수
        self._lock = threading.RLock()  # 스레드 안전성을 위한 락

        logger.debug(f"링 버퍼 생성: capacity={capacity}, dtype={data_type}")

    def append(self, value: Union[float, int]) -> None:
        """
        새 데이터를 버퍼에 추가

        Args:
            value: 추가할 데이터

        Note:
            버퍼가 가득 찬 경우 가장 오래된 데이터를 덮어씁니다.
        """
        with self._lock:
            self._buffer[self._head] = value
            self._head = (self._head + 1) % self._capacity

            if self._size < self._capacity:
                self._size += 1

    def extend(self, values: Union[List, np.ndarray]) -> None:
        """
        여러 데이터를 버퍼에 추가

        Args:
            values: 추가할 데이터 배열
        """
        values = np.asarray(values)
        for value in values:
            self.append(value)

    def get_data(self, count: Optional[int] = None) -> np.ndarray:
        """
        버퍼에서 데이터를 가져옴

        Args:
            count: 가져올 데이터 개수 (None이면 전체)

        Returns:
            np.ndarray: 시간순으로 정렬된 데이터 배열
        """
        with self._lock:
            if self._size == 0:
                return np.array([], dtype=self._data_type)

            if count is None:
                count = self._size
            else:
                count = min(count, self._size)

            if self._size < self._capacity:
                # 버퍼가 아직 가득 차지 않은 경우
                start_idx = max(0, self._size - count)
                return self._buffer[start_idx:self._size].copy()
            else:
                # 버퍼가 가득 찬 경우 (순환)
                if count >= self._capacity:
                    # 전체 데이터 요청
                    result = np.zeros(self._capacity, dtype=self._data_type)
                    result[:self._capacity - self._head] = self._buffer[self._head:]
                    result[self._capacity - self._head:] = self._buffer[:self._head]
                    return result
                else:
                    # 일부 데이터 요청
                    start_pos = (self._head - count) % self._capacity
                    if start_pos + count <= self._capacity:
                        return self._buffer[start_pos:start_pos + count].copy()
                    else:
                        # 순환 영역에 걸친 경우
                        part1_size = self._capacity - start_pos
                        part2_size = count - part1_size
                        result = np.zeros(count, dtype=self._data_type)
                        result[:part1_size] = self._buffer[start_pos:]
                        result[part1_size:] = self._buffer[:part2_size]
                        return result

    def get_latest(self) -> Optional[float]:
        """
        가장 최근 데이터 반환

        Returns:
            Optional[float]: 최근 데이터, 데이터가 없으면 None
        """
        with self._lock:
            if self._size == 0:
                return None

            latest_idx = (self._head - 1) % self._capacity
            return float(self._buffer[latest_idx])

    def get_window(self, start: int, end: Optional[int] = None) -> np.ndarray:
        """
        지정된 범위의 데이터 윈도우 반환

        Args:
            start (int): 시작 인덱스 (음수 허용, -1은 최신 데이터)
            end (Optional[int]): 종료 인덱스 (None이면 최신까지)

        Returns:
            np.ndarray: 윈도우 데이터
        """
        with self._lock:
            if self._size == 0:
                return np.array([], dtype=self._data_type)

            # 전체 데이터 가져오기
            full_data = self.get_data()

            # 음수 인덱스 처리
            if start < 0:
                start = len(full_data) + start
            if end is not None and end < 0:
                end = len(full_data) + end

            start = max(0, start)
            if end is None:
                end = len(full_data)
            else:
                end = min(end, len(full_data))

            if start >= end:
                return np.array([], dtype=self._data_type)

            return full_data[start:end]

    def is_full(self) -> bool:
        """버퍼가 가득 찼는지 확인"""
        with self._lock:
            return self._size >= self._capacity

    def __len__(self) -> int:
        """현재 저장된 데이터 수 반환"""
        with self._lock:
            return self._size

    @property
    def capacity(self) -> int:
        """버퍼 용량 반환"""
        return self._capacity

    @property
    def size(self) -> int:
        """현재 크기 반환"""
        with self._lock:
            return self._size

    def clear(self) -> None:
        """버퍼 초기화"""
        with self._lock:
            self._head = 0
            self._size = 0
            self._buffer.fill(0)

    def memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 정보 반환"""
        return {
            'buffer_size_bytes': self._buffer.nbytes,
            'buffer_size_mb': self._buffer.nbytes / 1024 / 1024,
            'capacity': self._capacity,
            'current_size': self._size,
            'data_type': self._data_type,
            'utilization_percent': (self._size / self._capacity) * 100
        }


class SlidingWindow:
    """
    슬라이딩 윈도우 최적화 클래스

    지정된 기간의 데이터 윈도우를 효율적으로 제공합니다.
    뷰(view) 방식으로 메모리 복사를 최소화합니다.

    특징:
    - O(1) 윈도우 이동
    - 메모리 복사 최소화
    - 다양한 윈도우 크기 지원
    - 지표 계산에 최적화
    """

    def __init__(self, ring_buffer: RingBuffer, window_size: int):
        """
        슬라이딩 윈도우 초기화

        Args:
            ring_buffer (RingBuffer): 데이터 소스 링 버퍼
            window_size (int): 윈도우 크기

        Raises:
            ValueError: window_size가 1보다 작거나 버퍼 용량보다 큰 경우
        """
        if window_size < 1:
            raise ValueError("윈도우 크기는 1 이상이어야 합니다")
        if window_size > ring_buffer.capacity:
            raise ValueError("윈도우 크기가 링 버퍼 용량보다 클 수 없습니다")

        self._ring_buffer = ring_buffer
        self._window_size = window_size
        self._cached_data = None
        self._last_buffer_size = 0

        logger.debug(f"슬라이딩 윈도우 생성: window_size={window_size}")

    def get_current_window(self) -> np.ndarray:
        """
        현재 윈도우 데이터 반환

        Returns:
            np.ndarray: 윈도우 데이터 (크기가 부족하면 사용 가능한 모든 데이터)
        """
        current_size = self._ring_buffer.size

        # 캐시 무효화 체크
        if (self._cached_data is None or
            current_size != self._last_buffer_size):
            self._update_cache()

        return self._cached_data

    def _update_cache(self) -> None:
        """캐시 업데이트"""
        current_size = self._ring_buffer.size
        if current_size == 0:
            self._cached_data = np.array([])
        else:
            window_size = min(self._window_size, current_size)
            self._cached_data = self._ring_buffer.get_data(window_size)

        self._last_buffer_size = current_size

    def get_window_at_offset(self, offset: int) -> np.ndarray:
        """
        지정된 오프셋에서의 윈도우 반환

        Args:
            offset (int): 현재 위치에서의 오프셋 (음수: 과거)

        Returns:
            np.ndarray: 윈도우 데이터
        """
        current_size = self._ring_buffer.size
        if current_size == 0:
            return np.array([])

        # 종료 위치 계산
        end_pos = current_size + offset
        if end_pos <= 0:
            return np.array([])

        start_pos = max(0, end_pos - self._window_size)
        return self._ring_buffer.get_window(start_pos, end_pos)

    def is_ready(self) -> bool:
        """윈도우가 준비되었는지 확인 (충분한 데이터가 있는지)"""
        return self._ring_buffer.size >= self._window_size

    @property
    def window_size(self) -> int:
        """윈도우 크기 반환"""
        return self._window_size

    @property
    def available_size(self) -> int:
        """현재 사용 가능한 윈도우 크기 반환"""
        return min(self._window_size, self._ring_buffer.size)


class CachedIndicatorEngine:
    """
    캐시된 지표 계산 엔진

    지표 계산 결과를 캐싱하여 중복 계산을 방지합니다.
    새 데이터 추가 시 증분 업데이트를 지원합니다.

    특징:
    - 지표 계산 결과 캐싱
    - 증분 업데이트 지원
    - 캐시 히트율 모니터링
    - 다양한 지표 함수 지원
    """

    def __init__(self, ring_buffer: RingBuffer):
        """
        캐시된 지표 엔진 초기화

        Args:
            ring_buffer (RingBuffer): 데이터 소스 링 버퍼
        """
        self._ring_buffer = ring_buffer
        self._cache = {}  # 지표별 캐시
        self._last_calculated_size = {}  # 지표별 마지막 계산된 데이터 크기
        self._calculation_functions = {}  # 지표별 계산 함수
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.RLock()

        logger.debug("캐시된 지표 엔진 초기화")

    def register_indicator(self, name: str, calculation_func: Callable,
                          window_size: int, **kwargs) -> None:
        """
        지표 등록

        Args:
            name (str): 지표 이름
            calculation_func (Callable): 계산 함수
            window_size (int): 필요한 윈도우 크기
            **kwargs: 계산 함수에 전달할 추가 매개변수
        """
        with self._lock:
            self._calculation_functions[name] = {
                'func': calculation_func,
                'window_size': window_size,
                'kwargs': kwargs
            }
            self._cache[name] = RingBuffer(capacity=self._ring_buffer.capacity)
            self._last_calculated_size[name] = 0

            logger.debug(f"지표 등록: {name}, window_size={window_size}")

    def calculate_indicator(self, name: str, force_recalculate: bool = False) -> Optional[float]:
        """
        지표 계산

        Args:
            name (str): 지표 이름
            force_recalculate (bool): 강제 재계산 여부

        Returns:
            Optional[float]: 계산된 지표 값, 계산할 수 없으면 None
        """
        with self._lock:
            if name not in self._calculation_functions:
                raise ValueError(f"등록되지 않은 지표: {name}")

            current_size = self._ring_buffer.size
            indicator_info = self._calculation_functions[name]
            window_size = indicator_info['window_size']

            # 충분한 데이터가 있는지 확인
            if current_size < window_size:
                return None

            # 캐시 확인
            last_size = self._last_calculated_size[name]
            if not force_recalculate and current_size == last_size:
                # 캐시 히트
                self._cache_hits += 1
                cached_data = self._cache[name].get_latest()
                return cached_data

            # 캐시 미스 또는 강제 재계산
            self._cache_misses += 1

            # 새로운 값들만 계산 (증분 업데이트)
            if current_size > last_size and not force_recalculate:
                # 증분 계산 시도
                new_values = self._calculate_incremental(name, last_size, current_size)
                if new_values is not None:
                    # 증분 계산 성공
                    for value in new_values:
                        self._cache[name].append(value)
                    self._last_calculated_size[name] = current_size
                    return self._cache[name].get_latest()

            # 전체 재계산
            return self._calculate_full(name)

    def _calculate_incremental(self, name: str, last_size: int, current_size: int) -> Optional[List[float]]:
        """
        증분 계산 시도

        Args:
            name (str): 지표 이름
            last_size (int): 이전 데이터 크기
            current_size (int): 현재 데이터 크기

        Returns:
            Optional[List[float]]: 새로 계산된 값들, 증분 계산 불가능하면 None
        """
        indicator_info = self._calculation_functions[name]
        window_size = indicator_info['window_size']

        # 증분 계산이 가능한 경우만 처리 (예: 이동평균 등)
        if hasattr(indicator_info['func'], 'supports_incremental'):
            if indicator_info['func'].supports_incremental:
                new_values = []
                for i in range(last_size, current_size):
                    if i >= window_size - 1:
                        window_data = self._ring_buffer.get_window(i - window_size + 1, i + 1)
                        value = indicator_info['func'](window_data, **indicator_info['kwargs'])
                        new_values.append(value)
                return new_values

        return None

    def _calculate_full(self, name: str) -> Optional[float]:
        """
        전체 재계산

        Args:
            name (str): 지표 이름

        Returns:
            Optional[float]: 계산된 최신 지표 값
        """
        indicator_info = self._calculation_functions[name]
        window_size = indicator_info['window_size']
        current_size = self._ring_buffer.size

        if current_size < window_size:
            return None

        # 현재 윈도우에 대해서만 계산
        window_data = self._ring_buffer.get_data(window_size)
        value = indicator_info['func'](window_data, **indicator_info['kwargs'])

        # 캐시에 저장
        if self._cache[name].size == 0:
            self._cache[name].append(value)
        else:
            # 기존 캐시 업데이트
            self._cache[name].clear()
            self._cache[name].append(value)

        self._last_calculated_size[name] = current_size
        return value

    def get_indicator_history(self, name: str, count: Optional[int] = None) -> np.ndarray:
        """
        지표 히스토리 반환

        Args:
            name (str): 지표 이름
            count (Optional[int]): 반환할 데이터 개수

        Returns:
            np.ndarray: 지표 히스토리
        """
        with self._lock:
            if name not in self._cache:
                return np.array([])

            return self._cache[name].get_data(count)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 반환

        Returns:
            Dict[str, Any]: 캐시 통계 정보
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': hit_rate,
            'total_requests': total_requests,
            'registered_indicators': list(self._calculation_functions.keys()),
            'cache_memory_usage_mb': sum(
                cache.memory_usage()['buffer_size_mb']
                for cache in self._cache.values()
            )
        }

    def clear_cache(self, indicator_name: Optional[str] = None) -> None:
        """
        캐시 초기화

        Args:
            indicator_name (Optional[str]): 특정 지표만 초기화 (None이면 전체)
        """
        with self._lock:
            if indicator_name is None:
                # 전체 캐시 초기화
                for cache in self._cache.values():
                    cache.clear()
                self._last_calculated_size.clear()
                self._cache_hits = 0
                self._cache_misses = 0
                logger.info("전체 캐시 초기화")
            else:
                # 특정 지표 캐시 초기화
                if indicator_name in self._cache:
                    self._cache[indicator_name].clear()
                    self._last_calculated_size[indicator_name] = 0
                    logger.info(f"지표 캐시 초기화: {indicator_name}")


# 지표 계산 함수들 (캐시 엔진과 함께 사용)
def sma_function(data: np.ndarray, **kwargs) -> float:
    """단순 이동평균 계산 함수"""
    return np.mean(data)

def ema_function(data: np.ndarray, alpha: float = None, period: int = None, **kwargs) -> float:
    """지수 이동평균 계산 함수"""
    if alpha is None and period is not None:
        alpha = 2.0 / (period + 1)
    elif alpha is None:
        alpha = 0.1  # 기본값

    if len(data) < period if period else len(data) < 1:
        return np.mean(data)

    # 첫 번째 EMA는 SMA로 시작 (기존 구현과 일치)
    if len(data) >= period:
        # 첫 period개의 평균을 초기 EMA로 사용
        result = np.mean(data[:period])

        # 나머지 데이터에 대해 EMA 계산
        for i in range(period, len(data)):
            result = alpha * data[i] + (1 - alpha) * result

        return result
    else:
        # 데이터가 부족하면 SMA 반환
        return np.mean(data)

# 증분 계산 지원 표시
sma_function.supports_incremental = True
ema_function.supports_incremental = False  # EMA는 증분 계산이 복잡하므로 비활성화


class OptimizedTechnicalIndicatorEngine:
    """
    링 버퍼 기반 최적화된 기술적 지표 엔진

    기존 TechnicalIndicatorEngine을 링 버퍼 기반으로 최적화한 버전입니다.
    메모리 사용량과 계산 성능이 크게 향상되었습니다.
    """

    def __init__(self, capacity: int = 1000):
        """
        최적화된 지표 엔진 초기화

        Args:
            capacity (int): 링 버퍼 용량
        """
        self._ring_buffer = RingBuffer(capacity=capacity)
        self._cache_engine = CachedIndicatorEngine(self._ring_buffer)
        self._indicators = {}  # 등록된 지표들

        # 기본 지표들 등록
        self._register_default_indicators()

        logger.info(f"최적화된 기술적 지표 엔진 초기화: capacity={capacity}")

    def _register_default_indicators(self):
        """기본 지표들 등록"""
        # SMA 지표들
        for period in [5, 10, 20, 50, 200]:
            self._cache_engine.register_indicator(
                f'sma_{period}', sma_function, period
            )

        # EMA 지표들
        for period in [5, 10, 12, 20, 26, 50]:
            self._cache_engine.register_indicator(
                f'ema_{period}', ema_function, period, period=period
            )

    def add_data(self, value: Union[float, int]) -> Dict[str, Optional[float]]:
        """
        새 데이터 추가 및 모든 지표 계산

        Args:
            value: 새로운 데이터 값

        Returns:
            Dict[str, Optional[float]]: 계산된 모든 지표 값들
        """
        self._ring_buffer.append(value)

        # 모든 등록된 지표 계산
        results = {}
        for indicator_name in self._cache_engine._calculation_functions.keys():
            results[indicator_name] = self._cache_engine.calculate_indicator(indicator_name)

        return results

    def get_indicator(self, name: str) -> Optional[float]:
        """
        특정 지표의 현재 값 반환

        Args:
            name (str): 지표 이름

        Returns:
            Optional[float]: 지표 값
        """
        return self._cache_engine.calculate_indicator(name)

    def get_indicator_history(self, name: str, count: Optional[int] = None) -> np.ndarray:
        """
        지표 히스토리 반환

        Args:
            name (str): 지표 이름
            count (Optional[int]): 반환할 데이터 개수

        Returns:
            np.ndarray: 지표 히스토리
        """
        return self._cache_engine.get_indicator_history(name, count)

    def register_custom_indicator(self, name: str, calculation_func: Callable,
                                 window_size: int, **kwargs) -> None:
        """
        사용자 정의 지표 등록

        Args:
            name (str): 지표 이름
            calculation_func (Callable): 계산 함수
            window_size (int): 필요한 윈도우 크기
            **kwargs: 계산 함수에 전달할 추가 매개변수
        """
        self._cache_engine.register_indicator(name, calculation_func, window_size, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        """
        엔진 상태 정보 반환

        Returns:
            Dict[str, Any]: 상태 정보
        """
        buffer_stats = self._ring_buffer.memory_usage()
        cache_stats = self._cache_engine.get_cache_stats()

        return {
            'buffer_capacity': self._ring_buffer.capacity,
            'buffer_size': self._ring_buffer.size,
            'buffer_utilization_percent': buffer_stats['utilization_percent'],
            'memory_usage_mb': buffer_stats['buffer_size_mb'] + cache_stats['cache_memory_usage_mb'],
            'cache_hit_rate_percent': cache_stats['hit_rate_percent'],
            'registered_indicators': cache_stats['registered_indicators'],
            'is_ready': self._ring_buffer.size > 0
        }

    def clear(self) -> None:
        """모든 데이터와 캐시 초기화"""
        self._ring_buffer.clear()
        self._cache_engine.clear_cache()
        logger.info("최적화된 지표 엔진 초기화 완료")