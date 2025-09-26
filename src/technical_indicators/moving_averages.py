"""
이동평균 지표 구현

단순 이동평균(SMA)과 지수 이동평균(EMA)을 계산하는 클래스들을 제공합니다.
실시간 데이터 처리와 효율적인 증분 계산에 최적화되어 있습니다.
"""

import numpy as np
import logging
from typing import Union, List, Optional

from .base_engine import TechnicalIndicatorEngine

logger = logging.getLogger(__name__)


class SimpleMovingAverage(TechnicalIndicatorEngine):
    """
    단순 이동평균(Simple Moving Average, SMA) 계산 클래스

    SMA 공식: SMA(n) = (P1 + P2 + ... + Pn) / n
    - P1, P2, ..., Pn: 최근 n개의 가격
    - n: 계산 기간

    특징:
    - 모든 가격에 동일한 가중치 적용
    - 과거 데이터 변화에 즉시 반응
    - 계산이 단순하고 직관적
    """

    def __init__(self, period: int, max_history: int = 10000):
        """
        단순 이동평균 계산기 초기화

        Args:
            period (int): 이동평균 계산 기간
            max_history (int): 최대 보관할 데이터 개수

        Example:
            >>> sma_20 = SimpleMovingAverage(period=20)
            >>> sma_20.add_data([100, 101, 102, 103, 104])
        """
        super().__init__(period, max_history)
        logger.info(f"SMA 계산기 생성: {period}일 기간")

    def _calculate_indicator(self) -> np.ndarray:
        """
        단순 이동평균 계산

        Returns:
            np.ndarray: 계산된 SMA 값들

        Note:
            데이터 길이가 period보다 짧으면 빈 배열 반환
        """
        if len(self.data_history) < self.period:
            # 워밍업 기간 동안은 빈 배열 반환
            return np.array([])

        # 모든 가능한 SMA 값 계산
        sma_values = []
        data_len = len(self.data_history)

        for i in range(self.period - 1, data_len):
            # i번째 위치에서 period개 데이터의 평균 계산
            window_data = self.data_history[i - self.period + 1:i + 1]
            sma_value = np.mean(window_data)
            sma_values.append(sma_value)

        self.indicator_history = np.array(sma_values)
        return self.indicator_history

    def add_data_incremental(self, new_price: float) -> Optional[float]:
        """
        단일 데이터 포인트를 추가하고 증분 계산 수행
        메모리 효율성을 위해 최적화된 버전

        Args:
            new_price (float): 새로운 가격 데이터

        Returns:
            Optional[float]: 계산된 새 SMA 값, 워밍업 중이면 None

        Example:
            >>> sma = SimpleMovingAverage(5)
            >>> for price in [100, 101, 102, 103, 104, 105]:
            ...     result = sma.add_data_incremental(price)
            ...     if result:
            ...         print(f"SMA: {result:.2f}")
        """
        validated_price = self._validate_input(new_price)[0]

        # 데이터 히스토리에 추가
        if self.data_history.size == 0:
            self.data_history = np.array([validated_price])
        else:
            self.data_history = np.append(self.data_history, validated_price)

        self._manage_memory()

        if len(self.data_history) < self.period:
            return None

        # 새 SMA 값 계산
        window_data = self.data_history[-self.period:]
        new_sma = np.mean(window_data)

        # 지표 히스토리 업데이트
        if self.indicator_history.size == 0:
            self.indicator_history = np.array([new_sma])
        else:
            self.indicator_history = np.append(self.indicator_history, new_sma)

        if not self.is_warmed_up:
            self.is_warmed_up = True
            logger.info(f"SMA({self.period}) 워밍업 완료")

        return new_sma


class ExponentialMovingAverage(TechnicalIndicatorEngine):
    """
    지수 이동평균(Exponential Moving Average, EMA) 계산 클래스

    EMA 공식: EMA(t) = α × P(t) + (1-α) × EMA(t-1)
    - α (알파): 평활화 계수 = 2 / (period + 1)
    - P(t): 현재 가격
    - EMA(t-1): 이전 EMA 값

    특징:
    - 최근 가격에 더 큰 가중치 부여
    - 가격 변화에 빠르게 반응
    - 증분 계산으로 효율적
    """

    def __init__(self, period: int, max_history: int = 10000, initial_ema: Optional[float] = None):
        """
        지수 이동평균 계산기 초기화

        Args:
            period (int): 이동평균 계산 기간
            max_history (int): 최대 보관할 데이터 개수
            initial_ema (Optional[float]): 초기 EMA 값 (None이면 첫 SMA 사용)

        Example:
            >>> ema_12 = ExponentialMovingAverage(period=12)
            >>> ema_12.add_data([100, 101, 102, 103, 104])
        """
        super().__init__(period, max_history)
        self.alpha = 2.0 / (period + 1)  # 평활화 계수
        self.initial_ema = initial_ema
        self.previous_ema = None

        logger.info(f"EMA 계산기 생성: {period}일 기간, α={self.alpha:.4f}")

    def _calculate_indicator(self) -> np.ndarray:
        """
        지수 이동평균 계산

        Returns:
            np.ndarray: 계산된 EMA 값들

        Note:
            첫 번째 EMA는 SMA로 초기화됨
        """
        if len(self.data_history) < self.period:
            return np.array([])

        ema_values = []
        data_len = len(self.data_history)

        # 첫 번째 EMA는 SMA로 시작
        if self.initial_ema is not None:
            current_ema = self.initial_ema
            start_idx = 0  # 모든 데이터에 대해 EMA 계산
            if data_len > 0:
                # 첫 번째 값부터 EMA 계산
                for i in range(data_len):
                    current_price = self.data_history[i]
                    current_ema = self.alpha * current_price + (1 - self.alpha) * current_ema
                    ema_values.append(current_ema)
        else:
            # 첫 period개 데이터의 SMA를 첫 EMA로 사용
            first_sma = np.mean(self.data_history[:self.period])
            current_ema = first_sma
            ema_values.append(current_ema)

            # 나머지 EMA 값들 계산
            for i in range(self.period, data_len):
                current_price = self.data_history[i]
                current_ema = self.alpha * current_price + (1 - self.alpha) * current_ema
                ema_values.append(current_ema)

        self.indicator_history = np.array(ema_values)
        self.previous_ema = current_ema if ema_values else None
        return self.indicator_history

    def add_data_incremental(self, new_price: float) -> Optional[float]:
        """
        단일 데이터 포인트를 추가하고 증분 EMA 계산 수행

        Args:
            new_price (float): 새로운 가격 데이터

        Returns:
            Optional[float]: 계산된 새 EMA 값, 워밍업 중이면 None

        Example:
            >>> ema = ExponentialMovingAverage(12)
            >>> for price in [100, 101, 102, 103, 104]:
            ...     result = ema.add_data_incremental(price)
            ...     if result:
            ...         print(f"EMA: {result:.2f}")
        """
        validated_price = self._validate_input(new_price)[0]

        # 데이터 히스토리에 추가
        if self.data_history.size == 0:
            self.data_history = np.array([validated_price])
        else:
            self.data_history = np.append(self.data_history, validated_price)

        self._manage_memory()

        if len(self.data_history) < self.period:
            return None

        # EMA 계산
        if self.previous_ema is None:
            # 첫 번째 EMA는 SMA로 초기화
            if self.initial_ema is not None:
                self.previous_ema = self.initial_ema
            else:
                window_data = self.data_history[-self.period:]
                self.previous_ema = np.mean(window_data)

        # 새 EMA 계산
        new_ema = self.alpha * validated_price + (1 - self.alpha) * self.previous_ema
        self.previous_ema = new_ema

        # 지표 히스토리 업데이트
        if self.indicator_history.size == 0:
            self.indicator_history = np.array([new_ema])
        else:
            self.indicator_history = np.append(self.indicator_history, new_ema)

        if not self.is_warmed_up:
            self.is_warmed_up = True
            logger.info(f"EMA({self.period}) 워밍업 완료, 첫 값: {new_ema:.4f}")

        return new_ema

    def get_alpha(self) -> float:
        """
        현재 평활화 계수(알파) 반환

        Returns:
            float: 평활화 계수
        """
        return self.alpha

    def set_initial_ema(self, value: float):
        """
        초기 EMA 값 설정

        Args:
            value (float): 설정할 초기 EMA 값

        Note:
            이미 데이터가 있는 경우 재계산이 필요할 수 있습니다.
        """
        self.initial_ema = value
        if len(self.data_history) >= self.period:
            logger.info(f"초기 EMA 값 변경: {value}, 재계산 필요할 수 있음")


class MovingAverageFactory:
    """
    이동평균 객체를 생성하는 팩토리 클래스

    다양한 설정으로 이동평균 객체를 쉽게 생성할 수 있습니다.
    """

    @staticmethod
    def create_sma(period: int, max_history: int = 10000) -> SimpleMovingAverage:
        """
        단순 이동평균 객체 생성

        Args:
            period (int): 계산 기간
            max_history (int): 최대 히스토리

        Returns:
            SimpleMovingAverage: 생성된 SMA 객체
        """
        return SimpleMovingAverage(period, max_history)

    @staticmethod
    def create_ema(period: int, max_history: int = 10000, initial_ema: Optional[float] = None) -> ExponentialMovingAverage:
        """
        지수 이동평균 객체 생성

        Args:
            period (int): 계산 기간
            max_history (int): 최대 히스토리
            initial_ema (Optional[float]): 초기 EMA 값

        Returns:
            ExponentialMovingAverage: 생성된 EMA 객체
        """
        return ExponentialMovingAverage(period, max_history, initial_ema)

    @staticmethod
    def create_standard_set(max_history: int = 10000) -> dict:
        """
        표준 이동평균 세트 생성 (5, 10, 20, 50, 200일)

        Args:
            max_history (int): 최대 히스토리

        Returns:
            dict: 생성된 이동평균 객체들의 딕셔너리

        Example:
            >>> ma_set = MovingAverageFactory.create_standard_set()
            >>> sma_20 = ma_set['sma_20']
            >>> ema_12 = ma_set['ema_12']
        """
        return {
            'sma_5': SimpleMovingAverage(5, max_history),
            'sma_10': SimpleMovingAverage(10, max_history),
            'sma_20': SimpleMovingAverage(20, max_history),
            'sma_50': SimpleMovingAverage(50, max_history),
            'sma_200': SimpleMovingAverage(200, max_history),
            'ema_5': ExponentialMovingAverage(5, max_history),
            'ema_10': ExponentialMovingAverage(10, max_history),
            'ema_12': ExponentialMovingAverage(12, max_history),
            'ema_20': ExponentialMovingAverage(20, max_history),
            'ema_26': ExponentialMovingAverage(26, max_history),
            'ema_50': ExponentialMovingAverage(50, max_history),
        }