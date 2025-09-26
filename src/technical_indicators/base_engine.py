"""
기술적 지표 계산을 위한 기본 엔진 클래스

모든 기술적 지표 클래스가 상속받을 기본 클래스를 정의합니다.
공통 기능과 인터페이스를 제공하여 일관성 있는 구현을 보장합니다.
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class TechnicalIndicatorEngine(ABC):
    """
    모든 기술적 지표 계산 클래스의 기본 클래스

    공통 기능:
    - 데이터 검증
    - 메모리 관리
    - 실시간 업데이트 지원
    - 예외 처리
    """

    def __init__(self, period: int, max_history: int = 10000):
        """
        기술적 지표 엔진 초기화

        Args:
            period (int): 계산 기간 (예: 20일 이동평균의 경우 20)
            max_history (int): 최대 보관할 데이터 개수 (메모리 효율성)

        Raises:
            ValueError: period가 1보다 작거나 max_history보다 큰 경우
        """
        if period < 1:
            raise ValueError("계산 기간은 1 이상이어야 합니다")
        if period > max_history:
            raise ValueError("계산 기간이 최대 히스토리보다 클 수 없습니다")

        self.period = period
        self.max_history = max_history
        self.data_history = np.array([])
        self.indicator_history = np.array([])
        self.is_warmed_up = False

        logger.info(f"{self.__class__.__name__} 초기화: period={period}, max_history={max_history}")

    def _validate_input(self, data: Union[float, List[float], np.ndarray]) -> np.ndarray:
        """
        입력 데이터를 검증하고 numpy 배열로 변환

        Args:
            data: 검증할 데이터

        Returns:
            np.ndarray: 검증된 numpy 배열

        Raises:
            ValueError: 잘못된 데이터 형식이나 값이 포함된 경우
        """
        try:
            if isinstance(data, (int, float)):
                data_array = np.array([data], dtype=np.float64)
            else:
                data_array = np.array(data, dtype=np.float64)

            if data_array.size == 0:
                raise ValueError("빈 데이터는 처리할 수 없습니다")

            # NaN이나 무한값 검사
            if np.any(np.isnan(data_array)):
                logger.warning("NaN 값이 포함된 데이터가 감지되었습니다")
                # NaN 값을 이전 값으로 대체하거나 제거
                data_array = data_array[~np.isnan(data_array)]
                if data_array.size == 0:
                    raise ValueError("모든 데이터가 NaN입니다")

            if np.any(np.isinf(data_array)):
                raise ValueError("무한값이 포함된 데이터는 처리할 수 없습니다")

            return data_array

        except Exception as e:
            logger.error(f"데이터 검증 실패: {e}")
            raise ValueError(f"잘못된 입력 데이터: {e}")

    def _manage_memory(self):
        """
        메모리 사용량을 관리하여 지정된 히스토리 크기를 유지
        """
        if len(self.data_history) > self.max_history:
            # 오래된 데이터 제거
            excess = len(self.data_history) - self.max_history
            self.data_history = self.data_history[excess:]
            if len(self.indicator_history) > 0:
                self.indicator_history = self.indicator_history[excess:]

            logger.debug(f"메모리 관리: {excess}개 데이터 제거")

    def add_data(self, new_data: Union[float, List[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        새로운 데이터를 추가하고 지표를 계산

        Args:
            new_data: 추가할 새로운 데이터

        Returns:
            계산된 지표 값(들)
        """
        validated_data = self._validate_input(new_data)

        # 기존 데이터에 새 데이터 추가
        if self.data_history.size == 0:
            self.data_history = validated_data
        else:
            self.data_history = np.concatenate([self.data_history, validated_data])

        # 메모리 관리
        self._manage_memory()

        # 지표 계산
        new_indicators = self._calculate_indicator()

        # 워밍업 상태 확인
        if not self.is_warmed_up and len(self.data_history) >= self.period:
            self.is_warmed_up = True
            logger.info(f"{self.__class__.__name__} 워밍업 완료: {len(self.data_history)}개 데이터")

        return new_indicators

    def get_current_value(self) -> Optional[float]:
        """
        현재 지표 값 반환

        Returns:
            float: 현재 지표 값, 계산할 수 없으면 None
        """
        if not self.is_warmed_up or len(self.indicator_history) == 0:
            return None
        return float(self.indicator_history[-1])

    def get_history(self, count: Optional[int] = None) -> np.ndarray:
        """
        지표 히스토리 반환

        Args:
            count: 반환할 최근 데이터 개수 (None이면 전체)

        Returns:
            np.ndarray: 지표 히스토리 배열
        """
        if count is None:
            return self.indicator_history.copy()
        else:
            return self.indicator_history[-count:].copy() if len(self.indicator_history) >= count else self.indicator_history.copy()

    def reset(self):
        """
        엔진 상태를 초기화
        """
        self.data_history = np.array([])
        self.indicator_history = np.array([])
        self.is_warmed_up = False
        logger.info(f"{self.__class__.__name__} 상태 초기화")

    def get_status(self) -> Dict[str, Any]:
        """
        현재 엔진 상태 정보 반환

        Returns:
            Dict: 상태 정보 딕셔너리
        """
        return {
            'period': self.period,
            'max_history': self.max_history,
            'data_count': len(self.data_history),
            'indicator_count': len(self.indicator_history),
            'is_warmed_up': self.is_warmed_up,
            'current_value': self.get_current_value(),
            'memory_usage_mb': (self.data_history.nbytes + self.indicator_history.nbytes) / 1024 / 1024
        }

    @abstractmethod
    def _calculate_indicator(self) -> np.ndarray:
        """
        실제 지표 계산을 수행하는 추상 메서드
        각 구체적인 지표 클래스에서 구현해야 함

        Returns:
            np.ndarray: 계산된 지표 값들
        """
        pass

    def __str__(self) -> str:
        status = self.get_status()
        return f"{self.__class__.__name__}(period={status['period']}, warmed_up={status['is_warmed_up']}, current={status['current_value']})"

    def __repr__(self) -> str:
        return self.__str__()