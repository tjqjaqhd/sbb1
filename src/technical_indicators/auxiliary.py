"""
보조 기술지표 구현

Williams %R, CCI(Commodity Channel Index), ROC(Rate of Change), ATR(Average True Range) 등
다양한 보조 기술지표들을 계산하는 클래스들을 제공합니다.
기존 TechnicalIndicatorEngine을 상속받아 실시간 데이터 처리와 효율적인 증분 계산에 최적화되어 있습니다.
"""

import numpy as np
import logging
from typing import Union, List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

from .base_engine import TechnicalIndicatorEngine

logger = logging.getLogger(__name__)


class WilliamsRIndicator(TechnicalIndicatorEngine):
    """
    Williams %R 지표 계산 클래스

    Williams %R은 모멘텀 오실레이터로, 현재 종가가 특정 기간의 최고가-최저가 범위에서
    어디에 위치하는지를 나타내는 지표입니다.

    계산 공식:
    %R = ((Highest High - Close) / (Highest High - Lowest Low)) × -100

    특징:
    - -100~0 범위의 값
    - -20 이상: 과매수 구간
    - -80 이하: 과매도 구간
    - 역방향 오실레이터 (낮을수록 강세)
    """

    def __init__(self, period: int = 14, max_history: int = 10000,
                 overbought_threshold: float = -20.0, oversold_threshold: float = -80.0):
        """
        Williams %R 지표 계산기 초기화

        Args:
            period (int): 계산 기간 (기본값: 14)
            max_history (int): 최대 보관할 데이터 개수
            overbought_threshold (float): 과매수 임계값 (기본값: -20)
            oversold_threshold (float): 과매도 임계값 (기본값: -80)

        Example:
            >>> williams_r = WilliamsRIndicator(period=14)
            >>> # HLC 데이터로 계산
            >>> williams_r.add_hlc_data(high=102, low=98, close=100)
        """
        super().__init__(period, max_history)

        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

        # HLC 데이터 저장
        self.high_history = np.array([])
        self.low_history = np.array([])
        self.close_history = np.array([])

        logger.info(f"Williams %R 지표 생성: {period}일 기간, 과매수={overbought_threshold}, 과매도={oversold_threshold}")

    def add_hlc_data(self, high: float, low: float, close: float) -> Optional[float]:
        """
        HLC(High-Low-Close) 데이터를 추가하고 Williams %R 계산

        Args:
            high (float): 고가
            low (float): 저가
            close (float): 종가

        Returns:
            Optional[float]: 계산된 Williams %R 값, 워밍업 중이면 None
        """
        # 데이터 검증
        validated_high = self._validate_input(high)[0]
        validated_low = self._validate_input(low)[0]
        validated_close = self._validate_input(close)[0]

        # 가격 논리 검증
        if validated_high < validated_low:
            raise ValueError(f"고가({validated_high})가 저가({validated_low})보다 낮습니다")
        if not (validated_low <= validated_close <= validated_high):
            logger.warning(f"종가({validated_close})가 고가-저가 범위를 벗어남")

        # HLC 히스토리에 추가
        self.high_history = np.append(self.high_history, validated_high)
        self.low_history = np.append(self.low_history, validated_low)
        self.close_history = np.append(self.close_history, validated_close)

        # 기본 data_history는 close 가격으로 설정 (기본 클래스 호환성)
        if self.data_history.size == 0:
            self.data_history = np.array([validated_close])
        else:
            self.data_history = np.append(self.data_history, validated_close)

        # 메모리 관리
        self._manage_memory()

        # Williams %R 계산
        williams_r_value = self._calculate_current_williams_r()

        if williams_r_value is not None:
            if self.indicator_history.size == 0:
                self.indicator_history = np.array([williams_r_value])
            else:
                self.indicator_history = np.append(self.indicator_history, williams_r_value)

            if not self.is_warmed_up and len(self.close_history) >= self.period:
                self.is_warmed_up = True
                logger.info(f"Williams %R({self.period}) 워밍업 완료, 첫 값: {williams_r_value:.2f}")

        return williams_r_value

    def _calculate_current_williams_r(self) -> Optional[float]:
        """현재 Williams %R 값 계산"""
        if len(self.close_history) < self.period:
            return None

        # 최근 period 동안의 HLC 데이터
        recent_high = self.high_history[-self.period:]
        recent_low = self.low_history[-self.period:]
        current_close = self.close_history[-1]

        # Highest High와 Lowest Low 계산
        highest_high = np.max(recent_high)
        lowest_low = np.min(recent_low)

        # Williams %R 계산
        if highest_high == lowest_low:
            # 가격 변동이 없는 경우
            return -50.0  # 중간값 반환

        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100.0
        return williams_r

    def _calculate_indicator(self) -> np.ndarray:
        """
        전체 Williams %R 지표 계산 (기본 클래스 호환성)

        Returns:
            np.ndarray: 계산된 Williams %R 값들
        """
        if len(self.close_history) < self.period:
            return np.array([])

        williams_r_values = []

        for i in range(self.period - 1, len(self.close_history)):
            # 해당 기간의 HLC 데이터
            period_high = self.high_history[i - self.period + 1:i + 1]
            period_low = self.low_history[i - self.period + 1:i + 1]
            current_close = self.close_history[i]

            # Highest High와 Lowest Low
            highest_high = np.max(period_high)
            lowest_low = np.min(period_low)

            # Williams %R 계산
            if highest_high == lowest_low:
                williams_r = -50.0
            else:
                williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100.0

            williams_r_values.append(williams_r)

        self.indicator_history = np.array(williams_r_values)
        return self.indicator_history

    def _manage_memory(self):
        """메모리 관리 - HLC 히스토리 포함"""
        super()._manage_memory()

        if len(self.high_history) > self.max_history:
            excess = len(self.high_history) - self.max_history
            self.high_history = self.high_history[excess:]
            self.low_history = self.low_history[excess:]
            self.close_history = self.close_history[excess:]

    def get_signal(self) -> Dict[str, Any]:
        """
        현재 Williams %R 기반 매매 신호 반환

        Returns:
            Dict: 매매 신호 정보
        """
        current_williams_r = self.get_current_value()
        if current_williams_r is None:
            return {
                'signal': 'neutral',
                'strength': 0,
                'williams_r_value': None,
                'description': 'Williams %R 계산 불가'
            }

        # 신호 분류 (Williams %R은 역방향 지표)
        if current_williams_r >= self.overbought_threshold:
            if current_williams_r >= -10:
                signal = 'strong_sell'
                strength = min(10, int((-current_williams_r) / 2) + 8)
                description = f'극도 과매수 (Williams %R: {current_williams_r:.1f})'
            else:
                signal = 'sell'
                strength = min(8, int((self.overbought_threshold - current_williams_r) / 5) + 6)
                description = f'과매수 (Williams %R: {current_williams_r:.1f})'
        elif current_williams_r <= self.oversold_threshold:
            if current_williams_r <= -90:
                signal = 'strong_buy'
                strength = min(10, int((-90 - current_williams_r) / 2) + 8)
                description = f'극도 과매도 (Williams %R: {current_williams_r:.1f})'
            else:
                signal = 'buy'
                strength = min(8, int((current_williams_r - self.oversold_threshold) / 5) + 6)
                description = f'과매도 (Williams %R: {current_williams_r:.1f})'
        else:
            signal = 'neutral'
            strength = 5
            description = f'중립 (Williams %R: {current_williams_r:.1f})'

        return {
            'signal': signal,
            'strength': strength,
            'williams_r_value': current_williams_r,
            'description': description,
            'overbought_threshold': self.overbought_threshold,
            'oversold_threshold': self.oversold_threshold
        }

    def reset(self):
        """Williams %R 상태 초기화"""
        super().reset()
        self.high_history = np.array([])
        self.low_history = np.array([])
        self.close_history = np.array([])


class CCIIndicator(TechnicalIndicatorEngine):
    """
    CCI(Commodity Channel Index) 지표 계산 클래스

    CCI는 현재 가격이 통계적 평균에서 얼마나 벗어났는지를 측정하는 모멘텀 오실레이터입니다.

    계산 공식:
    1. Typical Price = (High + Low + Close) / 3
    2. SMA(Typical Price) = period 기간의 Typical Price 단순이동평균
    3. Mean Deviation = Σ|Typical Price - SMA(Typical Price)| / period
    4. CCI = (Typical Price - SMA(Typical Price)) / (0.015 × Mean Deviation)

    특징:
    - 무제한 범위 (보통 -100 ~ +100 범위에서 대부분 움직임)
    - +100 이상: 과매수 구간
    - -100 이하: 과매도 구간
    - 발산 패턴 분석 가능
    """

    def __init__(self, period: int = 20, max_history: int = 10000,
                 overbought_threshold: float = 100.0, oversold_threshold: float = -100.0,
                 constant: float = 0.015):
        """
        CCI 지표 계산기 초기화

        Args:
            period (int): 계산 기간 (기본값: 20)
            max_history (int): 최대 보관할 데이터 개수
            overbought_threshold (float): 과매수 임계값 (기본값: 100)
            oversold_threshold (float): 과매도 임계값 (기본값: -100)
            constant (float): CCI 계산 상수 (기본값: 0.015)

        Example:
            >>> cci = CCIIndicator(period=20)
            >>> cci.add_hlc_data(high=102, low=98, close=100)
        """
        super().__init__(period, max_history)

        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.constant = constant

        # HLC 데이터 저장
        self.high_history = np.array([])
        self.low_history = np.array([])
        self.close_history = np.array([])
        self.typical_price_history = np.array([])

        logger.info(f"CCI 지표 생성: {period}일 기간, 과매수={overbought_threshold}, 과매도={oversold_threshold}")

    def add_hlc_data(self, high: float, low: float, close: float) -> Optional[float]:
        """
        HLC(High-Low-Close) 데이터를 추가하고 CCI 계산

        Args:
            high (float): 고가
            low (float): 저가
            close (float): 종가

        Returns:
            Optional[float]: 계산된 CCI 값, 워밍업 중이면 None
        """
        # 데이터 검증
        validated_high = self._validate_input(high)[0]
        validated_low = self._validate_input(low)[0]
        validated_close = self._validate_input(close)[0]

        # 가격 논리 검증
        if validated_high < validated_low:
            raise ValueError(f"고가({validated_high})가 저가({validated_low})보다 낮습니다")
        if not (validated_low <= validated_close <= validated_high):
            logger.warning(f"종가({validated_close})가 고가-저가 범위를 벗어남")

        # HLC 히스토리에 추가
        self.high_history = np.append(self.high_history, validated_high)
        self.low_history = np.append(self.low_history, validated_low)
        self.close_history = np.append(self.close_history, validated_close)

        # Typical Price 계산
        typical_price = (validated_high + validated_low + validated_close) / 3.0
        self.typical_price_history = np.append(self.typical_price_history, typical_price)

        # 기본 data_history는 typical price로 설정
        if self.data_history.size == 0:
            self.data_history = np.array([typical_price])
        else:
            self.data_history = np.append(self.data_history, typical_price)

        # 메모리 관리
        self._manage_memory()

        # CCI 계산
        cci_value = self._calculate_current_cci()

        if cci_value is not None:
            if self.indicator_history.size == 0:
                self.indicator_history = np.array([cci_value])
            else:
                self.indicator_history = np.append(self.indicator_history, cci_value)

            if not self.is_warmed_up and len(self.typical_price_history) >= self.period:
                self.is_warmed_up = True
                logger.info(f"CCI({self.period}) 워밍업 완료, 첫 값: {cci_value:.2f}")

        return cci_value

    def _calculate_current_cci(self) -> Optional[float]:
        """현재 CCI 값 계산"""
        if len(self.typical_price_history) < self.period:
            return None

        # 최근 period 동안의 Typical Price
        recent_tp = self.typical_price_history[-self.period:]
        current_tp = self.typical_price_history[-1]

        # SMA(Typical Price) 계산
        sma_tp = np.mean(recent_tp)

        # Mean Deviation 계산
        deviations = np.abs(recent_tp - sma_tp)
        mean_deviation = np.mean(deviations)

        # CCI 계산
        if mean_deviation == 0:
            return 0.0  # 편차가 없으면 0 반환

        cci = (current_tp - sma_tp) / (self.constant * mean_deviation)
        return cci

    def _calculate_indicator(self) -> np.ndarray:
        """
        전체 CCI 지표 계산

        Returns:
            np.ndarray: 계산된 CCI 값들
        """
        if len(self.typical_price_history) < self.period:
            return np.array([])

        cci_values = []

        for i in range(self.period - 1, len(self.typical_price_history)):
            # 해당 기간의 Typical Price
            period_tp = self.typical_price_history[i - self.period + 1:i + 1]
            current_tp = self.typical_price_history[i]

            # SMA와 Mean Deviation 계산
            sma_tp = np.mean(period_tp)
            deviations = np.abs(period_tp - sma_tp)
            mean_deviation = np.mean(deviations)

            # CCI 계산
            if mean_deviation == 0:
                cci = 0.0
            else:
                cci = (current_tp - sma_tp) / (self.constant * mean_deviation)

            cci_values.append(cci)

        self.indicator_history = np.array(cci_values)
        return self.indicator_history

    def _manage_memory(self):
        """메모리 관리 - HLC 및 Typical Price 히스토리 포함"""
        super()._manage_memory()

        if len(self.high_history) > self.max_history:
            excess = len(self.high_history) - self.max_history
            self.high_history = self.high_history[excess:]
            self.low_history = self.low_history[excess:]
            self.close_history = self.close_history[excess:]
            self.typical_price_history = self.typical_price_history[excess:]

    def get_signal(self) -> Dict[str, Any]:
        """
        현재 CCI 기반 매매 신호 반환

        Returns:
            Dict: 매매 신호 정보
        """
        current_cci = self.get_current_value()
        if current_cci is None:
            return {
                'signal': 'neutral',
                'strength': 0,
                'cci_value': None,
                'description': 'CCI 계산 불가'
            }

        # 신호 분류
        if current_cci >= self.overbought_threshold:
            if current_cci >= 200:
                signal = 'strong_sell'
                strength = min(10, int((current_cci - 200) / 50) + 8)
                description = f'극도 과매수 (CCI: {current_cci:.1f})'
            else:
                signal = 'sell'
                strength = min(8, int((current_cci - self.overbought_threshold) / 25) + 6)
                description = f'과매수 (CCI: {current_cci:.1f})'
        elif current_cci <= self.oversold_threshold:
            if current_cci <= -200:
                signal = 'strong_buy'
                strength = min(10, int((-200 - current_cci) / 50) + 8)
                description = f'극도 과매도 (CCI: {current_cci:.1f})'
            else:
                signal = 'buy'
                strength = min(8, int((self.oversold_threshold - current_cci) / 25) + 6)
                description = f'과매도 (CCI: {current_cci:.1f})'
        else:
            signal = 'neutral'
            strength = 5
            description = f'중립 (CCI: {current_cci:.1f})'

        return {
            'signal': signal,
            'strength': strength,
            'cci_value': current_cci,
            'description': description,
            'overbought_threshold': self.overbought_threshold,
            'oversold_threshold': self.oversold_threshold
        }

    def reset(self):
        """CCI 상태 초기화"""
        super().reset()
        self.high_history = np.array([])
        self.low_history = np.array([])
        self.close_history = np.array([])
        self.typical_price_history = np.array([])


class ROCIndicator(TechnicalIndicatorEngine):
    """
    ROC(Rate of Change) 지표 계산 클래스

    ROC는 현재 가격과 n 기간 전 가격 간의 퍼센트 변화율을 나타내는 모멘텀 오실레이터입니다.

    계산 공식:
    ROC = ((Close - Close[n periods ago]) / Close[n periods ago]) × 100

    특징:
    - 무제한 범위 (퍼센트 변화율)
    - 양수: 상승 모멘텀
    - 음수: 하락 모멘텀
    - 0: 변화 없음
    - 발산 분석 가능
    """

    def __init__(self, period: int = 12, max_history: int = 10000):
        """
        ROC 지표 계산기 초기화

        Args:
            period (int): 계산 기간 (기본값: 12)
            max_history (int): 최대 보관할 데이터 개수

        Example:
            >>> roc = ROCIndicator(period=12)
            >>> roc.add_data(100)
        """
        super().__init__(period, max_history)

        logger.info(f"ROC 지표 생성: {period}일 기간")

    def _calculate_indicator(self) -> np.ndarray:
        """
        ROC 지표 계산

        Returns:
            np.ndarray: 계산된 ROC 값들
        """
        if len(self.data_history) < self.period + 1:
            return np.array([])

        roc_values = []

        for i in range(self.period, len(self.data_history)):
            current_price = self.data_history[i]
            previous_price = self.data_history[i - self.period]

            # ROC 계산
            if previous_price != 0:
                roc = ((current_price - previous_price) / previous_price) * 100.0
            else:
                roc = 0.0  # 이전 가격이 0이면 변화율 0

            roc_values.append(roc)

        self.indicator_history = np.array(roc_values)
        return self.indicator_history

    def get_signal(self) -> Dict[str, Any]:
        """
        현재 ROC 기반 매매 신호 반환

        Returns:
            Dict: 매매 신호 정보
        """
        current_roc = self.get_current_value()
        if current_roc is None:
            return {
                'signal': 'neutral',
                'strength': 0,
                'roc_value': None,
                'description': 'ROC 계산 불가'
            }

        # 신호 분류
        if current_roc > 10:
            signal = 'strong_buy'
            strength = min(10, int(current_roc / 2) + 6)
            description = f'강한 상승 모멘텀 (ROC: {current_roc:.2f}%)'
        elif current_roc > 5:
            signal = 'buy'
            strength = min(8, int(current_roc) + 4)
            description = f'상승 모멘텀 (ROC: {current_roc:.2f}%)'
        elif current_roc > 0:
            signal = 'weak_buy'
            strength = 6
            description = f'약한 상승 모멘텀 (ROC: {current_roc:.2f}%)'
        elif current_roc < -10:
            signal = 'strong_sell'
            strength = min(10, int(-current_roc / 2) + 6)
            description = f'강한 하락 모멘텀 (ROC: {current_roc:.2f}%)'
        elif current_roc < -5:
            signal = 'sell'
            strength = min(8, int(-current_roc) + 4)
            description = f'하락 모멘텀 (ROC: {current_roc:.2f}%)'
        elif current_roc < 0:
            signal = 'weak_sell'
            strength = 6
            description = f'약한 하락 모멘텀 (ROC: {current_roc:.2f}%)'
        else:
            signal = 'neutral'
            strength = 5
            description = f'모멘텀 없음 (ROC: {current_roc:.2f}%)'

        return {
            'signal': signal,
            'strength': strength,
            'roc_value': current_roc,
            'description': description,
            'momentum_direction': 'up' if current_roc > 0 else 'down' if current_roc < 0 else 'flat'
        }


class ATRIndicator(TechnicalIndicatorEngine):
    """
    ATR(Average True Range) 지표 계산 클래스

    ATR은 가격의 변동성을 측정하는 지표로, True Range의 Wilder's smoothing 평균입니다.
    Task 5의 ATRCalculator보다 효율적이고 통합적인 구현을 제공합니다.

    계산 공식:
    1. True Range = Max(High-Low, |High-Close[prev]|, |Low-Close[prev]|)
    2. ATR = Wilder's smoothing of True Range
    3. Wilder's smoothing: ATR[t] = ((n-1) × ATR[t-1] + TR[t]) / n

    특징:
    - 항상 양수 값
    - 높을수록 변동성 큼
    - 낮을수록 변동성 작음
    - 절대적 변동성 측정 (퍼센트 아님)
    """

    def __init__(self, period: int = 14, max_history: int = 10000):
        """
        ATR 지표 계산기 초기화

        Args:
            period (int): 계산 기간 (기본값: 14)
            max_history (int): 최대 보관할 데이터 개수

        Example:
            >>> atr = ATRIndicator(period=14)
            >>> atr.add_hlc_data(high=102, low=98, close=100)
        """
        super().__init__(period, max_history)

        # HLC 데이터 저장
        self.high_history = np.array([])
        self.low_history = np.array([])
        self.close_history = np.array([])
        self.true_range_history = np.array([])

        # Wilder's smoothing을 위한 ATR 값 저장
        self.current_atr = None
        self.wilder_alpha = 1.0 / period  # Wilder's smoothing factor

        logger.info(f"ATR 지표 생성: {period}일 기간")

    def add_hlc_data(self, high: float, low: float, close: float) -> Optional[float]:
        """
        HLC(High-Low-Close) 데이터를 추가하고 ATR 계산

        Args:
            high (float): 고가
            low (float): 저가
            close (float): 종가

        Returns:
            Optional[float]: 계산된 ATR 값, 워밍업 중이면 None
        """
        # 데이터 검증
        validated_high = self._validate_input(high)[0]
        validated_low = self._validate_input(low)[0]
        validated_close = self._validate_input(close)[0]

        # 가격 논리 검증
        if validated_high < validated_low:
            raise ValueError(f"고가({validated_high})가 저가({validated_low})보다 낮습니다")
        if not (validated_low <= validated_close <= validated_high):
            logger.warning(f"종가({validated_close})가 고가-저가 범위를 벗어남")

        # HLC 히스토리에 추가
        self.high_history = np.append(self.high_history, validated_high)
        self.low_history = np.append(self.low_history, validated_low)
        self.close_history = np.append(self.close_history, validated_close)

        # True Range 계산
        true_range = self._calculate_true_range(validated_high, validated_low, validated_close)
        self.true_range_history = np.append(self.true_range_history, true_range)

        # 기본 data_history는 close 가격으로 설정
        if self.data_history.size == 0:
            self.data_history = np.array([validated_close])
        else:
            self.data_history = np.append(self.data_history, validated_close)

        # 메모리 관리
        self._manage_memory()

        # ATR 계산
        atr_value = self._calculate_current_atr()

        if atr_value is not None:
            if self.indicator_history.size == 0:
                self.indicator_history = np.array([atr_value])
            else:
                self.indicator_history = np.append(self.indicator_history, atr_value)

            if not self.is_warmed_up and len(self.true_range_history) >= self.period:
                self.is_warmed_up = True
                logger.info(f"ATR({self.period}) 워밍업 완료, 첫 값: {atr_value:.6f}")

        return atr_value

    def _calculate_true_range(self, high: float, low: float, close: float) -> float:
        """True Range 계산"""
        if len(self.close_history) == 1:
            # 첫 번째 데이터인 경우 High - Low 사용
            return high - low

        # 이전 종가
        prev_close = self.close_history[-2]

        # True Range = Max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        return max(tr1, tr2, tr3)

    def _calculate_current_atr(self) -> Optional[float]:
        """현재 ATR 값 계산 (Wilder's smoothing)"""
        if len(self.true_range_history) < self.period:
            return None

        if self.current_atr is None:
            # 첫 번째 ATR: True Range의 단순 평균
            self.current_atr = np.mean(self.true_range_history[-self.period:])
        else:
            # Wilder's smoothing: ATR = ((n-1) × ATR_prev + TR) / n
            current_tr = self.true_range_history[-1]
            self.current_atr = ((self.period - 1) * self.current_atr + current_tr) / self.period

        return self.current_atr

    def _calculate_indicator(self) -> np.ndarray:
        """
        전체 ATR 지표 계산

        Returns:
            np.ndarray: 계산된 ATR 값들
        """
        if len(self.true_range_history) < self.period:
            return np.array([])

        atr_values = []
        current_atr = None

        for i in range(self.period - 1, len(self.true_range_history)):
            if current_atr is None:
                # 첫 번째 ATR: True Range의 단순 평균
                current_atr = np.mean(self.true_range_history[i - self.period + 1:i + 1])
            else:
                # Wilder's smoothing
                current_tr = self.true_range_history[i]
                current_atr = ((self.period - 1) * current_atr + current_tr) / self.period

            atr_values.append(current_atr)

        self.indicator_history = np.array(atr_values)
        return self.indicator_history

    def _manage_memory(self):
        """메모리 관리 - HLC 및 True Range 히스토리 포함"""
        super()._manage_memory()

        if len(self.high_history) > self.max_history:
            excess = len(self.high_history) - self.max_history
            self.high_history = self.high_history[excess:]
            self.low_history = self.low_history[excess:]
            self.close_history = self.close_history[excess:]
            self.true_range_history = self.true_range_history[excess:]

    def get_volatility_level(self, current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        변동성 수준 분석

        Args:
            current_price (Optional[float]): 현재 가격 (ATR 퍼센티지 계산용)

        Returns:
            Dict: 변동성 분석 결과
        """
        current_atr = self.get_current_value()
        if current_atr is None:
            return {
                'atr_value': None,
                'volatility_level': 'unknown',
                'description': 'ATR 계산 불가'
            }

        # ATR 히스토리를 이용한 상대적 변동성 평가
        if len(self.indicator_history) < 5:
            volatility_level = 'insufficient_data'
            description = 'ATR 계산 기간 부족'
        else:
            # 최근 ATR과 과거 평균 비교
            recent_atr_avg = np.mean(self.indicator_history[-5:])
            historical_atr_avg = np.mean(self.indicator_history[:-5]) if len(self.indicator_history) > 5 else recent_atr_avg

            ratio = recent_atr_avg / historical_atr_avg if historical_atr_avg > 0 else 1.0

            if ratio >= 1.5:
                volatility_level = 'very_high'
                description = f'매우 높은 변동성 (ATR: {current_atr:.6f})'
            elif ratio >= 1.2:
                volatility_level = 'high'
                description = f'높은 변동성 (ATR: {current_atr:.6f})'
            elif ratio >= 0.8:
                volatility_level = 'normal'
                description = f'보통 변동성 (ATR: {current_atr:.6f})'
            elif ratio >= 0.6:
                volatility_level = 'low'
                description = f'낮은 변동성 (ATR: {current_atr:.6f})'
            else:
                volatility_level = 'very_low'
                description = f'매우 낮은 변동성 (ATR: {current_atr:.6f})'

        result = {
            'atr_value': current_atr,
            'volatility_level': volatility_level,
            'description': description
        }

        # 현재 가격이 제공되면 ATR 퍼센티지 계산
        if current_price is not None and current_price > 0:
            atr_percentage = (current_atr / current_price) * 100
            result['atr_percentage'] = atr_percentage
            result['description'] += f' ({atr_percentage:.2f}%)'

        return result

    def reset(self):
        """ATR 상태 초기화"""
        super().reset()
        self.high_history = np.array([])
        self.low_history = np.array([])
        self.close_history = np.array([])
        self.true_range_history = np.array([])
        self.current_atr = None


class AuxiliaryIndicatorFactory:
    """
    보조 지표 객체를 생성하는 팩토리 클래스

    다양한 설정으로 보조 지표 객체를 쉽게 생성할 수 있습니다.
    """

    @staticmethod
    def create_williams_r(period: int = 14, max_history: int = 10000,
                         overbought_threshold: float = -20.0,
                         oversold_threshold: float = -80.0) -> WilliamsRIndicator:
        """
        Williams %R 지표 객체 생성

        Args:
            period (int): 계산 기간
            max_history (int): 최대 히스토리
            overbought_threshold (float): 과매수 임계값
            oversold_threshold (float): 과매도 임계값

        Returns:
            WilliamsRIndicator: 생성된 Williams %R 객체
        """
        return WilliamsRIndicator(period, max_history, overbought_threshold, oversold_threshold)

    @staticmethod
    def create_cci(period: int = 20, max_history: int = 10000,
                   overbought_threshold: float = 100.0, oversold_threshold: float = -100.0,
                   constant: float = 0.015) -> CCIIndicator:
        """
        CCI 지표 객체 생성

        Args:
            period (int): 계산 기간
            max_history (int): 최대 히스토리
            overbought_threshold (float): 과매수 임계값
            oversold_threshold (float): 과매도 임계값
            constant (float): CCI 계산 상수

        Returns:
            CCIIndicator: 생성된 CCI 객체
        """
        return CCIIndicator(period, max_history, overbought_threshold, oversold_threshold, constant)

    @staticmethod
    def create_roc(period: int = 12, max_history: int = 10000) -> ROCIndicator:
        """
        ROC 지표 객체 생성

        Args:
            period (int): 계산 기간
            max_history (int): 최대 히스토리

        Returns:
            ROCIndicator: 생성된 ROC 객체
        """
        return ROCIndicator(period, max_history)

    @staticmethod
    def create_atr(period: int = 14, max_history: int = 10000) -> ATRIndicator:
        """
        ATR 지표 객체 생성

        Args:
            period (int): 계산 기간
            max_history (int): 최대 히스토리

        Returns:
            ATRIndicator: 생성된 ATR 객체
        """
        return ATRIndicator(period, max_history)

    @staticmethod
    def create_standard_auxiliary_set(max_history: int = 10000) -> Dict[str, Union[WilliamsRIndicator, CCIIndicator, ROCIndicator, ATRIndicator]]:
        """
        표준 보조 지표 세트 생성

        Args:
            max_history (int): 최대 히스토리

        Returns:
            dict: 생성된 보조 지표 객체들의 딕셔너리

        Example:
            >>> indicators = AuxiliaryIndicatorFactory.create_standard_auxiliary_set()
            >>> williams_r = indicators['williams_r_14']
            >>> cci = indicators['cci_20']
        """
        return {
            'williams_r_14': WilliamsRIndicator(14, max_history),
            'williams_r_21': WilliamsRIndicator(21, max_history),
            'cci_20': CCIIndicator(20, max_history),
            'cci_14': CCIIndicator(14, max_history),
            'roc_12': ROCIndicator(12, max_history),
            'roc_9': ROCIndicator(9, max_history),
            'roc_21': ROCIndicator(21, max_history),
            'atr_14': ATRIndicator(14, max_history),
            'atr_21': ATRIndicator(21, max_history)
        }

    @staticmethod
    def create_custom_set(config: Dict[str, Dict[str, Any]],
                         max_history: int = 10000) -> Dict[str, Union[WilliamsRIndicator, CCIIndicator, ROCIndicator, ATRIndicator]]:
        """
        사용자 정의 보조 지표 세트 생성

        Args:
            config (Dict): 지표 설정 딕셔너리
            max_history (int): 최대 히스토리

        Returns:
            dict: 생성된 보조 지표 객체들의 딕셔너리

        Example:
            >>> config = {
            ...     'my_williams_r': {'type': 'williams_r', 'period': 10, 'overbought': -15},
            ...     'my_cci': {'type': 'cci', 'period': 25, 'constant': 0.020}
            ... }
            >>> indicators = AuxiliaryIndicatorFactory.create_custom_set(config)
        """
        indicators = {}

        for name, params in config.items():
            indicator_type = params.get('type')
            period = params.get('period', 14)

            try:
                if indicator_type == 'williams_r':
                    overbought = params.get('overbought', -20.0)
                    oversold = params.get('oversold', -80.0)
                    indicators[name] = WilliamsRIndicator(period, max_history, overbought, oversold)

                elif indicator_type == 'cci':
                    overbought = params.get('overbought', 100.0)
                    oversold = params.get('oversold', -100.0)
                    constant = params.get('constant', 0.015)
                    indicators[name] = CCIIndicator(period, max_history, overbought, oversold, constant)

                elif indicator_type == 'roc':
                    indicators[name] = ROCIndicator(period, max_history)

                elif indicator_type == 'atr':
                    indicators[name] = ATRIndicator(period, max_history)

                else:
                    logger.warning(f"알 수 없는 지표 타입: {indicator_type} ({name})")

            except Exception as e:
                logger.error(f"지표 생성 실패 {name}: {str(e)}")

        return indicators


def compare_with_task5_atr(hlc_data: List[Tuple[float, float, float]], period: int = 14) -> Dict[str, Any]:
    """
    Task 6.4의 ATR 구현과 Task 5의 ATRCalculator 결과를 비교하는 함수

    Args:
        hlc_data: (high, low, close) 튜플의 리스트
        period: ATR 계산 기간

    Returns:
        Dict: 비교 결과
    """
    try:
        # Task 6.4 ATR 계산
        atr_indicator = ATRIndicator(period)

        # HLC 데이터 추가하여 ATR 계산
        for high, low, close in hlc_data:
            atr_indicator.add_hlc_data(high, low, close)

        task6_atr_value = atr_indicator.get_current_value()

        # Task 5의 ATR는 TA-Lib 기반이므로 직접 비교
        import talib
        high_prices = np.array([h for h, l, c in hlc_data])
        low_prices = np.array([l for h, l, c in hlc_data])
        close_prices = np.array([c for h, l, c in hlc_data])

        task5_atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
        task5_atr_value = task5_atr_values[-1] if len(task5_atr_values) > 0 and not np.isnan(task5_atr_values[-1]) else None

        # 비교 결과
        comparison_result = {
            'task6_atr_value': task6_atr_value,
            'task5_atr_value': task5_atr_value,
            'values_match': False,
            'difference': None,
            'data_points': len(hlc_data),
            'comparison_note': '비교 결과'
        }

        if task6_atr_value is not None and task5_atr_value is not None:
            difference = abs(task6_atr_value - task5_atr_value)
            comparison_result['difference'] = difference
            comparison_result['values_match'] = difference < 0.000001  # 매우 작은 차이는 일치로 간주
            comparison_result['comparison_note'] = f'차이: {difference:.8f} ({"일치" if difference < 0.000001 else "불일치"})'

        logger.info(f"ATR 비교 완료: Task6={task6_atr_value}, Task5={task5_atr_value}")
        return comparison_result

    except Exception as e:
        logger.error(f"ATR 비교 중 오류: {str(e)}")
        return {
            'error': str(e),
            'comparison_note': '비교 실패'
        }