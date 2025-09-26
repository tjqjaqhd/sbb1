"""
오실레이터 지표 구현

RSI(Relative Strength Index)와 MACD(Moving Average Convergence Divergence) 등
모멘텀 기반 오실레이터 지표들을 계산하는 클래스들을 제공합니다.
기존 TechnicalIndicatorEngine을 상속받아 실시간 데이터 처리와 효율적인 증분 계산에 최적화되어 있습니다.
"""

import numpy as np
import logging
from typing import Union, List, Optional, Dict, Any, Tuple

from .base_engine import TechnicalIndicatorEngine
from .moving_averages import ExponentialMovingAverage

logger = logging.getLogger(__name__)


class RSIIndicator(TechnicalIndicatorEngine):
    """
    RSI(Relative Strength Index) 지표 계산 클래스

    RSI는 상대강도지수로, 주가의 상승압력과 하락압력 간의 상대적 강도를 나타내는 모멘텀 오실레이터입니다.

    계산 공식:
    - RS = 평균 상승폭 / 평균 하락폭
    - RSI = 100 - (100 / (1 + RS))
    - Wilder's smoothing: α = 1/period

    특징:
    - 0~100 범위의 값
    - 70 이상: 과매수 구간
    - 30 이하: 과매도 구간
    - Wilder's smoothing 방법 사용
    """

    def __init__(self, period: int = 14, max_history: int = 10000,
                 overbought_threshold: float = 70.0, oversold_threshold: float = 30.0):
        """
        RSI 지표 계산기 초기화

        Args:
            period (int): RSI 계산 기간 (기본값: 14)
            max_history (int): 최대 보관할 데이터 개수
            overbought_threshold (float): 과매수 임계값 (기본값: 70)
            oversold_threshold (float): 과매도 임계값 (기본값: 30)

        Example:
            >>> rsi = RSIIndicator(period=14)
            >>> rsi.add_data([100, 101, 99, 102, 98, 103])
        """
        super().__init__(period, max_history)

        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold

        # Wilder's smoothing factor (1/period)
        self.alpha = 1.0 / period

        # 평균 상승/하락폭 저장
        self.avg_gain = None
        self.avg_loss = None

        # 이전 가격 저장 (증분 계산용)
        self.previous_price = None

        logger.info(f"RSI 지표 생성: {period}일 기간, 과매수={overbought_threshold}, 과매도={oversold_threshold}")

    def _calculate_indicator(self) -> np.ndarray:
        """
        RSI 지표 계산

        Returns:
            np.ndarray: 계산된 RSI 값들
        """
        if len(self.data_history) < self.period + 1:
            return np.array([])

        # 가격 변화 계산
        price_changes = np.diff(self.data_history)

        # 상승/하락 분리
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)

        rsi_values = []

        # 첫 번째 평균 (단순평균)
        if len(gains) >= self.period:
            avg_gain = np.mean(gains[:self.period])
            avg_loss = np.mean(losses[:self.period])

            # 첫 번째 RSI 계산
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0  # 손실이 없으면 RSI는 100

            rsi_values.append(rsi)

            # 나머지 RSI 값들 (Wilder's smoothing)
            for i in range(self.period, len(gains)):
                # Wilder's smoothing 적용
                avg_gain = self.alpha * gains[i] + (1 - self.alpha) * avg_gain
                avg_loss = self.alpha * losses[i] + (1 - self.alpha) * avg_loss

                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100.0

                rsi_values.append(rsi)

            # 상태 업데이트
            self.avg_gain = avg_gain
            self.avg_loss = avg_loss

        self.indicator_history = np.array(rsi_values)
        return self.indicator_history

    def add_data_incremental(self, new_price: float) -> Optional[float]:
        """
        단일 가격 데이터를 추가하고 증분 RSI 계산 수행

        Args:
            new_price (float): 새로운 가격 데이터

        Returns:
            Optional[float]: 계산된 새 RSI 값, 워밍업 중이면 None
        """
        validated_price = self._validate_input(new_price)[0]

        # 이전 가격이 없으면 저장만 하고 return
        if self.previous_price is None:
            self.previous_price = validated_price
            if self.data_history.size == 0:
                self.data_history = np.array([validated_price])
            else:
                self.data_history = np.append(self.data_history, validated_price)
            self._manage_memory()
            return None

        # 가격 변화 계산
        price_change = validated_price - self.previous_price
        gain = max(0, price_change)
        loss = max(0, -price_change)

        # 데이터 히스토리에 추가
        if self.data_history.size == 0:
            self.data_history = np.array([validated_price])
        else:
            self.data_history = np.append(self.data_history, validated_price)

        self._manage_memory()

        # 충분한 데이터가 없으면 워밍업 단계
        if len(self.data_history) <= self.period:
            self.previous_price = validated_price
            return None

        # 첫 번째 RSI 계산 (초기화)
        if self.avg_gain is None or self.avg_loss is None:
            # 초기 평균 계산
            price_changes = np.diff(self.data_history[-self.period-1:])
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)

            self.avg_gain = np.mean(gains)
            self.avg_loss = np.mean(losses)
        else:
            # Wilder's smoothing 적용
            self.avg_gain = self.alpha * gain + (1 - self.alpha) * self.avg_gain
            self.avg_loss = self.alpha * loss + (1 - self.alpha) * self.avg_loss

        # RSI 계산
        if self.avg_loss != 0:
            rs = self.avg_gain / self.avg_loss
            new_rsi = 100 - (100 / (1 + rs))
        else:
            new_rsi = 100.0

        # 지표 히스토리 업데이트
        if self.indicator_history.size == 0:
            self.indicator_history = np.array([new_rsi])
        else:
            self.indicator_history = np.append(self.indicator_history, new_rsi)

        self.previous_price = validated_price

        if not self.is_warmed_up:
            self.is_warmed_up = True
            logger.info(f"RSI({self.period}) 워밍업 완료, 첫 값: {new_rsi:.2f}")

        return new_rsi

    def get_signal(self) -> Dict[str, Any]:
        """
        현재 RSI 기반 매매 신호 반환

        Returns:
            Dict: 매매 신호 정보
        """
        current_rsi = self.get_current_value()
        if current_rsi is None:
            return {
                'signal': 'neutral',
                'strength': 0,
                'rsi_value': None,
                'description': 'RSI 계산 불가'
            }

        # 신호 분류
        if current_rsi >= self.overbought_threshold:
            if current_rsi >= 80:
                signal = 'strong_sell'
                strength = min(10, int((current_rsi - 80) / 2) + 8)
                description = f'극도 과매수 (RSI: {current_rsi:.1f})'
            else:
                signal = 'sell'
                strength = min(8, int((current_rsi - self.overbought_threshold) / 5) + 6)
                description = f'과매수 (RSI: {current_rsi:.1f})'
        elif current_rsi <= self.oversold_threshold:
            if current_rsi <= 20:
                signal = 'strong_buy'
                strength = min(10, int((20 - current_rsi) / 2) + 8)
                description = f'극도 과매도 (RSI: {current_rsi:.1f})'
            else:
                signal = 'buy'
                strength = min(8, int((self.oversold_threshold - current_rsi) / 5) + 6)
                description = f'과매도 (RSI: {current_rsi:.1f})'
        else:
            signal = 'neutral'
            strength = 5
            description = f'중립 (RSI: {current_rsi:.1f})'

        return {
            'signal': signal,
            'strength': strength,
            'rsi_value': current_rsi,
            'description': description,
            'overbought_threshold': self.overbought_threshold,
            'oversold_threshold': self.oversold_threshold
        }

    def reset(self):
        """RSI 상태 초기화"""
        super().reset()
        self.avg_gain = None
        self.avg_loss = None
        self.previous_price = None


class MACDIndicator(TechnicalIndicatorEngine):
    """
    MACD(Moving Average Convergence Divergence) 지표 계산 클래스

    MACD는 두 개의 지수이동평균선의 수렴과 발산을 분석하는 모멘텀 오실레이터입니다.

    계산 공식:
    - MACD Line = EMA12 - EMA26
    - Signal Line = EMA9(MACD Line)
    - Histogram = MACD Line - Signal Line

    특징:
    - 트렌드 추종과 모멘텀 분석 결합
    - 골든크로스/데드크로스 신호
    - 다이버전스 분석 가능
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                 max_history: int = 10000):
        """
        MACD 지표 계산기 초기화

        Args:
            fast_period (int): 빠른 EMA 기간 (기본값: 12)
            slow_period (int): 느린 EMA 기간 (기본값: 26)
            signal_period (int): 시그널 라인 EMA 기간 (기본값: 9)
            max_history (int): 최대 보관할 데이터 개수

        Example:
            >>> macd = MACDIndicator(fast_period=12, slow_period=26, signal_period=9)
            >>> macd.add_data([100, 101, 99, 102, 98, 103])
        """
        # 느린 EMA 기간을 전체 기간으로 설정
        super().__init__(slow_period, max_history)

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        # EMA 계산기들
        self.fast_ema = ExponentialMovingAverage(fast_period, max_history)
        self.slow_ema = ExponentialMovingAverage(slow_period, max_history)
        self.signal_ema = ExponentialMovingAverage(signal_period, max_history)

        # MACD 구성 요소들
        self.macd_line_history = np.array([])
        self.signal_line_history = np.array([])
        self.histogram_history = np.array([])

        logger.info(f"MACD 지표 생성: EMA({fast_period}, {slow_period}), Signal({signal_period})")

    def _calculate_indicator(self) -> np.ndarray:
        """
        MACD 지표 계산

        Returns:
            np.ndarray: MACD Line 값들 (기본 반환값)
        """
        if len(self.data_history) < self.slow_period:
            return np.array([])

        # EMA들 계산
        self.fast_ema.data_history = self.data_history.copy()
        self.slow_ema.data_history = self.data_history.copy()

        fast_ema_values = self.fast_ema._calculate_indicator()
        slow_ema_values = self.slow_ema._calculate_indicator()

        if len(fast_ema_values) == 0 or len(slow_ema_values) == 0:
            return np.array([])

        # MACD Line 계산 (Fast EMA - Slow EMA)
        # 배열 길이 맞춤
        min_length = min(len(fast_ema_values), len(slow_ema_values))
        fast_ema_aligned = fast_ema_values[-min_length:]
        slow_ema_aligned = slow_ema_values[-min_length:]

        macd_line = fast_ema_aligned - slow_ema_aligned
        self.macd_line_history = macd_line

        # Signal Line 계산 (MACD Line의 EMA)
        if len(macd_line) >= self.signal_period:
            self.signal_ema.data_history = macd_line.copy()
            signal_line = self.signal_ema._calculate_indicator()
            self.signal_line_history = signal_line

            # Histogram 계산 (MACD - Signal)
            if len(signal_line) > 0:
                hist_length = min(len(macd_line), len(signal_line))
                macd_aligned = macd_line[-hist_length:]
                signal_aligned = signal_line[-hist_length:]
                histogram = macd_aligned - signal_aligned
                self.histogram_history = histogram
            else:
                self.histogram_history = np.array([])
        else:
            self.signal_line_history = np.array([])
            self.histogram_history = np.array([])

        # 기본 지표 히스토리는 MACD Line으로 설정
        self.indicator_history = macd_line
        return self.indicator_history

    def add_data_incremental(self, new_price: float) -> Optional[Dict[str, float]]:
        """
        단일 가격 데이터를 추가하고 증분 MACD 계산 수행

        Args:
            new_price (float): 새로운 가격 데이터

        Returns:
            Optional[Dict]: 계산된 MACD 구성 요소들, 워밍업 중이면 None
        """
        validated_price = self._validate_input(new_price)[0]

        # 기본 클래스의 데이터 추가
        if self.data_history.size == 0:
            self.data_history = np.array([validated_price])
        else:
            self.data_history = np.append(self.data_history, validated_price)

        self._manage_memory()

        # EMA들에 데이터 추가
        fast_ema_value = self.fast_ema.add_data_incremental(validated_price)
        slow_ema_value = self.slow_ema.add_data_incremental(validated_price)

        if fast_ema_value is None or slow_ema_value is None:
            return None

        # MACD Line 계산
        macd_value = fast_ema_value - slow_ema_value

        # MACD Line 히스토리 업데이트
        if self.macd_line_history.size == 0:
            self.macd_line_history = np.array([macd_value])
        else:
            self.macd_line_history = np.append(self.macd_line_history, macd_value)

        # Signal Line 계산
        signal_value = self.signal_ema.add_data_incremental(macd_value)
        signal_line_calculated = False

        if signal_value is not None:
            if self.signal_line_history.size == 0:
                self.signal_line_history = np.array([signal_value])
            else:
                self.signal_line_history = np.append(self.signal_line_history, signal_value)
            signal_line_calculated = True

        # Histogram 계산
        histogram_value = None
        if signal_line_calculated and signal_value is not None:
            histogram_value = macd_value - signal_value

            if self.histogram_history.size == 0:
                self.histogram_history = np.array([histogram_value])
            else:
                self.histogram_history = np.append(self.histogram_history, histogram_value)

        # 기본 지표 히스토리 업데이트 (MACD Line)
        if self.indicator_history.size == 0:
            self.indicator_history = np.array([macd_value])
        else:
            self.indicator_history = np.append(self.indicator_history, macd_value)

        if not self.is_warmed_up and len(self.data_history) >= self.slow_period:
            self.is_warmed_up = True
            logger.info(f"MACD 워밍업 완료, MACD: {macd_value:.4f}")

        return {
            'macd': macd_value,
            'signal': signal_value,
            'histogram': histogram_value
        }

    def get_macd_components(self) -> Dict[str, Optional[float]]:
        """
        현재 MACD 구성 요소들 반환

        Returns:
            Dict: MACD Line, Signal Line, Histogram 현재값들
        """
        macd_value = self.macd_line_history[-1] if len(self.macd_line_history) > 0 else None
        signal_value = self.signal_line_history[-1] if len(self.signal_line_history) > 0 else None
        histogram_value = self.histogram_history[-1] if len(self.histogram_history) > 0 else None

        return {
            'macd': macd_value,
            'signal': signal_value,
            'histogram': histogram_value
        }

    def get_signal(self) -> Dict[str, Any]:
        """
        현재 MACD 기반 매매 신호 반환

        Returns:
            Dict: 매매 신호 정보
        """
        components = self.get_macd_components()
        macd_value = components['macd']
        signal_value = components['signal']
        histogram_value = components['histogram']

        if macd_value is None or signal_value is None:
            return {
                'signal': 'neutral',
                'strength': 0,
                'description': 'MACD 계산 불가',
                'components': components
            }

        # 신호 분석
        signal = 'neutral'
        strength = 5
        description = 'MACD 중립'

        # 골든크로스/데드크로스 확인
        if macd_value > signal_value:
            # MACD가 Signal 위에 있음 (상승 신호)
            if histogram_value is not None and histogram_value > 0:
                strength = min(8, int(abs(histogram_value) * 1000) + 6)
                signal = 'buy'
                description = f'골든크로스 (MACD > Signal)'
            else:
                strength = 6
                signal = 'weak_buy'
                description = f'약한 상승 신호'
        elif macd_value < signal_value:
            # MACD가 Signal 아래에 있음 (하락 신호)
            if histogram_value is not None and histogram_value < 0:
                strength = min(8, int(abs(histogram_value) * 1000) + 6)
                signal = 'sell'
                description = f'데드크로스 (MACD < Signal)'
            else:
                strength = 6
                signal = 'weak_sell'
                description = f'약한 하락 신호'

        # 히스토그램 트렌드 확인으로 신호 강도 조정
        if len(self.histogram_history) >= 3:
            recent_histogram = self.histogram_history[-3:]
            if len(recent_histogram) == 3:
                # 히스토그램이 증가 추세면 매수 신호 강화
                if recent_histogram[2] > recent_histogram[1] > recent_histogram[0]:
                    if signal in ['buy', 'weak_buy']:
                        strength = min(10, strength + 2)
                        description += ' (상승 가속)'
                # 히스토그램이 감소 추세면 매도 신호 강화
                elif recent_histogram[2] < recent_histogram[1] < recent_histogram[0]:
                    if signal in ['sell', 'weak_sell']:
                        strength = min(10, strength + 2)
                        description += ' (하락 가속)'

        return {
            'signal': signal,
            'strength': strength,
            'description': description,
            'components': components,
            'crossover_status': 'bullish' if macd_value > signal_value else 'bearish'
        }

    def get_history_components(self, count: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        MACD 구성 요소들의 히스토리 반환

        Args:
            count: 반환할 최근 데이터 개수 (None이면 전체)

        Returns:
            Dict: 각 구성 요소의 히스토리 배열들
        """
        if count is None:
            return {
                'macd': self.macd_line_history.copy(),
                'signal': self.signal_line_history.copy(),
                'histogram': self.histogram_history.copy()
            }
        else:
            return {
                'macd': self.macd_line_history[-count:].copy() if len(self.macd_line_history) >= count else self.macd_line_history.copy(),
                'signal': self.signal_line_history[-count:].copy() if len(self.signal_line_history) >= count else self.signal_line_history.copy(),
                'histogram': self.histogram_history[-count:].copy() if len(self.histogram_history) >= count else self.histogram_history.copy()
            }

    def reset(self):
        """MACD 상태 초기화"""
        super().reset()
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.signal_ema.reset()
        self.macd_line_history = np.array([])
        self.signal_line_history = np.array([])
        self.histogram_history = np.array([])


class OscillatorFactory:
    """
    오실레이터 지표 객체를 생성하는 팩토리 클래스

    다양한 설정으로 오실레이터 지표 객체를 쉽게 생성할 수 있습니다.
    """

    @staticmethod
    def create_rsi(period: int = 14, max_history: int = 10000,
                   overbought_threshold: float = 70.0, oversold_threshold: float = 30.0) -> RSIIndicator:
        """
        RSI 지표 객체 생성

        Args:
            period (int): RSI 계산 기간
            max_history (int): 최대 히스토리
            overbought_threshold (float): 과매수 임계값
            oversold_threshold (float): 과매도 임계값

        Returns:
            RSIIndicator: 생성된 RSI 객체
        """
        return RSIIndicator(period, max_history, overbought_threshold, oversold_threshold)

    @staticmethod
    def create_macd(fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                    max_history: int = 10000) -> MACDIndicator:
        """
        MACD 지표 객체 생성

        Args:
            fast_period (int): 빠른 EMA 기간
            slow_period (int): 느린 EMA 기간
            signal_period (int): 시그널 EMA 기간
            max_history (int): 최대 히스토리

        Returns:
            MACDIndicator: 생성된 MACD 객체
        """
        return MACDIndicator(fast_period, slow_period, signal_period, max_history)

    @staticmethod
    def create_standard_set(max_history: int = 10000) -> Dict[str, Union[RSIIndicator, MACDIndicator]]:
        """
        표준 오실레이터 세트 생성

        Args:
            max_history (int): 최대 히스토리

        Returns:
            dict: 생성된 오실레이터 객체들의 딕셔너리

        Example:
            >>> oscillators = OscillatorFactory.create_standard_set()
            >>> rsi_14 = oscillators['rsi_14']
            >>> macd_std = oscillators['macd_standard']
        """
        return {
            'rsi_14': RSIIndicator(14, max_history),
            'rsi_21': RSIIndicator(21, max_history),
            'rsi_9': RSIIndicator(9, max_history),
            'macd_standard': MACDIndicator(12, 26, 9, max_history),
            'macd_fast': MACDIndicator(8, 17, 9, max_history),
            'macd_slow': MACDIndicator(19, 39, 9, max_history)
        }


def compare_with_task5_rsi(price_data: List[float], period: int = 14) -> Dict[str, Any]:
    """
    Task 6.2의 RSI 구현과 Task 5의 RSI 결과를 비교하는 함수

    Args:
        price_data: 가격 데이터 리스트
        period: RSI 계산 기간

    Returns:
        Dict: 비교 결과
    """
    try:
        # Task 6.2 RSI 계산
        rsi_indicator = RSIIndicator(period)

        # 데이터 추가하여 RSI 계산
        for price in price_data:
            rsi_indicator.add_data_incremental(price)

        task6_rsi_values = rsi_indicator.get_history()
        task6_current_rsi = rsi_indicator.get_current_value()

        # Task 5의 RSI는 TA-Lib 기반이므로 직접 비교
        import talib
        task5_rsi_values = talib.RSI(np.array(price_data, dtype=np.float64), timeperiod=period)
        task5_current_rsi = task5_rsi_values[-1] if len(task5_rsi_values) > 0 and not np.isnan(task5_rsi_values[-1]) else None

        # 유효한 값들만 비교
        valid_task6 = task6_rsi_values[~np.isnan(task6_rsi_values)] if len(task6_rsi_values) > 0 else np.array([])
        valid_task5 = task5_rsi_values[~np.isnan(task5_rsi_values)] if len(task5_rsi_values) > 0 else np.array([])

        # 비교 결과
        comparison_result = {
            'task6_current_rsi': task6_current_rsi,
            'task5_current_rsi': task5_current_rsi,
            'values_match': False,
            'difference': None,
            'task6_count': len(valid_task6),
            'task5_count': len(valid_task5),
            'comparison_note': '비교 결과'
        }

        if task6_current_rsi is not None and task5_current_rsi is not None:
            difference = abs(task6_current_rsi - task5_current_rsi)
            comparison_result['difference'] = difference
            comparison_result['values_match'] = difference < 0.01  # 0.01 이하 차이면 일치로 간주
            comparison_result['comparison_note'] = f'차이: {difference:.4f} ({"일치" if difference < 0.01 else "불일치"})'

        logger.info(f"RSI 비교 완료: Task6={task6_current_rsi}, Task5={task5_current_rsi}")
        return comparison_result

    except Exception as e:
        logger.error(f"RSI 비교 중 오류: {str(e)}")
        return {
            'error': str(e),
            'comparison_note': '비교 실패'
        }