"""
변동성 기반 기술적 지표 구현

볼린저 밴드와 스토캐스틱 오실레이터를 계산하는 클래스들을 제공합니다.
실시간 데이터 처리와 효율적인 증분 계산에 최적화되어 있습니다.
"""

import numpy as np
import logging
from typing import Union, List, Optional, Dict, Tuple, Any

from .base_engine import TechnicalIndicatorEngine
from .moving_averages import SimpleMovingAverage

logger = logging.getLogger(__name__)


class BollingerBands(TechnicalIndicatorEngine):
    """
    볼린저 밴드 (Bollinger Bands) 계산 클래스

    볼린저 밴드는 가격의 변동성을 측정하고 과매수/과매도 구간을 판단하는 지표입니다.

    공식:
    - Middle Band (SMA) = 단순이동평균(20일)
    - Upper Band = SMA + (2 × Standard Deviation)
    - Lower Band = SMA - (2 × Standard Deviation)
    - Band Width = (Upper Band - Lower Band) / Middle Band × 100
    - %B = (Close - Lower Band) / (Upper Band - Lower Band) × 100

    특징:
    - 변동성이 클 때 밴드가 확장
    - 변동성이 작을 때 밴드가 수축 (스퀴즈)
    - 상단/하단 밴드 터치 시 반전 신호
    """

    def __init__(self, period: int = 20, stddev_multiplier: float = 2.0, max_history: int = 10000):
        """
        볼린저 밴드 계산기 초기화

        Args:
            period (int): 이동평균 계산 기간 (기본값: 20일)
            stddev_multiplier (float): 표준편차 배수 (기본값: 2.0)
            max_history (int): 최대 보관할 데이터 개수

        Example:
            >>> bb = BollingerBands(period=20, stddev_multiplier=2.0)
            >>> bb.add_data([100, 101, 102, 103, 104])
        """
        super().__init__(period, max_history)
        self.stddev_multiplier = stddev_multiplier

        # SMA 계산을 위한 하위 지표
        self.sma = SimpleMovingAverage(period, max_history)

        # 결과 저장을 위한 배열들
        self.upper_band_history = np.array([])
        self.middle_band_history = np.array([])  # SMA와 동일
        self.lower_band_history = np.array([])
        self.band_width_history = np.array([])
        self.percent_b_history = np.array([])

        logger.info(f"볼린저 밴드 계산기 생성: {period}일 기간, 표준편차 배수={stddev_multiplier}")

    def _calculate_indicator(self) -> np.ndarray:
        """
        볼린저 밴드 계산

        Returns:
            np.ndarray: [upper_band, middle_band, lower_band, band_width, percent_b] 형태의 최신 값
        """
        if len(self.data_history) < self.period:
            return np.array([])

        try:
            # SMA 계산 (Middle Band)
            self.sma.data_history = self.data_history.copy()
            sma_values = self.sma._calculate_indicator()

            if len(sma_values) == 0:
                return np.array([])

            # 각 SMA 지점에서 표준편차 계산
            upper_bands = []
            lower_bands = []
            band_widths = []
            percent_bs = []

            data_len = len(self.data_history)

            for i in range(self.period - 1, data_len):
                # 현재 윈도우 데이터
                window_data = self.data_history[i - self.period + 1:i + 1]
                current_price = self.data_history[i]

                # SMA와 표준편차 계산
                sma_value = np.mean(window_data)
                std_value = np.std(window_data, ddof=0)  # 모집단 표준편차

                # 볼린저 밴드 계산
                upper_band = sma_value + (self.stddev_multiplier * std_value)
                lower_band = sma_value - (self.stddev_multiplier * std_value)

                # Band Width 계산 (백분율)
                if sma_value != 0:
                    band_width = ((upper_band - lower_band) / sma_value) * 100
                else:
                    band_width = 0.0

                # %B 계산 (밴드 내 위치, 0-100%)
                if upper_band != lower_band:
                    percent_b = ((current_price - lower_band) / (upper_band - lower_band)) * 100
                else:
                    percent_b = 50.0  # 밴드가 같으면 중앙으로 설정

                upper_bands.append(upper_band)
                lower_bands.append(lower_band)
                band_widths.append(band_width)
                percent_bs.append(percent_b)

            # 히스토리 업데이트
            self.upper_band_history = np.array(upper_bands)
            self.middle_band_history = sma_values
            self.lower_band_history = np.array(lower_bands)
            self.band_width_history = np.array(band_widths)
            self.percent_b_history = np.array(percent_bs)

            # indicator_history는 최신 모든 값들의 조합으로 설정
            # [upper, middle, lower, band_width, percent_b]의 형태로 마지막 값들 반환
            if len(upper_bands) > 0:
                latest_values = np.array([
                    upper_bands[-1],
                    sma_values[-1],
                    lower_bands[-1],
                    band_widths[-1],
                    percent_bs[-1]
                ])
                # 1차원 배열로 저장 (기본 클래스와 호환성 위해)
                self.indicator_history = latest_values
                return latest_values
            else:
                return np.array([])

        except Exception as e:
            logger.error(f"볼린저 밴드 계산 중 오류: {e}")
            return np.array([])

    def get_current_bands(self) -> Optional[Dict[str, float]]:
        """
        현재 볼린저 밴드 값들 반환

        Returns:
            Dict: 현재 볼린저 밴드 정보 또는 None

        Example:
            >>> bands = bb.get_current_bands()
            >>> print(f"Upper: {bands['upper']}, Middle: {bands['middle']}")
        """
        if not self.is_warmed_up or len(self.upper_band_history) == 0:
            return None

        return {
            'upper_band': float(self.upper_band_history[-1]),
            'middle_band': float(self.middle_band_history[-1]),
            'lower_band': float(self.lower_band_history[-1]),
            'band_width': float(self.band_width_history[-1]),
            'percent_b': float(self.percent_b_history[-1]),
            'current_price': float(self.data_history[-1])
        }

    def get_band_signals(self) -> Dict[str, Any]:
        """
        볼린저 밴드 기반 매매 신호 생성

        Returns:
            Dict: 매매 신호 정보

        신호 종류:
        - oversold: %B < 0 (하단 밴드 이하)
        - overbought: %B > 100 (상단 밴드 이상)
        - squeeze: Band Width < 평균 대비 낮음
        - expansion: Band Width > 평균 대비 높음
        """
        if not self.is_warmed_up:
            return {'signal': 'insufficient_data', 'strength': 0}

        try:
            current_bands = self.get_current_bands()
            if not current_bands:
                return {'signal': 'calculation_error', 'strength': 0}

            percent_b = current_bands['percent_b']
            band_width = current_bands['band_width']

            # 최근 밴드 폭 평균 계산 (스퀴즈 판정용)
            recent_periods = min(20, len(self.band_width_history))
            if recent_periods > 5:
                avg_band_width = np.mean(self.band_width_history[-recent_periods:])
                bw_ratio = band_width / avg_band_width if avg_band_width > 0 else 1.0
            else:
                bw_ratio = 1.0

            signals = []
            signal_strength = 0

            # %B 기반 신호
            if percent_b <= 0:
                signals.append('oversold')
                signal_strength += 8
            elif percent_b >= 100:
                signals.append('overbought')
                signal_strength += 8
            elif percent_b <= 20:
                signals.append('near_oversold')
                signal_strength += 5
            elif percent_b >= 80:
                signals.append('near_overbought')
                signal_strength += 5

            # 밴드 폭 기반 신호
            if bw_ratio <= 0.5:
                signals.append('extreme_squeeze')
                signal_strength += 7
            elif bw_ratio <= 0.7:
                signals.append('squeeze')
                signal_strength += 5
            elif bw_ratio >= 1.5:
                signals.append('expansion')
                signal_strength += 3

            # 주요 신호 결정
            if 'extreme_squeeze' in signals:
                primary_signal = 'breakout_imminent'
            elif 'squeeze' in signals:
                primary_signal = 'breakout_setup'
            elif 'oversold' in signals:
                primary_signal = 'buy'
            elif 'overbought' in signals:
                primary_signal = 'sell'
            elif 'expansion' in signals:
                primary_signal = 'trend_continuation'
            else:
                primary_signal = 'neutral'

            return {
                'primary_signal': primary_signal,
                'signals': signals,
                'signal_strength': min(10, signal_strength),
                'percent_b': percent_b,
                'band_width': band_width,
                'bw_ratio': bw_ratio,
                'current_bands': current_bands
            }

        except Exception as e:
            logger.error(f"볼린저 밴드 신호 생성 중 오류: {e}")
            return {'signal': 'error', 'strength': 0}

    def get_band_history(self, count: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        볼린저 밴드 히스토리 반환

        Args:
            count: 반환할 최근 데이터 개수 (None이면 전체)

        Returns:
            Dict: 각 밴드별 히스토리 배열
        """
        if count is None:
            return {
                'upper_band': self.upper_band_history.copy(),
                'middle_band': self.middle_band_history.copy(),
                'lower_band': self.lower_band_history.copy(),
                'band_width': self.band_width_history.copy(),
                'percent_b': self.percent_b_history.copy()
            }
        else:
            return {
                'upper_band': self.upper_band_history[-count:].copy() if len(self.upper_band_history) >= count else self.upper_band_history.copy(),
                'middle_band': self.middle_band_history[-count:].copy() if len(self.middle_band_history) >= count else self.middle_band_history.copy(),
                'lower_band': self.lower_band_history[-count:].copy() if len(self.lower_band_history) >= count else self.lower_band_history.copy(),
                'band_width': self.band_width_history[-count:].copy() if len(self.band_width_history) >= count else self.band_width_history.copy(),
                'percent_b': self.percent_b_history[-count:].copy() if len(self.percent_b_history) >= count else self.percent_b_history.copy()
            }


class StochasticOscillator(TechnicalIndicatorEngine):
    """
    스토캐스틱 오실레이터 (Stochastic Oscillator) 계산 클래스

    스토캐스틱은 현재 가격이 일정 기간의 최고가-최저가 범위에서 어느 위치에 있는지를 나타내는 지표입니다.

    공식:
    - Fast %K = ((Close - Lowest Low) / (Highest High - Lowest Low)) × 100
    - Slow %K = SMA(Fast %K, smoothing_k)
    - %D = SMA(Slow %K, smoothing_d)

    특징:
    - 0-100 범위의 오실레이터
    - 80 이상: 과매수, 20 이하: 과매도
    - %K와 %D의 골든/데드크로스로 매매 신호 생성
    """

    def __init__(self, period: int = 14, smoothing_k: int = 3, smoothing_d: int = 3, max_history: int = 10000):
        """
        스토캐스틱 오실레이터 계산기 초기화

        Args:
            period (int): 계산 기간 (기본값: 14일)
            smoothing_k (int): Slow %K 평활화 기간 (기본값: 3일)
            smoothing_d (int): %D 평활화 기간 (기본값: 3일)
            max_history (int): 최대 보관할 데이터 개수

        Example:
            >>> stoch = StochasticOscillator(period=14, smoothing_k=3, smoothing_d=3)
            >>> # OHLC 데이터 형태로 입력: [high, low, close]
            >>> stoch.add_ohlc_data([105, 95, 100])
        """
        super().__init__(period, max_history)
        self.smoothing_k = smoothing_k
        self.smoothing_d = smoothing_d

        # OHLC 데이터 저장
        self.high_history = np.array([])
        self.low_history = np.array([])
        self.close_history = np.array([])

        # 스토캐스틱 값들 저장
        self.fast_k_history = np.array([])
        self.slow_k_history = np.array([])
        self.d_history = np.array([])

        # 평활화를 위한 SMA 계산기들
        self.sma_k = SimpleMovingAverage(smoothing_k, max_history)
        self.sma_d = SimpleMovingAverage(smoothing_d, max_history)

        logger.info(f"스토캐스틱 오실레이터 생성: {period}일 기간, K 평활화={smoothing_k}, D 평활화={smoothing_d}")

    def add_ohlc_data(self, ohlc_data: Union[List[float], Tuple[float, float, float]]) -> Optional[Dict[str, float]]:
        """
        OHLC 데이터를 추가하고 스토캐스틱 계산

        Args:
            ohlc_data: [high, low, close] 형태의 데이터

        Returns:
            Dict: 계산된 스토캐스틱 값들 또는 None

        Example:
            >>> result = stoch.add_ohlc_data([105.5, 94.2, 100.1])
            >>> if result:
            ...     print(f"Fast %K: {result['fast_k']:.2f}")
        """
        try:
            if len(ohlc_data) != 3:
                raise ValueError("OHLC 데이터는 [high, low, close] 형태여야 합니다")

            high, low, close = map(float, ohlc_data)

            # 유효성 검사
            if high < low:
                raise ValueError(f"고가({high})가 저가({low})보다 낮을 수 없습니다")
            if not (low <= close <= high):
                logger.warning(f"종가({close})가 고가({high})-저가({low}) 범위를 벗어났습니다")

            # 데이터 히스토리에 추가
            if self.high_history.size == 0:
                self.high_history = np.array([high])
                self.low_history = np.array([low])
                self.close_history = np.array([close])
            else:
                self.high_history = np.append(self.high_history, high)
                self.low_history = np.append(self.low_history, low)
                self.close_history = np.append(self.close_history, close)

            # 데이터 히스토리도 업데이트 (베이스 클래스 호환성)
            self.data_history = self.close_history.copy()

            # 메모리 관리
            self._manage_memory()

            # 스토캐스틱 계산
            result = self._calculate_stochastic()

            # 워밍업 상태 확인
            if not self.is_warmed_up and len(self.close_history) >= self.period:
                self.is_warmed_up = True
                logger.info(f"스토캐스틱({self.period}) 워밍업 완료")

            return result

        except Exception as e:
            logger.error(f"OHLC 데이터 추가 중 오류: {e}")
            return None

    def _manage_memory(self):
        """메모리 관리를 위해 오버라이드"""
        super()._manage_memory()

        # OHLC 히스토리도 함께 관리
        if len(self.high_history) > self.max_history:
            excess = len(self.high_history) - self.max_history
            self.high_history = self.high_history[excess:]
            self.low_history = self.low_history[excess:]
            self.close_history = self.close_history[excess:]

            # 스토캐스틱 히스토리도 관리
            if len(self.fast_k_history) > 0:
                self.fast_k_history = self.fast_k_history[excess:] if len(self.fast_k_history) > excess else self.fast_k_history
                self.slow_k_history = self.slow_k_history[excess:] if len(self.slow_k_history) > excess else self.slow_k_history
                self.d_history = self.d_history[excess:] if len(self.d_history) > excess else self.d_history

    def _calculate_indicator(self) -> np.ndarray:
        """
        베이스 클래스 호환성을 위한 지표 계산 (내부적으로 _calculate_stochastic 호출)
        """
        return self._calculate_stochastic_values()

    def _calculate_stochastic(self) -> Optional[Dict[str, float]]:
        """
        스토캐스틱 오실레이터 계산 (단일 값 반환)

        Returns:
            Dict: 최신 스토캐스틱 값들 또는 None
        """
        if len(self.close_history) < self.period:
            return None

        try:
            # 전체 스토캐스틱 값 계산
            self._calculate_stochastic_values()

            # 최신 값들 반환
            if len(self.fast_k_history) > 0:
                result = {
                    'fast_k': float(self.fast_k_history[-1]),
                    'slow_k': float(self.slow_k_history[-1]) if len(self.slow_k_history) > 0 else None,
                    'd': float(self.d_history[-1]) if len(self.d_history) > 0 else None
                }
                return {k: v for k, v in result.items() if v is not None}

            return None

        except Exception as e:
            logger.error(f"스토캐스틱 계산 중 오류: {e}")
            return None

    def _calculate_stochastic_values(self) -> np.ndarray:
        """
        전체 스토캐스틱 값들 계산

        Returns:
            np.ndarray: [fast_k, slow_k, d] 형태의 최신 값들
        """
        if len(self.close_history) < self.period:
            return np.array([])

        try:
            fast_k_values = []
            data_len = len(self.close_history)

            # Fast %K 계산
            for i in range(self.period - 1, data_len):
                # 현재 기간 내의 최고가, 최저가 찾기
                window_high = self.high_history[i - self.period + 1:i + 1]
                window_low = self.low_history[i - self.period + 1:i + 1]
                current_close = self.close_history[i]

                highest_high = np.max(window_high)
                lowest_low = np.min(window_low)

                # Fast %K 계산
                if highest_high != lowest_low:
                    fast_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
                else:
                    fast_k = 50.0  # 범위가 0이면 중앙값

                # 0-100 범위로 클램프
                fast_k = max(0.0, min(100.0, fast_k))
                fast_k_values.append(fast_k)

            self.fast_k_history = np.array(fast_k_values)

            # Slow %K 계산 (Fast %K의 SMA)
            if len(fast_k_values) >= self.smoothing_k:
                slow_k_values = []
                for i in range(self.smoothing_k - 1, len(fast_k_values)):
                    window_fast_k = fast_k_values[i - self.smoothing_k + 1:i + 1]
                    slow_k = np.mean(window_fast_k)
                    slow_k_values.append(slow_k)

                self.slow_k_history = np.array(slow_k_values)
            else:
                self.slow_k_history = np.array([])

            # %D 계산 (Slow %K의 SMA)
            if len(self.slow_k_history) >= self.smoothing_d:
                d_values = []
                for i in range(self.smoothing_d - 1, len(self.slow_k_history)):
                    window_slow_k = self.slow_k_history[i - self.smoothing_d + 1:i + 1]
                    d_value = np.mean(window_slow_k)
                    d_values.append(d_value)

                self.d_history = np.array(d_values)
            else:
                self.d_history = np.array([])

            # indicator_history 업데이트 (최신 값들)
            if len(self.fast_k_history) > 0:
                latest_values = [self.fast_k_history[-1]]
                if len(self.slow_k_history) > 0:
                    latest_values.append(self.slow_k_history[-1])
                else:
                    latest_values.append(np.nan)
                if len(self.d_history) > 0:
                    latest_values.append(self.d_history[-1])
                else:
                    latest_values.append(np.nan)

                self.indicator_history = np.array(latest_values)
                return np.array(latest_values)

            return np.array([])

        except Exception as e:
            logger.error(f"스토캐스틱 값 계산 중 오류: {e}")
            return np.array([])

    def get_current_values(self) -> Optional[Dict[str, float]]:
        """
        현재 스토캐스틱 값들 반환

        Returns:
            Dict: 현재 스토캐스틱 정보 또는 None
        """
        if not self.is_warmed_up:
            return None

        try:
            result = {}

            if len(self.fast_k_history) > 0:
                result['fast_k'] = float(self.fast_k_history[-1])

            if len(self.slow_k_history) > 0:
                result['slow_k'] = float(self.slow_k_history[-1])

            if len(self.d_history) > 0:
                result['d'] = float(self.d_history[-1])

            return result if result else None

        except Exception as e:
            logger.error(f"현재 스토캐스틱 값 조회 중 오류: {e}")
            return None

    def get_stochastic_signals(self) -> Dict[str, Any]:
        """
        스토캐스틱 기반 매매 신호 생성

        Returns:
            Dict: 매매 신호 정보

        신호 종류:
        - overbought: %K > 80
        - oversold: %K < 20
        - golden_cross: %K가 %D를 상향 돌파
        - dead_cross: %K가 %D를 하향 돌파
        """
        current_values = self.get_current_values()
        if not current_values or 'slow_k' not in current_values or 'd' not in current_values:
            return {'signal': 'insufficient_data', 'strength': 0}

        try:
            slow_k = current_values['slow_k']
            d_value = current_values['d']

            signals = []
            signal_strength = 0

            # 과매수/과매도 신호
            if slow_k >= 80:
                signals.append('overbought')
                signal_strength += 6
                if slow_k >= 90:
                    signals.append('extreme_overbought')
                    signal_strength += 2
            elif slow_k <= 20:
                signals.append('oversold')
                signal_strength += 6
                if slow_k <= 10:
                    signals.append('extreme_oversold')
                    signal_strength += 2

            # 크로스 신호 확인 (최근 2개 값 필요)
            if len(self.slow_k_history) >= 2 and len(self.d_history) >= 2:
                prev_k = self.slow_k_history[-2]
                prev_d = self.d_history[-2]

                # 골든 크로스 (%K가 %D를 상향 돌파)
                if prev_k <= prev_d and slow_k > d_value:
                    signals.append('golden_cross')
                    signal_strength += 8
                # 데드 크로스 (%K가 %D를 하향 돌파)
                elif prev_k >= prev_d and slow_k < d_value:
                    signals.append('dead_cross')
                    signal_strength += 8

            # 주요 신호 결정
            if 'golden_cross' in signals:
                if 'oversold' in signals or 'extreme_oversold' in signals:
                    primary_signal = 'strong_buy'
                else:
                    primary_signal = 'buy'
            elif 'dead_cross' in signals:
                if 'overbought' in signals or 'extreme_overbought' in signals:
                    primary_signal = 'strong_sell'
                else:
                    primary_signal = 'sell'
            elif 'extreme_oversold' in signals:
                primary_signal = 'oversold_bounce'
            elif 'extreme_overbought' in signals:
                primary_signal = 'overbought_correction'
            else:
                primary_signal = 'neutral'

            return {
                'primary_signal': primary_signal,
                'signals': signals,
                'signal_strength': min(10, signal_strength),
                'slow_k': slow_k,
                'd': d_value,
                'k_d_diff': slow_k - d_value,
                'current_values': current_values
            }

        except Exception as e:
            logger.error(f"스토캐스틱 신호 생성 중 오류: {e}")
            return {'signal': 'error', 'strength': 0}

    def get_stochastic_history(self, count: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        스토캐스틱 히스토리 반환

        Args:
            count: 반환할 최근 데이터 개수 (None이면 전체)

        Returns:
            Dict: 각 스토캐스틱 값별 히스토리 배열
        """
        if count is None:
            return {
                'fast_k': self.fast_k_history.copy(),
                'slow_k': self.slow_k_history.copy(),
                'd': self.d_history.copy()
            }
        else:
            return {
                'fast_k': self.fast_k_history[-count:].copy() if len(self.fast_k_history) >= count else self.fast_k_history.copy(),
                'slow_k': self.slow_k_history[-count:].copy() if len(self.slow_k_history) >= count else self.slow_k_history.copy(),
                'd': self.d_history[-count:].copy() if len(self.d_history) >= count else self.d_history.copy()
            }


class VolatilityIndicatorFactory:
    """
    변동성 지표 객체를 생성하는 팩토리 클래스

    다양한 설정으로 볼린저 밴드와 스토캐스틱 객체를 쉽게 생성할 수 있습니다.
    """

    @staticmethod
    def create_bollinger_bands(period: int = 20, stddev_multiplier: float = 2.0, max_history: int = 10000) -> BollingerBands:
        """
        볼린저 밴드 객체 생성

        Args:
            period (int): 계산 기간
            stddev_multiplier (float): 표준편차 배수
            max_history (int): 최대 히스토리

        Returns:
            BollingerBands: 생성된 볼린저 밴드 객체
        """
        return BollingerBands(period, stddev_multiplier, max_history)

    @staticmethod
    def create_stochastic(period: int = 14, smoothing_k: int = 3, smoothing_d: int = 3, max_history: int = 10000) -> StochasticOscillator:
        """
        스토캐스틱 오실레이터 객체 생성

        Args:
            period (int): 계산 기간
            smoothing_k (int): %K 평활화 기간
            smoothing_d (int): %D 평활화 기간
            max_history (int): 최대 히스토리

        Returns:
            StochasticOscillator: 생성된 스토캐스틱 객체
        """
        return StochasticOscillator(period, smoothing_k, smoothing_d, max_history)

    @staticmethod
    def create_standard_volatility_set(max_history: int = 10000) -> Dict[str, Union[BollingerBands, StochasticOscillator]]:
        """
        표준 변동성 지표 세트 생성

        Args:
            max_history (int): 최대 히스토리

        Returns:
            Dict: 생성된 변동성 지표 객체들의 딕셔너리

        Example:
            >>> indicators = VolatilityIndicatorFactory.create_standard_volatility_set()
            >>> bb_20 = indicators['bb_20_2']
            >>> stoch_14 = indicators['stoch_14_3_3']
        """
        return {
            # 볼린저 밴드 변형들
            'bb_20_2': BollingerBands(20, 2.0, max_history),     # 표준 볼린저 밴드
            'bb_20_1.5': BollingerBands(20, 1.5, max_history),   # 좁은 볼린저 밴드
            'bb_20_2.5': BollingerBands(20, 2.5, max_history),   # 넓은 볼린저 밴드
            'bb_10_2': BollingerBands(10, 2.0, max_history),     # 단기 볼린저 밴드
            'bb_50_2': BollingerBands(50, 2.0, max_history),     # 장기 볼린저 밴드

            # 스토캐스틱 변형들
            'stoch_14_3_3': StochasticOscillator(14, 3, 3, max_history),  # 표준 스토캐스틱
            'stoch_5_3_3': StochasticOscillator(5, 3, 3, max_history),    # 단기 스토캐스틱
            'stoch_21_3_3': StochasticOscillator(21, 3, 3, max_history),  # 장기 스토캐스틱
            'stoch_14_1_1': StochasticOscillator(14, 1, 1, max_history),  # 빠른 스토캐스틱
            'stoch_14_5_5': StochasticOscillator(14, 5, 5, max_history),  # 느린 스토캐스틱
        }