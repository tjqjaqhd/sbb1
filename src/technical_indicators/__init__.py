"""
기술적 지표 계산 엔진 패키지

이 패키지는 주식 거래를 위한 다양한 기술적 지표를 계산하는 모듈들을 포함합니다.
실시간 데이터 처리와 효율적인 메모리 사용에 최적화되어 있습니다.
"""

from .base_engine import TechnicalIndicatorEngine
from .moving_averages import SimpleMovingAverage, ExponentialMovingAverage, MovingAverageFactory
from .oscillators import RSIIndicator, MACDIndicator, OscillatorFactory, compare_with_task5_rsi
from .volatility import BollingerBands, StochasticOscillator, VolatilityIndicatorFactory
from .auxiliary import (
    WilliamsRIndicator, CCIIndicator, ROCIndicator, ATRIndicator,
    AuxiliaryIndicatorFactory, compare_with_task5_atr
)
from .ring_buffer import (
    RingBuffer, SlidingWindow, CachedIndicatorEngine,
    OptimizedTechnicalIndicatorEngine, sma_function, ema_function
)

__all__ = [
    'TechnicalIndicatorEngine',
    'SimpleMovingAverage',
    'ExponentialMovingAverage',
    'MovingAverageFactory',
    'RSIIndicator',
    'MACDIndicator',
    'OscillatorFactory',
    'compare_with_task5_rsi',
    'BollingerBands',
    'StochasticOscillator',
    'VolatilityIndicatorFactory',
    'WilliamsRIndicator',
    'CCIIndicator',
    'ROCIndicator',
    'ATRIndicator',
    'AuxiliaryIndicatorFactory',
    'compare_with_task5_atr',
    'RingBuffer',
    'SlidingWindow',
    'CachedIndicatorEngine',
    'OptimizedTechnicalIndicatorEngine',
    'sma_function',
    'ema_function'
]

__version__ = '1.0.0'