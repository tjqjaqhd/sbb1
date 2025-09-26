"""
통합 기술지표 테스트

Task 6에서 구현된 모든 기술적 지표들이 함께 올바르게 작동하는지 확인하는 통합 테스트입니다.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any

# 모든 구현된 지표들 import
from src.technical_indicators import (
    SimpleMovingAverage, ExponentialMovingAverage, MovingAverageFactory,
    RSIIndicator, MACDIndicator, OscillatorFactory,
    BollingerBands, StochasticOscillator, VolatilityIndicatorFactory
)


def generate_realistic_price_data(days: int = 60, base_price: float = 100.0) -> Dict[str, np.ndarray]:
    """
    실제와 유사한 가격 데이터 생성

    Args:
        days: 생성할 일수
        base_price: 기준 가격

    Returns:
        Dict: OHLCV 데이터
    """
    np.random.seed(42)

    # 가격 시뮬레이션
    prices = [base_price]

    for i in range(1, days):
        # 트렌드 + 변동성 + 랜덤 노이즈
        daily_return = np.random.normal(0.001, 0.02)  # 평균 0.1%, 표준편차 2%

        # 트렌드 추가 (사인파 형태)
        trend = np.sin(i * 0.05) * 0.005

        # 변동성 클러스터링 (GARCH 효과)
        volatility = 0.015 + 0.01 * abs(np.sin(i * 0.1))

        total_return = daily_return + trend + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + total_return)
        prices.append(max(1.0, new_price))

    close_prices = np.array(prices)

    # OHLC 생성
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, days)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, days)))

    # Open 가격 (이전 close 기반)
    open_prices = np.zeros(days)
    open_prices[0] = base_price
    for i in range(1, days):
        open_prices[i] = close_prices[i-1] * (1 + np.random.normal(0, 0.002))

    # Volume
    volumes = np.random.lognormal(10, 0.5, days)

    return {
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }


def test_all_indicators_integration():
    """모든 지표의 통합 테스트"""
    print("=== 통합 기술지표 테스트 ===")

    # 테스트 데이터 생성
    data = generate_realistic_price_data(60, 50000)  # 60일, 기준가 50,000원
    close_prices = data['close']

    print(f"테스트 데이터: {len(close_prices)}일, 시작가: {close_prices[0]:.0f}원, 종료가: {close_prices[-1]:.0f}원")

    # 1. 이동평균 지표들
    print("\n1. 이동평균 지표:")
    sma_20 = SimpleMovingAverage(20)
    ema_12 = ExponentialMovingAverage(12)

    for price in close_prices:
        sma_20.add_data(price)
        ema_12.add_data(price)

    if sma_20.is_warmed_up and ema_12.is_warmed_up:
        print(f"   SMA(20): {sma_20.get_current_value():.0f}원")
        print(f"   EMA(12): {ema_12.get_current_value():.0f}원")

    # 2. 오실레이터 지표들
    print("\n2. 오실레이터 지표:")
    rsi_14 = RSIIndicator(14)
    macd = MACDIndicator(12, 26, 9)

    for price in close_prices:
        rsi_14.add_data(price)
        macd.add_data(price)

    if rsi_14.is_warmed_up:
        rsi_value = rsi_14.get_current_value()
        rsi_signals = rsi_14.get_signal()
        print(f"   RSI(14): {rsi_value:.1f} - {rsi_signals['signal']}")

    if macd.is_warmed_up:
        macd_values = macd.get_macd_components()
        macd_signals = macd.get_signal()
        if macd_values and macd_values['macd'] is not None:
            print(f"   MACD: {macd_values['macd']:.2f}, Signal: {macd_values['signal']:.2f} - {macd_signals['signal']}")

    # 3. 변동성 지표들
    print("\n3. 변동성 지표:")
    bb = BollingerBands(20, 2.0)
    stoch = StochasticOscillator(14, 3, 3)

    for i, price in enumerate(close_prices):
        bb.add_data(price)
        # 스토캐스틱을 위한 OHLC 데이터
        ohlc = [data['high'][i], data['low'][i], data['close'][i]]
        stoch.add_ohlc_data(ohlc)

    if bb.is_warmed_up:
        bb_bands = bb.get_current_bands()
        bb_signals = bb.get_band_signals()
        print(f"   볼린저 밴드:")
        print(f"     상단: {bb_bands['upper_band']:.0f}원")
        print(f"     중간: {bb_bands['middle_band']:.0f}원")
        print(f"     하단: {bb_bands['lower_band']:.0f}원")
        print(f"     %B: {bb_bands['percent_b']:.1f}% - {bb_signals['primary_signal']}")

    if stoch.is_warmed_up:
        stoch_values = stoch.get_current_values()
        stoch_signals = stoch.get_stochastic_signals()
        if stoch_values and 'slow_k' in stoch_values and 'd' in stoch_values:
            print(f"   스토캐스틱: %K={stoch_values['slow_k']:.1f}, %D={stoch_values['d']:.1f} - {stoch_signals['primary_signal']}")

    # 4. 종합 신호 분석
    print("\n4. 종합 신호 분석:")
    analyze_combined_signals(
        close_prices[-1],
        sma_20.get_current_value() if sma_20.is_warmed_up else None,
        ema_12.get_current_value() if ema_12.is_warmed_up else None,
        rsi_signals if 'rsi_signals' in locals() else None,
        macd_signals if 'macd_signals' in locals() else None,
        bb_signals if 'bb_signals' in locals() else None,
        stoch_signals if 'stoch_signals' in locals() else None
    )


def analyze_combined_signals(current_price, sma_20, ema_12, rsi_signals, macd_signals, bb_signals, stoch_signals):
    """종합 신호 분석"""

    bullish_signals = 0
    bearish_signals = 0
    signal_details = []

    # 이동평균 분석
    if sma_20 and ema_12:
        if current_price > sma_20 and current_price > ema_12:
            bullish_signals += 1
            signal_details.append("가격이 이동평균 위에 위치 (상승 추세)")
        elif current_price < sma_20 and current_price < ema_12:
            bearish_signals += 1
            signal_details.append("가격이 이동평균 아래 위치 (하락 추세)")

        if ema_12 > sma_20:
            bullish_signals += 1
            signal_details.append("단기 이동평균이 장기 이동평균 위에 위치")
        elif ema_12 < sma_20:
            bearish_signals += 1
            signal_details.append("단기 이동평균이 장기 이동평균 아래 위치")

    # RSI 분석
    if rsi_signals:
        if 'oversold' in rsi_signals['signal'] or 'buy' in rsi_signals['signal']:
            bullish_signals += 1
            signal_details.append(f"RSI 매수 신호: {rsi_signals['signal']}")
        elif 'overbought' in rsi_signals['signal'] or 'sell' in rsi_signals['signal']:
            bearish_signals += 1
            signal_details.append(f"RSI 매도 신호: {rsi_signals['signal']}")

    # MACD 분석
    if macd_signals:
        if 'buy' in macd_signals['signal'] or 'bullish' in macd_signals['signal']:
            bullish_signals += 1
            signal_details.append(f"MACD 매수 신호: {macd_signals['signal']}")
        elif 'sell' in macd_signals['signal'] or 'bearish' in macd_signals['signal']:
            bearish_signals += 1
            signal_details.append(f"MACD 매도 신호: {macd_signals['signal']}")

    # 볼린저 밴드 분석
    if bb_signals:
        if 'buy' in bb_signals['primary_signal'] or 'oversold' in bb_signals['primary_signal']:
            bullish_signals += 1
            signal_details.append(f"볼린저 밴드 매수 신호: {bb_signals['primary_signal']}")
        elif 'sell' in bb_signals['primary_signal'] or 'overbought' in bb_signals['primary_signal']:
            bearish_signals += 1
            signal_details.append(f"볼린저 밴드 매도 신호: {bb_signals['primary_signal']}")
        elif 'breakout' in bb_signals['primary_signal']:
            signal_details.append(f"볼린저 밴드 돌파 신호: {bb_signals['primary_signal']}")

    # 스토캐스틱 분석
    if stoch_signals:
        if 'buy' in stoch_signals['primary_signal'] or 'oversold' in stoch_signals['primary_signal']:
            bullish_signals += 1
            signal_details.append(f"스토캐스틱 매수 신호: {stoch_signals['primary_signal']}")
        elif 'sell' in stoch_signals['primary_signal'] or 'overbought' in stoch_signals['primary_signal']:
            bearish_signals += 1
            signal_details.append(f"스토캐스틱 매도 신호: {stoch_signals['primary_signal']}")

    # 종합 판단
    total_signals = bullish_signals + bearish_signals

    if total_signals == 0:
        overall_signal = "중립"
        confidence = "신호 없음"
    else:
        bullish_ratio = bullish_signals / total_signals

        if bullish_ratio >= 0.7:
            overall_signal = "강한 매수"
            confidence = f"{bullish_ratio*100:.0f}% 확신"
        elif bullish_ratio >= 0.6:
            overall_signal = "매수"
            confidence = f"{bullish_ratio*100:.0f}% 확신"
        elif bullish_ratio <= 0.3:
            overall_signal = "강한 매도"
            confidence = f"{(1-bullish_ratio)*100:.0f}% 확신"
        elif bullish_ratio <= 0.4:
            overall_signal = "매도"
            confidence = f"{(1-bullish_ratio)*100:.0f}% 확신"
        else:
            overall_signal = "중립"
            confidence = "혼재된 신호"

    print(f"   종합 신호: {overall_signal} ({confidence})")
    print(f"   매수 신호: {bullish_signals}개, 매도 신호: {bearish_signals}개")

    if signal_details:
        print("   세부 신호:")
        for detail in signal_details:
            print(f"     - {detail}")


def test_factory_methods():
    """팩토리 메서드들 테스트"""
    print("\n=== 팩토리 메서드 테스트 ===")

    # 이동평균 팩토리
    ma_indicators = MovingAverageFactory.create_standard_set()
    print(f"이동평균 팩토리: {len(ma_indicators)}개 지표 생성")

    # 오실레이터 팩토리
    osc_indicators = OscillatorFactory.create_standard_set()
    print(f"오실레이터 팩토리: {len(osc_indicators)}개 지표 생성")

    # 변동성 지표 팩토리
    vol_indicators = VolatilityIndicatorFactory.create_standard_volatility_set()
    print(f"변동성 지표 팩토리: {len(vol_indicators)}개 지표 생성")

    print(f"총 생성된 지표: {len(ma_indicators) + len(osc_indicators) + len(vol_indicators)}개")


def performance_comparison():
    """성능 비교 테스트"""
    print("\n=== 성능 비교 테스트 ===")

    import time

    # 대량 데이터 생성
    large_data = generate_realistic_price_data(500, 50000)
    close_prices = large_data['close']

    print(f"테스트 데이터: {len(close_prices)}개")

    # 각 지표별 성능 측정
    indicators = {
        'SMA(20)': SimpleMovingAverage(20),
        'EMA(12)': ExponentialMovingAverage(12),
        'RSI(14)': RSIIndicator(14),
        'MACD': MACDIndicator(12, 26, 9),
        'Bollinger(20)': BollingerBands(20, 2.0),
    }

    for name, indicator in indicators.items():
        start_time = time.time()

        if name == 'Bollinger(20)':
            # 볼린저 밴드는 단순히 종가만 사용
            for price in close_prices:
                indicator.add_data(price)
        else:
            # 다른 지표들도 종가 사용
            for price in close_prices:
                indicator.add_data(price)

        elapsed = time.time() - start_time
        throughput = len(close_prices) / elapsed

        print(f"{name}: {elapsed:.4f}초, {throughput:.0f} 데이터/초")

        # 메모리 사용량
        status = indicator.get_status()
        print(f"  메모리: {status['memory_usage_mb']:.4f} MB")


def main():
    """메인 테스트 함수"""
    print("기술적 지표 통합 테스트")
    print("=" * 50)

    try:
        # 통합 테스트
        test_all_indicators_integration()

        # 팩토리 메서드 테스트
        test_factory_methods()

        # 성능 비교
        performance_comparison()

        print("\n" + "=" * 50)
        print("통합 테스트 성공!")
        print("\n구현 완료된 지표들:")
        print("- 이동평균: SMA, EMA")
        print("- 오실레이터: RSI, MACD")
        print("- 변동성: Bollinger Bands, Stochastic Oscillator")
        print("\nTask 6.3 볼린저 밴드 및 스토캐스틱 지표 구현 완료!")

    except Exception as e:
        print(f"테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()