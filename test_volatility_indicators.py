"""
변동성 지표 테스트 스크립트

볼린저 밴드와 스토캐스틱 오실레이터의 정확성을 검증하고
기존 BollingerAnalyzer와 결과를 비교합니다.
"""

import asyncio
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any

# 테스트 대상 모듈들
from src.technical_indicators.volatility import BollingerBands, StochasticOscillator, VolatilityIndicatorFactory
from src.services.bollinger_analyzer import BollingerAnalyzer
from src.database.config import DatabaseConfig

# TA-Lib와 비교를 위한 import
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib를 사용할 수 없습니다. 기본 테스트만 진행합니다.")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_data(length: int = 50) -> Dict[str, np.ndarray]:
    """
    테스트용 가격 데이터 생성

    Args:
        length: 생성할 데이터 개수

    Returns:
        Dict: OHLCV 데이터
    """
    np.random.seed(42)  # 재현 가능한 결과

    # 기본 가격에서 시작
    base_price = 100.0
    prices = [base_price]

    # 변동성이 있는 가격 시뮬레이션
    for i in range(1, length):
        # 트렌드 + 노이즈
        trend = np.sin(i * 0.1) * 0.02  # 주기적인 트렌드
        noise = np.random.normal(0, 0.01)  # 랜덤 노이즈
        change = trend + noise

        new_price = prices[-1] * (1 + change)
        prices.append(max(1.0, new_price))  # 최소 가격 보장

    close_prices = np.array(prices)

    # OHLC 데이터 생성
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, length)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, length)))
    open_prices = low_prices + (high_prices - low_prices) * np.random.random(length)
    volumes = np.random.uniform(1000, 5000, length)

    return {
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }


def test_bollinger_bands_basic():
    """볼린저 밴드 기본 테스트"""
    print("\n=== 볼린저 밴드 기본 테스트 ===")

    # 테스트 데이터 생성
    test_data = generate_test_data(30)
    close_prices = test_data['close']

    # 볼린저 밴드 계산기 생성
    bb = BollingerBands(period=20, stddev_multiplier=2.0)

    # 데이터 추가 및 계산
    results = []
    for price in close_prices:
        result = bb.add_data(price)
        if bb.is_warmed_up:
            current_bands = bb.get_current_bands()
            if current_bands:
                results.append(current_bands)

    print(f"총 {len(results)}개의 볼린저 밴드 값 계산됨")

    if results:
        latest = results[-1]
        print(f"최신 볼린저 밴드:")
        print(f"  상단: {latest['upper_band']:.2f}")
        print(f"  중간: {latest['middle_band']:.2f}")
        print(f"  하단: {latest['lower_band']:.2f}")
        print(f"  밴드폭: {latest['band_width']:.2f}%")
        print(f"  %B: {latest['percent_b']:.2f}%")

        # 매매 신호 테스트
        signals = bb.get_band_signals()
        print(f"매매 신호: {signals['primary_signal']} (강도: {signals['signal_strength']})")
        print(f"세부 신호: {signals['signals']}")

    return bb, results


def test_stochastic_basic():
    """스토캐스틱 기본 테스트"""
    print("\n=== 스토캐스틱 오실레이터 기본 테스트 ===")

    # 테스트 데이터 생성
    test_data = generate_test_data(30)

    # 스토캐스틱 계산기 생성
    stoch = StochasticOscillator(period=14, smoothing_k=3, smoothing_d=3)

    # OHLC 데이터 추가
    results = []
    for i in range(len(test_data['close'])):
        ohlc = [test_data['high'][i], test_data['low'][i], test_data['close'][i]]
        result = stoch.add_ohlc_data(ohlc)
        if result:
            results.append(result)

    print(f"총 {len(results)}개의 스토캐스틱 값 계산됨")

    if results:
        latest = results[-1]
        print(f"최신 스토캐스틱 값:")
        for key, value in latest.items():
            print(f"  {key}: {value:.2f}")

        # 매매 신호 테스트
        signals = stoch.get_stochastic_signals()
        print(f"매매 신호: {signals['primary_signal']} (강도: {signals['signal_strength']})")
        print(f"세부 신호: {signals['signals']}")

    return stoch, results


def test_talib_comparison():
    """TA-Lib와 결과 비교"""
    if not TALIB_AVAILABLE:
        print("\n=== TA-Lib 비교 테스트 건너뜀 ===")
        return

    print("\n=== TA-Lib 결과 비교 테스트 ===")

    # 테스트 데이터 생성
    test_data = generate_test_data(50)
    close_prices = test_data['close']
    high_prices = test_data['high']
    low_prices = test_data['low']

    # 1. 볼린저 밴드 비교
    print("\n1. 볼린저 밴드 비교:")

    # 우리 구현
    bb = BollingerBands(period=20, stddev_multiplier=2.0)
    for price in close_prices:
        bb.add_data(price)

    our_bands = bb.get_band_history()

    # TA-Lib 구현
    ta_upper, ta_middle, ta_lower = talib.BBANDS(
        close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )

    # 유효한 값들만 비교 (NaN 제외)
    valid_indices = ~(np.isnan(ta_upper) | np.isnan(ta_middle) | np.isnan(ta_lower))
    if np.any(valid_indices) and len(our_bands['upper_band']) > 0:
        # 마지막 몇 개 값 비교
        compare_count = min(5, len(our_bands['upper_band']), np.sum(valid_indices))

        our_upper = our_bands['upper_band'][-compare_count:]
        our_middle = our_bands['middle_band'][-compare_count:]
        our_lower = our_bands['lower_band'][-compare_count:]

        ta_upper_valid = ta_upper[valid_indices][-compare_count:]
        ta_middle_valid = ta_middle[valid_indices][-compare_count:]
        ta_lower_valid = ta_lower[valid_indices][-compare_count:]

        print(f"  최근 {compare_count}개 값 비교:")
        print("  상단 밴드 - 우리/TA-Lib:")
        for i in range(compare_count):
            print(f"    {our_upper[i]:.4f} / {ta_upper_valid[i]:.4f} (차이: {abs(our_upper[i] - ta_upper_valid[i]):.6f})")

        # 평균 오차 계산
        upper_diff = np.mean(np.abs(our_upper - ta_upper_valid))
        middle_diff = np.mean(np.abs(our_middle - ta_middle_valid))
        lower_diff = np.mean(np.abs(our_lower - ta_lower_valid))

        print(f"  평균 절대 오차:")
        print(f"    상단: {upper_diff:.6f}")
        print(f"    중간: {middle_diff:.6f}")
        print(f"    하단: {lower_diff:.6f}")

    # 2. 스토캐스틱 비교
    print("\n2. 스토캐스틱 비교:")

    # 우리 구현
    stoch = StochasticOscillator(period=14, smoothing_k=3, smoothing_d=3)
    for i in range(len(close_prices)):
        ohlc = [high_prices[i], low_prices[i], close_prices[i]]
        stoch.add_ohlc_data(ohlc)

    our_stoch = stoch.get_stochastic_history()

    # TA-Lib 구현
    ta_slowk, ta_slowd = talib.STOCH(
        high_prices, low_prices, close_prices,
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )

    # 유효한 값들만 비교
    valid_stoch = ~(np.isnan(ta_slowk) | np.isnan(ta_slowd))
    if np.any(valid_stoch) and len(our_stoch['slow_k']) > 0:
        compare_count = min(5, len(our_stoch['slow_k']), np.sum(valid_stoch))

        our_k = our_stoch['slow_k'][-compare_count:]
        our_d = our_stoch['d'][-compare_count:]

        ta_k_valid = ta_slowk[valid_stoch][-compare_count:]
        ta_d_valid = ta_slowd[valid_stoch][-compare_count:]

        print(f"  최근 {compare_count}개 값 비교:")
        print("  Slow %K - 우리/TA-Lib:")
        for i in range(compare_count):
            print(f"    {our_k[i]:.4f} / {ta_k_valid[i]:.4f} (차이: {abs(our_k[i] - ta_k_valid[i]):.6f})")

        # 평균 오차 계산
        k_diff = np.mean(np.abs(our_k - ta_k_valid))
        d_diff = np.mean(np.abs(our_d - ta_d_valid))

        print(f"  평균 절대 오차:")
        print(f"    Slow %K: {k_diff:.6f}")
        print(f"    %D: {d_diff:.6f}")


async def test_bollinger_analyzer_comparison():
    """기존 BollingerAnalyzer와 결과 비교"""
    print("\n=== 기존 BollingerAnalyzer와 비교 테스트 ===")

    try:
        # 데이터베이스 설정
        db_config = DatabaseConfig()

        # BollingerAnalyzer 인스턴스 생성
        analyzer = BollingerAnalyzer(db_config)

        # 테스트 데이터 생성 (OHLCV 형태)
        test_data = generate_test_data(40)

        # OHLCV 딕셔너리 리스트로 변환
        historical_data = []
        for i in range(len(test_data['close'])):
            historical_data.append({
                'timestamp': datetime.now(),
                'open': float(test_data['open'][i]),
                'high': float(test_data['high'][i]),
                'low': float(test_data['low'][i]),
                'close': float(test_data['close'][i]),
                'volume': float(test_data['volume'][i])
            })

        # 기존 BollingerAnalyzer로 계산
        bb_data = analyzer.calculate_bollinger_bands(historical_data, period=20, stddev_multiplier=2.0)

        if bb_data:
            print("기존 BollingerAnalyzer 계산 성공")

            # 유효한 인덱스 찾기
            valid_indices = ~(np.isnan(bb_data['upper']) | np.isnan(bb_data['middle']) | np.isnan(bb_data['lower']))
            valid_count = np.sum(valid_indices)

            if valid_count > 0:
                analyzer_upper = bb_data['upper'][valid_indices]
                analyzer_middle = bb_data['middle'][valid_indices]
                analyzer_lower = bb_data['lower'][valid_indices]

                print(f"기존 구현에서 {valid_count}개 유효 값 생성")
                print(f"최신 값 - 상단: {analyzer_upper[-1]:.4f}, 중간: {analyzer_middle[-1]:.4f}, 하단: {analyzer_lower[-1]:.4f}")

                # 우리 구현으로 계산
                bb = BollingerBands(period=20, stddev_multiplier=2.0)
                for data_point in historical_data:
                    bb.add_data(data_point['close'])

                our_bands = bb.get_band_history()

                if len(our_bands['upper_band']) > 0:
                    print(f"새 구현에서 {len(our_bands['upper_band'])}개 값 생성")
                    print(f"최신 값 - 상단: {our_bands['upper_band'][-1]:.4f}, 중간: {our_bands['middle_band'][-1]:.4f}, 하단: {our_bands['lower_band'][-1]:.4f}")

                    # 비교 가능한 개수 계산
                    compare_count = min(len(our_bands['upper_band']), len(analyzer_upper))

                    if compare_count > 0:
                        # 마지막 값들 비교
                        our_upper_last = our_bands['upper_band'][-compare_count:]
                        our_middle_last = our_bands['middle_band'][-compare_count:]
                        our_lower_last = our_bands['lower_band'][-compare_count:]

                        analyzer_upper_last = analyzer_upper[-compare_count:]
                        analyzer_middle_last = analyzer_middle[-compare_count:]
                        analyzer_lower_last = analyzer_lower[-compare_count:]

                        # 차이 계산
                        upper_diff = np.mean(np.abs(our_upper_last - analyzer_upper_last))
                        middle_diff = np.mean(np.abs(our_middle_last - analyzer_middle_last))
                        lower_diff = np.mean(np.abs(our_lower_last - analyzer_lower_last))

                        print(f"\n최근 {compare_count}개 값 비교 결과:")
                        print(f"평균 절대 오차:")
                        print(f"  상단 밴드: {upper_diff:.6f}")
                        print(f"  중간 밴드: {middle_diff:.6f}")
                        print(f"  하단 밴드: {lower_diff:.6f}")

                        # 허용 오차 내인지 확인
                        tolerance = 1e-10  # 매우 작은 허용 오차
                        if upper_diff < tolerance and middle_diff < tolerance and lower_diff < tolerance:
                            print("✓ 두 구현의 결과가 일치합니다!")
                        else:
                            print("⚠ 두 구현 간에 차이가 있습니다.")
                else:
                    print("새 구현에서 결과를 생성하지 못했습니다.")
            else:
                print("기존 구현에서 유효한 값을 생성하지 못했습니다.")
        else:
            print("기존 BollingerAnalyzer 계산 실패")

    except Exception as e:
        print(f"BollingerAnalyzer 비교 테스트 중 오류: {e}")


def test_factory_methods():
    """팩토리 메서드 테스트"""
    print("\n=== 팩토리 메서드 테스트 ===")

    # 표준 변동성 지표 세트 생성
    indicators = VolatilityIndicatorFactory.create_standard_volatility_set()

    print(f"생성된 지표 개수: {len(indicators)}")
    print("생성된 지표들:")
    for name, indicator in indicators.items():
        print(f"  {name}: {type(indicator).__name__}")

    # 테스트 데이터로 몇 개 지표 테스트
    test_data = generate_test_data(20)

    # 볼린저 밴드 테스트
    bb_20_2 = indicators['bb_20_2']
    for price in test_data['close']:
        bb_20_2.add_data(price)

    if bb_20_2.is_warmed_up:
        bands = bb_20_2.get_current_bands()
        print(f"\nbb_20_2 결과: 밴드폭 {bands['band_width']:.2f}%, %B {bands['percent_b']:.2f}%")

    # 스토캐스틱 테스트
    stoch_14 = indicators['stoch_14_3_3']
    for i in range(len(test_data['close'])):
        ohlc = [test_data['high'][i], test_data['low'][i], test_data['close'][i]]
        stoch_14.add_ohlc_data(ohlc)

    if stoch_14.is_warmed_up:
        values = stoch_14.get_current_values()
        print(f"stoch_14_3_3 결과: Slow %K {values['slow_k']:.2f}, %D {values['d']:.2f}")


def test_signal_generation():
    """신호 생성 테스트"""
    print("\n=== 신호 생성 테스트 ===")

    # 과매수/과매도 상황 시뮬레이션
    def create_trend_data(base_price: float, trend: float, length: int) -> np.ndarray:
        """트렌드가 있는 가격 데이터 생성"""
        prices = [base_price]
        for i in range(1, length):
            change = trend + np.random.normal(0, 0.005)
            new_price = prices[-1] * (1 + change)
            prices.append(max(1.0, new_price))
        return np.array(prices)

    # 1. 상승 트렌드 후 과매수 상황
    print("\n1. 상승 트렌드 후 과매수 테스트:")
    uptrend_data = create_trend_data(100, 0.02, 30)  # 2% 상승 트렌드

    bb_up = BollingerBands(period=20, stddev_multiplier=2.0)
    for price in uptrend_data:
        bb_up.add_data(price)

    if bb_up.is_warmed_up:
        signals = bb_up.get_band_signals()
        bands = bb_up.get_current_bands()
        print(f"  최종 가격: {uptrend_data[-1]:.2f}")
        print(f"  %B: {bands['percent_b']:.2f}%")
        print(f"  신호: {signals['primary_signal']} (강도: {signals['signal_strength']})")
        print(f"  세부 신호: {signals['signals']}")

    # 2. 하락 트렌드 후 과매도 상황
    print("\n2. 하락 트렌드 후 과매도 테스트:")
    downtrend_data = create_trend_data(100, -0.015, 30)  # 1.5% 하락 트렌드

    bb_down = BollingerBands(period=20, stddev_multiplier=2.0)
    for price in downtrend_data:
        bb_down.add_data(price)

    if bb_down.is_warmed_up:
        signals = bb_down.get_band_signals()
        bands = bb_down.get_current_bands()
        print(f"  최종 가격: {downtrend_data[-1]:.2f}")
        print(f"  %B: {bands['percent_b']:.2f}%")
        print(f"  신호: {signals['primary_signal']} (강도: {signals['signal_strength']})")
        print(f"  세부 신호: {signals['signals']}")

    # 3. 스토캐스틱 골든/데드 크로스 테스트
    print("\n3. 스토캐스틱 크로스 신호 테스트:")

    # 크로스 상황을 만들기 위한 특별한 데이터
    cross_data = generate_test_data(25)
    stoch_cross = StochasticOscillator(period=14, smoothing_k=3, smoothing_d=3)

    for i in range(len(cross_data['close'])):
        ohlc = [cross_data['high'][i], cross_data['low'][i], cross_data['close'][i]]
        result = stoch_cross.add_ohlc_data(ohlc)

        if result and 'slow_k' in result and 'd' in result:
            signals = stoch_cross.get_stochastic_signals()
            if 'cross' in signals['primary_signal']:
                print(f"  크로스 신호 발견: {signals['primary_signal']}")
                print(f"  Slow %K: {result['slow_k']:.2f}, %D: {result['d']:.2f}")
                print(f"  신호 강도: {signals['signal_strength']}")
                break


def performance_test():
    """성능 테스트"""
    print("\n=== 성능 테스트 ===")

    import time

    # 대량 데이터 생성
    large_data = generate_test_data(1000)

    # 볼린저 밴드 성능 테스트
    print("1. 볼린저 밴드 성능:")
    start_time = time.time()

    bb = BollingerBands(period=20, stddev_multiplier=2.0)
    for price in large_data['close']:
        bb.add_data(price)

    bb_time = time.time() - start_time
    print(f"  1000개 데이터 처리 시간: {bb_time:.4f}초")
    print(f"  초당 처리량: {1000/bb_time:.0f} 데이터/초")

    # 메모리 사용량 확인
    status = bb.get_status()
    print(f"  메모리 사용량: {status['memory_usage_mb']:.4f} MB")

    # 스토캐스틱 성능 테스트
    print("\n2. 스토캐스틱 성능:")
    start_time = time.time()

    stoch = StochasticOscillator(period=14, smoothing_k=3, smoothing_d=3)
    for i in range(len(large_data['close'])):
        ohlc = [large_data['high'][i], large_data['low'][i], large_data['close'][i]]
        stoch.add_ohlc_data(ohlc)

    stoch_time = time.time() - start_time
    print(f"  1000개 데이터 처리 시간: {stoch_time:.4f}초")
    print(f"  초당 처리량: {1000/stoch_time:.0f} 데이터/초")

    stoch_status = stoch.get_status()
    print(f"  메모리 사용량: {stoch_status['memory_usage_mb']:.4f} MB")


async def main():
    """메인 테스트 함수"""
    print("변동성 지표 테스트 시작")
    print("=" * 50)

    try:
        # 기본 기능 테스트
        test_bollinger_bands_basic()
        test_stochastic_basic()

        # TA-Lib 비교 테스트
        test_talib_comparison()

        # 기존 BollingerAnalyzer와 비교
        await test_bollinger_analyzer_comparison()

        # 팩토리 메서드 테스트
        test_factory_methods()

        # 신호 생성 테스트
        test_signal_generation()

        # 성능 테스트
        performance_test()

        print("\n" + "=" * 50)
        print("모든 테스트 완료!")

    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())