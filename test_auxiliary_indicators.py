"""
보조 기술지표(Williams %R, CCI, ROC, ATR) 종합 테스트 스크립트

Task 6.4에서 구현된 보조 기술지표들의 정확성, 성능, Task 5와의 호환성을 검증합니다.
모든 테스트는 알려진 데이터로 계산 정확성을 검증하고, TA-Lib 결과와 비교합니다.
"""

import sys
import os
import time
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Task 6.4 보조지표 import
    from technical_indicators.auxiliary import (
        WilliamsRIndicator, CCIIndicator, ROCIndicator, ATRIndicator,
        AuxiliaryIndicatorFactory, compare_with_task5_atr
    )

    # TA-Lib import (비교용)
    import talib

    print("모든 필수 모듈 import 성공")

except ImportError as e:
    print(f"모듈 import 실패: {e}")
    sys.exit(1)


def generate_test_data(days: int = 50) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    테스트용 HLCV 데이터 생성

    Args:
        days: 생성할 일수

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: High, Low, Close, Volume 데이터
    """
    np.random.seed(42)  # 재현 가능한 테스트 데이터

    # 기본 가격에서 시작
    base_price = 50000.0
    prices = [base_price]

    for i in range(days - 1):
        # 가격 변동 (-3% ~ +3%)
        change = np.random.normal(0, 0.02)
        new_price = prices[-1] * (1 + change)
        # 최소 가격 제한
        new_price = max(new_price, base_price * 0.5)
        prices.append(new_price)

    # HLCV 생성
    high_prices = []
    low_prices = []
    close_prices = prices
    volumes = []

    for i, close in enumerate(close_prices):
        # High: Close 기준 0~2% 위
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        # Low: Close 기준 0~2% 아래
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        # Volume: 랜덤
        volume = np.random.uniform(1000, 10000)

        high_prices.append(high)
        low_prices.append(low)
        volumes.append(volume)

    return high_prices, low_prices, close_prices, volumes


def test_williams_r_indicator():
    """Williams %R 지표 테스트"""
    print("\n" + "="*50)
    print("Williams %R 지표 테스트")
    print("="*50)

    # 테스트 데이터 생성
    high_data, low_data, close_data, _ = generate_test_data(30)

    # Williams %R 계산기 초기화
    williams_r = WilliamsRIndicator(period=14)

    # 데이터 추가 및 계산
    williams_r_values = []
    for i in range(len(close_data)):
        result = williams_r.add_hlc_data(high_data[i], low_data[i], close_data[i])
        if result is not None:
            williams_r_values.append(result)

    # TA-Lib과 비교 (Williams %R)
    try:
        talib_williams_r = talib.WILLR(
            np.array(high_data, dtype=np.float64),
            np.array(low_data, dtype=np.float64),
            np.array(close_data, dtype=np.float64),
            timeperiod=14
        )

        # 유효한 값들만 비교
        valid_talib = talib_williams_r[~np.isnan(talib_williams_r)]

        if len(williams_r_values) > 0 and len(valid_talib) > 0:
            # 마지막 값 비교
            our_last = williams_r_values[-1]
            talib_last = valid_talib[-1]
            difference = abs(our_last - talib_last)

            print(f"Williams %R 계산 결과:")
            print(f"  - 우리 구현: {our_last:.4f}")
            print(f"  - TA-Lib: {talib_last:.4f}")
            print(f"  - 차이: {difference:.6f}")
            print(f"  - 일치 여부: {'일치' if difference < 0.01 else '불일치'}")

            # 신호 테스트
            signal = williams_r.get_signal()
            print(f"  - 현재 신호: {signal['signal']} (강도: {signal['strength']})")
            print(f"  - 설명: {signal['description']}")
        else:
            print("계산된 Williams %R 값이 없습니다")

    except Exception as e:
        print(f"TA-Lib 비교 중 오류: {e}")

    # 상태 정보
    status = williams_r.get_status()
    print(f"  - 데이터 개수: {status['data_count']}")
    print(f"  - 워밍업 완료: {status['is_warmed_up']}")
    print(f"  - 메모리 사용량: {status['memory_usage_mb']:.3f} MB")


def test_cci_indicator():
    """CCI 지표 테스트"""
    print("\n" + "="*50)
    print("CCI(Commodity Channel Index) 지표 테스트")
    print("="*50)

    # 테스트 데이터 생성
    high_data, low_data, close_data, _ = generate_test_data(40)

    # CCI 계산기 초기화
    cci = CCIIndicator(period=20)

    # 데이터 추가 및 계산
    cci_values = []
    for i in range(len(close_data)):
        result = cci.add_hlc_data(high_data[i], low_data[i], close_data[i])
        if result is not None:
            cci_values.append(result)

    # TA-Lib과 비교
    try:
        talib_cci = talib.CCI(
            np.array(high_data, dtype=np.float64),
            np.array(low_data, dtype=np.float64),
            np.array(close_data, dtype=np.float64),
            timeperiod=20
        )

        # 유효한 값들만 비교
        valid_talib = talib_cci[~np.isnan(talib_cci)]

        if len(cci_values) > 0 and len(valid_talib) > 0:
            # 마지막 값 비교
            our_last = cci_values[-1]
            talib_last = valid_talib[-1]
            difference = abs(our_last - talib_last)

            print(f"CCI 계산 결과:")
            print(f"  - 우리 구현: {our_last:.4f}")
            print(f"  - TA-Lib: {talib_last:.4f}")
            print(f"  - 차이: {difference:.6f}")
            print(f"  - 일치 여부: {'일치' if difference < 0.1 else '불일치'}")

            # 신호 테스트
            signal = cci.get_signal()
            print(f"  - 현재 신호: {signal['signal']} (강도: {signal['strength']})")
            print(f"  - 설명: {signal['description']}")
        else:
            print("계산된 CCI 값이 없습니다")

    except Exception as e:
        print(f"TA-Lib 비교 중 오류: {e}")

    # 상태 정보
    status = cci.get_status()
    print(f"  - 데이터 개수: {status['data_count']}")
    print(f"  - 워밍업 완료: {status['is_warmed_up']}")
    print(f"  - 메모리 사용량: {status['memory_usage_mb']:.3f} MB")


def test_roc_indicator():
    """ROC 지표 테스트"""
    print("\n" + "="*50)
    print("ROC(Rate of Change) 지표 테스트")
    print("="*50)

    # 테스트 데이터 생성
    _, _, close_data, _ = generate_test_data(25)

    # ROC 계산기 초기화
    roc = ROCIndicator(period=12)

    # 데이터 추가 및 계산
    roc_values = []
    for price in close_data:
        result = roc.add_data(price)
        if len(result) > 0:
            roc_values.extend(result)

    # TA-Lib과 비교
    try:
        talib_roc = talib.ROC(np.array(close_data, dtype=np.float64), timeperiod=12)

        # 유효한 값들만 비교
        valid_talib = talib_roc[~np.isnan(talib_roc)]

        if len(roc_values) > 0 and len(valid_talib) > 0:
            # 마지막 값 비교
            our_last = roc_values[-1]
            talib_last = valid_talib[-1]
            difference = abs(our_last - talib_last)

            print(f"ROC 계산 결과:")
            print(f"  - 우리 구현: {our_last:.4f}%")
            print(f"  - TA-Lib: {talib_last:.4f}%")
            print(f"  - 차이: {difference:.6f}%")
            print(f"  - 일치 여부: {'일치' if difference < 0.01 else '불일치'}")

            # 신호 테스트
            signal = roc.get_signal()
            print(f"  - 현재 신호: {signal['signal']} (강도: {signal['strength']})")
            print(f"  - 설명: {signal['description']}")
            print(f"  - 모멘텀 방향: {signal['momentum_direction']}")
        else:
            print("계산된 ROC 값이 없습니다")

    except Exception as e:
        print(f"TA-Lib 비교 중 오류: {e}")

    # 상태 정보
    status = roc.get_status()
    print(f"  - 데이터 개수: {status['data_count']}")
    print(f"  - 워밍업 완료: {status['is_warmed_up']}")
    print(f"  - 메모리 사용량: {status['memory_usage_mb']:.3f} MB")


def test_atr_indicator():
    """ATR 지표 테스트"""
    print("\n" + "="*50)
    print("ATR(Average True Range) 지표 테스트")
    print("="*50)

    # 테스트 데이터 생성
    high_data, low_data, close_data, _ = generate_test_data(30)

    # ATR 계산기 초기화
    atr = ATRIndicator(period=14)

    # 데이터 추가 및 계산
    atr_values = []
    for i in range(len(close_data)):
        result = atr.add_hlc_data(high_data[i], low_data[i], close_data[i])
        if result is not None:
            atr_values.append(result)

    # Task 5 ATR과 비교
    hlc_data = list(zip(high_data, low_data, close_data))
    comparison_result = compare_with_task5_atr(hlc_data, period=14)

    print(f"ATR 계산 결과:")
    print(f"  - Task 6.4 구현: {comparison_result.get('task6_atr_value', 'N/A')}")
    print(f"  - Task 5 (TA-Lib): {comparison_result.get('task5_atr_value', 'N/A')}")
    if 'difference' in comparison_result:
        print(f"  - 차이: {comparison_result['difference']:.8f}")
    print(f"  - 일치 여부: {'일치' if comparison_result.get('values_match', False) else '불일치'}")
    print(f"  - {comparison_result.get('comparison_note', '비교 불가')}")

    # 변동성 분석 테스트
    if len(atr_values) > 0:
        current_price = close_data[-1]
        volatility_info = atr.get_volatility_level(current_price)

        print(f"  - 변동성 수준: {volatility_info['volatility_level']}")
        print(f"  - ATR 퍼센티지: {volatility_info.get('atr_percentage', 'N/A'):.4f}%")
        print(f"  - 설명: {volatility_info['description']}")

    # 상태 정보
    status = atr.get_status()
    print(f"  - 데이터 개수: {status['data_count']}")
    print(f"  - 워밍업 완료: {status['is_warmed_up']}")
    print(f"  - 메모리 사용량: {status['memory_usage_mb']:.3f} MB")


def test_factory_methods():
    """팩토리 메서드 테스트"""
    print("\n" + "="*50)
    print("AuxiliaryIndicatorFactory 테스트")
    print("="*50)

    # 표준 세트 생성
    standard_set = AuxiliaryIndicatorFactory.create_standard_auxiliary_set()

    print("표준 보조지표 세트:")
    for name, indicator in standard_set.items():
        print(f"  - {name}: {type(indicator).__name__}(period={indicator.period})")

    # 사용자 정의 세트 생성
    custom_config = {
        'fast_williams_r': {'type': 'williams_r', 'period': 7, 'overbought': -15, 'oversold': -85},
        'slow_cci': {'type': 'cci', 'period': 30, 'constant': 0.020},
        'short_roc': {'type': 'roc', 'period': 6},
        'long_atr': {'type': 'atr', 'period': 21}
    }

    custom_set = AuxiliaryIndicatorFactory.create_custom_set(custom_config)

    print("\n사용자 정의 보조지표 세트:")
    for name, indicator in custom_set.items():
        print(f"  - {name}: {type(indicator).__name__}(period={indicator.period})")


def test_performance():
    """성능 테스트"""
    print("\n" + "="*50)
    print("성능 및 메모리 사용량 테스트")
    print("="*50)

    # 대용량 데이터 생성
    large_high, large_low, large_close, _ = generate_test_data(1000)

    # 각 지표별 성능 테스트
    indicators = [
        ('Williams %R', WilliamsRIndicator(14)),
        ('CCI', CCIIndicator(20)),
        ('ROC', ROCIndicator(12)),
        ('ATR', ATRIndicator(14))
    ]

    for name, indicator in indicators:
        start_time = time.time()

        if hasattr(indicator, 'add_hlc_data'):
            # HLC 데이터 필요한 지표
            for i in range(len(large_close)):
                indicator.add_hlc_data(large_high[i], large_low[i], large_close[i])
        else:
            # Close 가격만 필요한 지표
            for price in large_close:
                indicator.add_data(price)

        end_time = time.time()
        processing_time = end_time - start_time

        status = indicator.get_status()

        print(f"{name} 성능:")
        print(f"  - 처리 시간: {processing_time:.4f}초")
        print(f"  - 데이터 개수: {status['data_count']}")
        print(f"  - 지표 개수: {status['indicator_count']}")
        print(f"  - 메모리 사용량: {status['memory_usage_mb']:.3f} MB")
        print(f"  - 초당 처리량: {len(large_close)/processing_time:.0f} 데이터/초")


def test_real_time_updates():
    """실시간 업데이트 테스트"""
    print("\n" + "="*50)
    print("실시간 업데이트 테스트")
    print("="*50)

    # Williams %R로 실시간 업데이트 테스트
    williams_r = WilliamsRIndicator(period=5)  # 짧은 기간으로 빠른 테스트

    print("순차적 데이터 추가 및 신호 변화:")

    # 테스트 데이터
    test_hlc = [
        (100, 95, 98),
        (102, 96, 100),
        (105, 99, 103),
        (104, 100, 102),
        (107, 101, 105),
        (110, 104, 108),  # 상승 추세
        (108, 103, 105),  # 하락 시작
        (106, 101, 103),
        (105, 100, 102),
        (103, 98, 100)    # 하락 추세
    ]

    for i, (high, low, close) in enumerate(test_hlc):
        result = williams_r.add_hlc_data(high, low, close)

        if result is not None:
            signal = williams_r.get_signal()
            print(f"  단계 {i+1}: Williams %R = {result:.2f}, 신호 = {signal['signal']}")


def main():
    """메인 테스트 실행"""
    print("보조 기술지표(Williams %R, CCI, ROC, ATR) 종합 테스트 시작")
    print(f"테스트 시작 시간: {datetime.now()}")

    try:
        # 개별 지표 테스트
        test_williams_r_indicator()
        test_cci_indicator()
        test_roc_indicator()
        test_atr_indicator()

        # 팩토리 메서드 테스트
        test_factory_methods()

        # 성능 테스트
        test_performance()

        # 실시간 업데이트 테스트
        test_real_time_updates()

        print("\n" + "="*50)
        print("모든 테스트 완료")
        print(f"테스트 종료 시간: {datetime.now()}")
        print("="*50)

    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)