"""
이동평균 지표 구현 테스트 스크립트

SMA와 EMA 구현의 정확성, 성능, 메모리 사용량을 검증합니다.
"""

import sys
import os
import time
import numpy as np
import logging
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from technical_indicators import SimpleMovingAverage, ExponentialMovingAverage, MovingAverageFactory

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_sma_accuracy():
    """SMA 계산 정확성 테스트"""
    print("\n=== SMA 정확성 테스트 ===")

    # 테스트 데이터: 알려진 결과를 가진 간단한 데이터
    test_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    period = 5

    sma = SimpleMovingAverage(period)
    results = sma.add_data(test_data)

    # 예상 결과 계산 (수동)
    expected_results = []
    for i in range(period-1, len(test_data)):
        window = test_data[i-period+1:i+1]
        expected_results.append(sum(window) / len(window))

    print(f"입력 데이터: {test_data}")
    print(f"SMA({period}) 결과: {results}")
    print(f"예상 결과: {expected_results}")

    # 정확성 검증
    if len(results) == len(expected_results):
        for i, (actual, expected) in enumerate(zip(results, expected_results)):
            if abs(actual - expected) < 1e-10:
                print(f"  테스트 {i+1}: PASS (실제: {actual:.6f}, 예상: {expected:.6f})")
            else:
                print(f"  테스트 {i+1}: FAIL (실제: {actual:.6f}, 예상: {expected:.6f})")
    else:
        print(f"길이 불일치: 실제 {len(results)}, 예상 {len(expected_results)}")


def test_ema_accuracy():
    """EMA 계산 정확성 테스트"""
    print("\n=== EMA 정확성 테스트 ===")

    # 간단한 테스트 데이터
    test_data = [22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29]
    period = 5

    ema = ExponentialMovingAverage(period)

    # 첫 번째 EMA는 SMA로 시작
    first_sma = sum(test_data[:period]) / period
    alpha = 2.0 / (period + 1)

    # 수동으로 EMA 계산
    expected_emas = [first_sma]
    current_ema = first_sma

    for price in test_data[period:]:
        current_ema = alpha * price + (1 - alpha) * current_ema
        expected_emas.append(current_ema)

    # 구현된 EMA로 계산
    results = ema.add_data(test_data)

    print(f"입력 데이터: {test_data}")
    print(f"EMA({period}) 알파: {alpha:.6f}")
    print(f"실제 결과: {[f'{x:.6f}' for x in results]}")
    print(f"예상 결과: {[f'{x:.6f}' for x in expected_emas]}")

    # 정확성 검증
    for i, (actual, expected) in enumerate(zip(results, expected_emas)):
        if abs(actual - expected) < 1e-6:
            print(f"  테스트 {i+1}: PASS")
        else:
            print(f"  테스트 {i+1}: FAIL (차이: {abs(actual - expected):.8f})")


def test_incremental_calculation():
    """증분 계산 테스트"""
    print("\n=== 증분 계산 테스트 ===")

    test_data = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    period = 5

    # 배치 계산
    sma_batch = SimpleMovingAverage(period)
    ema_batch = ExponentialMovingAverage(period)

    batch_sma_results = sma_batch.add_data(test_data)
    batch_ema_results = ema_batch.add_data(test_data)

    # 증분 계산
    sma_incremental = SimpleMovingAverage(period)
    ema_incremental = ExponentialMovingAverage(period)

    incremental_sma_results = []
    incremental_ema_results = []

    for price in test_data:
        sma_result = sma_incremental.add_data_incremental(price)
        ema_result = ema_incremental.add_data_incremental(price)

        if sma_result is not None:
            incremental_sma_results.append(sma_result)
        if ema_result is not None:
            incremental_ema_results.append(ema_result)

    print(f"SMA 배치 결과: {[f'{x:.4f}' for x in batch_sma_results]}")
    print(f"SMA 증분 결과: {[f'{x:.4f}' for x in incremental_sma_results]}")

    print(f"EMA 배치 결과: {[f'{x:.4f}' for x in batch_ema_results]}")
    print(f"EMA 증분 결과: {[f'{x:.4f}' for x in incremental_ema_results]}")

    # 일치성 검증
    sma_match = np.allclose(batch_sma_results, incremental_sma_results, rtol=1e-10)
    ema_match = np.allclose(batch_ema_results, incremental_ema_results, rtol=1e-10)

    print(f"SMA 배치-증분 일치: {'PASS' if sma_match else 'FAIL'}")
    print(f"EMA 배치-증분 일치: {'PASS' if ema_match else 'FAIL'}")


def test_performance():
    """성능 테스트"""
    print("\n=== 성능 테스트 ===")

    # 대용량 데이터 생성 (10만개)
    np.random.seed(42)
    large_data = 100 + np.cumsum(np.random.randn(100000) * 0.1)

    periods = [5, 20, 50, 200]

    for period in periods:
        print(f"\n--- Period {period} 성능 테스트 ---")

        # SMA 성능 테스트
        start_time = time.time()
        sma = SimpleMovingAverage(period)
        sma_results = sma.add_data(large_data)
        sma_time = time.time() - start_time

        # EMA 성능 테스트
        start_time = time.time()
        ema = ExponentialMovingAverage(period)
        ema_results = ema.add_data(large_data)
        ema_time = time.time() - start_time

        print(f"SMA 계산 시간: {sma_time:.4f}초")
        print(f"EMA 계산 시간: {ema_time:.4f}초")
        print(f"SMA 결과 개수: {len(sma_results)}")
        print(f"EMA 결과 개수: {len(ema_results)}")

        # 메모리 사용량
        print(f"SMA 메모리 사용: {sma.get_status()['memory_usage_mb']:.2f} MB")
        print(f"EMA 메모리 사용: {ema.get_status()['memory_usage_mb']:.2f} MB")


def test_edge_cases():
    """예외 상황 테스트"""
    print("\n=== 예외 상황 테스트 ===")

    # 1. 잘못된 period 값
    try:
        sma = SimpleMovingAverage(0)
        print("period=0 테스트: FAIL (예외가 발생해야 함)")
    except ValueError as e:
        print(f"period=0 테스트: PASS ({e})")

    # 2. 부족한 데이터
    sma = SimpleMovingAverage(10)
    result = sma.add_data([1, 2, 3])  # 10개보다 적은 데이터
    print(f"부족한 데이터 테스트: {'PASS' if len(result) == 0 else 'FAIL'}")

    # 3. NaN 값 처리
    try:
        sma = SimpleMovingAverage(5)
        test_data_with_nan = [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]
        result = sma.add_data(test_data_with_nan)
        print(f"NaN 처리 테스트: PASS (결과 길이: {len(result)})")
    except Exception as e:
        print(f"NaN 처리 테스트: FAIL ({e})")

    # 4. 무한값 처리
    try:
        ema = ExponentialMovingAverage(5)
        test_data_with_inf = [1, 2, 3, np.inf, 5]
        result = ema.add_data(test_data_with_inf)
        print("무한값 처리 테스트: FAIL (예외가 발생해야 함)")
    except ValueError as e:
        print(f"무한값 처리 테스트: PASS ({e})")


def test_factory():
    """팩토리 클래스 테스트"""
    print("\n=== 팩토리 클래스 테스트 ===")

    # 개별 생성
    sma_20 = MovingAverageFactory.create_sma(20)
    ema_12 = MovingAverageFactory.create_ema(12)

    print(f"SMA 생성: {sma_20}")
    print(f"EMA 생성: {ema_12}")

    # 표준 세트 생성
    ma_set = MovingAverageFactory.create_standard_set()
    print(f"표준 세트 생성: {len(ma_set)}개 지표")

    # 테스트 데이터로 모든 지표 계산
    test_data = np.random.randn(300) + 100

    for name, ma in ma_set.items():
        result = ma.add_data(test_data)
        current_value = ma.get_current_value()
        print(f"  {name}: 현재값 {current_value:.4f}, 워밍업 {ma.is_warmed_up}")


def test_memory_management():
    """메모리 관리 테스트"""
    print("\n=== 메모리 관리 테스트 ===")

    # 작은 max_history로 메모리 관리 테스트
    sma = SimpleMovingAverage(period=5, max_history=20)

    # 30개 데이터 추가 (max_history보다 많음)
    for i in range(30):
        sma.add_data_incremental(100 + i)

    status = sma.get_status()
    print(f"데이터 개수: {status['data_count']} (max_history: 20)")
    print(f"메모리 관리 테스트: {'PASS' if status['data_count'] <= 20 else 'FAIL'}")


def main():
    """모든 테스트 실행"""
    print("이동평균 지표 구현 종합 테스트 시작")
    print("=" * 50)

    try:
        test_sma_accuracy()
        test_ema_accuracy()
        test_incremental_calculation()
        test_performance()
        test_edge_cases()
        test_factory()
        test_memory_management()

        print("\n" + "=" * 50)
        print("모든 테스트 완료")

    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()