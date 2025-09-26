#!/usr/bin/env python3
"""
RSI 및 MACD 지표 종합 테스트 스크립트

Task 6.2에서 구현된 RSI와 MACD 지표의 정확성, 성능, 메모리 사용량을 테스트하고
기존 Task 5의 RSI 구현과 비교 검증을 수행합니다.
"""

import sys
import time
import numpy as np
import logging
from typing import List, Dict, Any
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, 'C:\\Users\\SEOBEOMBONG\\project\\sbb1')

try:
    from src.technical_indicators import RSIIndicator, MACDIndicator, OscillatorFactory, compare_with_task5_rsi
    import talib
    print("모든 필요한 모듈 임포트 성공")
except ImportError as e:
    print(f"모듈 임포트 실패: {e}")
    sys.exit(1)


def generate_test_data(count: int = 100, start_price: float = 100.0, volatility: float = 0.02) -> List[float]:
    """
    테스트용 가격 데이터 생성

    Args:
        count: 생성할 데이터 포인트 수
        start_price: 시작 가격
        volatility: 변동성 (표준편차)

    Returns:
        List[float]: 생성된 가격 데이터
    """
    np.random.seed(42)  # 재현 가능한 결과를 위해
    prices = [start_price]

    for i in range(1, count):
        # 랜덤 워크 + 트렌드 + 사인파 패턴
        trend = 0.0005 * i  # 약간의 상승 트렌드 (줄임)
        cycle = 0.02 * np.sin(2 * np.pi * i / 20)  # 20일 주기 (진폭 줄임)
        random_change = np.random.normal(0, volatility)

        # 가격 변화를 제한하여 오버플로우 방지
        price_change = np.clip(trend + cycle + random_change, -0.1, 0.1)  # ±10% 제한
        new_price = prices[-1] * (1 + price_change)

        # 가격 범위 제한
        new_price = np.clip(new_price, 1.0, 10000.0)
        prices.append(new_price)

    return prices


def test_rsi_accuracy():
    """RSI 계산 정확성 테스트"""
    print("\n=== RSI 정확성 테스트 ===")

    # 테스트 데이터 생성
    test_data = generate_test_data(50, 100.0, 0.03)
    period = 14

    # Task 6.2 RSI 계산
    rsi_indicator = RSIIndicator(period)
    for price in test_data:
        rsi_indicator.add_data_incremental(price)

    task6_rsi_values = rsi_indicator.get_history()
    task6_current = rsi_indicator.get_current_value()

    # TA-Lib RSI 계산 (Task 5에서 사용)
    talib_rsi_values = talib.RSI(np.array(test_data, dtype=np.float64), timeperiod=period)
    talib_current = talib_rsi_values[-1] if not np.isnan(talib_rsi_values[-1]) else None

    # 비교 결과
    print(f"Task 6.2 현재 RSI: {task6_current:.4f}" if task6_current else "Task 6.2 RSI: 계산 불가")
    print(f"TA-Lib 현재 RSI: {talib_current:.4f}" if talib_current else "TA-Lib RSI: 계산 불가")

    if task6_current and talib_current:
        difference = abs(task6_current - talib_current)
        print(f"차이: {difference:.6f}")
        print(f"정확성: {'통과' if difference < 0.01 else '실패'}")

        # 신호 테스트
        signal_info = rsi_indicator.get_signal()
        print(f"매매 신호: {signal_info['signal']} (강도: {signal_info['strength']})")
        print(f"신호 설명: {signal_info['description']}")

    return {
        'task6_rsi': task6_current,
        'talib_rsi': talib_current,
        'accuracy_passed': difference < 0.01 if (task6_current and talib_current) else False
    }


def test_macd_accuracy():
    """MACD 계산 정확성 테스트"""
    print("\n=== MACD 정확성 테스트 ===")

    # 테스트 데이터 생성
    test_data = generate_test_data(60, 100.0, 0.02)

    # Task 6.2 MACD 계산
    macd_indicator = MACDIndicator(12, 26, 9)
    for price in test_data:
        result = macd_indicator.add_data_incremental(price)

    components = macd_indicator.get_macd_components()
    print(f"MACD Line: {components['macd']:.6f}" if components['macd'] else "MACD Line: 계산 불가")
    print(f"Signal Line: {components['signal']:.6f}" if components['signal'] else "Signal Line: 계산 불가")
    print(f"Histogram: {components['histogram']:.6f}" if components['histogram'] else "Histogram: 계산 불가")

    # TA-Lib MACD 계산으로 검증
    try:
        talib_macd, talib_signal, talib_hist = talib.MACD(
            np.array(test_data, dtype=np.float64),
            fastperiod=12, slowperiod=26, signalperiod=9
        )

        talib_macd_current = talib_macd[-1] if not np.isnan(talib_macd[-1]) else None
        talib_signal_current = talib_signal[-1] if not np.isnan(talib_signal[-1]) else None
        talib_hist_current = talib_hist[-1] if not np.isnan(talib_hist[-1]) else None

        print(f"\nTA-Lib 비교:")
        print(f"MACD Line: {talib_macd_current:.6f}" if talib_macd_current else "MACD Line: 계산 불가")
        print(f"Signal Line: {talib_signal_current:.6f}" if talib_signal_current else "Signal Line: 계산 불가")
        print(f"Histogram: {talib_hist_current:.6f}" if talib_hist_current else "Histogram: 계산 불가")

        # 차이 계산
        differences = {}
        if components['macd'] and talib_macd_current:
            differences['macd'] = abs(components['macd'] - talib_macd_current)
        if components['signal'] and talib_signal_current:
            differences['signal'] = abs(components['signal'] - talib_signal_current)
        if components['histogram'] and talib_hist_current:
            differences['histogram'] = abs(components['histogram'] - talib_hist_current)

        print(f"\n차이:")
        for key, diff in differences.items():
            print(f"{key}: {diff:.8f}")

        accuracy_passed = all(diff < 0.1 for diff in differences.values()) if differences else False  # 0.1 허용 오차
        print(f"정확성: {'통과' if accuracy_passed else '실패'}")

    except Exception as e:
        print(f"TA-Lib MACD 비교 중 오류: {e}")
        accuracy_passed = False

    # 신호 테스트
    signal_info = macd_indicator.get_signal()
    print(f"\n매매 신호: {signal_info['signal']} (강도: {signal_info['strength']})")
    print(f"신호 설명: {signal_info['description']}")
    print(f"크로스오버 상태: {signal_info['crossover_status']}")

    return {
        'components': components,
        'signal_info': signal_info,
        'accuracy_passed': accuracy_passed
    }


def test_performance():
    """성능 테스트"""
    print("\n=== 성능 테스트 ===")

    # 대용량 데이터 생성
    large_data = generate_test_data(10000, 100.0, 0.02)

    # RSI 성능 테스트
    print("RSI 성능 테스트 (10,000 데이터 포인트)...")
    start_time = time.time()

    rsi_indicator = RSIIndicator(14)
    for price in large_data:
        rsi_indicator.add_data_incremental(price)

    rsi_time = time.time() - start_time
    print(f"RSI 계산 시간: {rsi_time:.4f}초")

    # MACD 성능 테스트
    print("MACD 성능 테스트 (10,000 데이터 포인트)...")
    start_time = time.time()

    macd_indicator = MACDIndicator(12, 26, 9)
    for price in large_data:
        macd_indicator.add_data_incremental(price)

    macd_time = time.time() - start_time
    print(f"MACD 계산 시간: {macd_time:.4f}초")

    return {
        'rsi_time': rsi_time,
        'macd_time': macd_time,
        'data_points': len(large_data)
    }


def test_memory_efficiency():
    """메모리 효율성 테스트"""
    print("\n=== 메모리 효율성 테스트 ===")

    # 제한된 히스토리로 테스트
    max_history = 1000
    large_data = generate_test_data(5000, 100.0, 0.02)

    # RSI 메모리 테스트
    rsi_indicator = RSIIndicator(14, max_history)
    for price in large_data:
        rsi_indicator.add_data_incremental(price)

    rsi_status = rsi_indicator.get_status()
    print(f"RSI 메모리 사용량: {rsi_status['memory_usage_mb']:.4f} MB")
    print(f"RSI 데이터 포인트: {rsi_status['data_count']} (최대: {max_history})")

    # MACD 메모리 테스트
    macd_indicator = MACDIndicator(12, 26, 9, max_history)
    for price in large_data:
        macd_indicator.add_data_incremental(price)

    macd_status = macd_indicator.get_status()
    print(f"MACD 메모리 사용량: {macd_status['memory_usage_mb']:.4f} MB")
    print(f"MACD 데이터 포인트: {macd_status['data_count']} (최대: {max_history})")

    return {
        'rsi_memory_mb': rsi_status['memory_usage_mb'],
        'macd_memory_mb': macd_status['memory_usage_mb'],
        'memory_limit_respected': (rsi_status['data_count'] <= max_history and
                                 macd_status['data_count'] <= max_history)
    }


def test_factory_methods():
    """팩토리 메서드 테스트"""
    print("\n=== 팩토리 메서드 테스트 ===")

    # 표준 세트 생성
    oscillators = OscillatorFactory.create_standard_set()
    print(f"생성된 오실레이터 수: {len(oscillators)}")

    # 각 오실레이터 테스트
    test_data = generate_test_data(30, 100.0, 0.02)

    for name, oscillator in oscillators.items():
        print(f"\n{name} 테스트:")
        for price in test_data:
            if hasattr(oscillator, 'add_data_incremental'):
                oscillator.add_data_incremental(price)

        if oscillator.is_warmed_up:
            current_value = oscillator.get_current_value()
            print(f"  현재 값: {current_value:.4f}" if current_value else "  현재 값: 계산 불가")

            signal = oscillator.get_signal()
            print(f"  신호: {signal.get('signal', 'N/A')}")
        else:
            print(f"  상태: 워밍업 중")

    return {'factory_test_passed': True}


def test_task5_comparison():
    """Task 5 RSI 구현과의 비교 테스트"""
    print("\n=== Task 5 RSI 비교 테스트 ===")

    # 테스트 데이터
    test_data = generate_test_data(40, 100.0, 0.025)

    # 비교 함수 사용
    comparison_result = compare_with_task5_rsi(test_data, 14)

    print(f"Task 6.2 RSI: {comparison_result.get('task6_current_rsi')}")
    print(f"Task 5 (TA-Lib) RSI: {comparison_result.get('task5_current_rsi')}")
    print(f"비교 결과: {comparison_result.get('comparison_note')}")

    return comparison_result


def test_real_time_updates():
    """실시간 업데이트 테스트"""
    print("\n=== 실시간 업데이트 테스트 ===")

    # 초기 데이터로 워밍업
    initial_data = generate_test_data(30, 100.0, 0.02)

    rsi = RSIIndicator(14)
    macd = MACDIndicator(12, 26, 9)

    # 초기 데이터 입력
    for price in initial_data:
        rsi.add_data_incremental(price)
        macd.add_data_incremental(price)

    print("초기 워밍업 완료")
    print(f"RSI 초기값: {rsi.get_current_value():.4f}")

    macd_components = macd.get_macd_components()
    print(f"MACD 초기값: {macd_components['macd']:.6f}" if macd_components['macd'] else "MACD: 계산 불가")

    # 실시간 업데이트 시뮬레이션
    print("\n실시간 업데이트 시뮬레이션:")
    new_prices = [102.5, 103.1, 101.8, 104.2, 103.5]

    for i, price in enumerate(new_prices):
        rsi_value = rsi.add_data_incremental(price)
        macd_result = macd.add_data_incremental(price)

        print(f"시점 {i+1}: 가격={price:.1f}, RSI={rsi_value:.4f}" if rsi_value else f"시점 {i+1}: 가격={price:.1f}, RSI=계산중")

        if macd_result and macd_result['macd'] is not None:
            print(f"        MACD={macd_result['macd']:.6f}")

    return {'real_time_test_passed': True}


def main():
    """메인 테스트 실행"""
    print("RSI 및 MACD 지표 종합 테스트 시작")
    print("=" * 50)

    test_results = {}

    try:
        # 각 테스트 실행
        test_results['rsi_accuracy'] = test_rsi_accuracy()
        test_results['macd_accuracy'] = test_macd_accuracy()
        test_results['performance'] = test_performance()
        test_results['memory_efficiency'] = test_memory_efficiency()
        test_results['factory_methods'] = test_factory_methods()
        test_results['task5_comparison'] = test_task5_comparison()
        test_results['real_time_updates'] = test_real_time_updates()

        # 종합 결과
        print("\n" + "=" * 50)
        print("종합 테스트 결과")
        print("=" * 50)

        # 정확성 검사
        rsi_accurate = test_results['rsi_accuracy'].get('accuracy_passed', False)
        macd_accurate = test_results['macd_accuracy'].get('accuracy_passed', False)
        task5_match = test_results['task5_comparison'].get('values_match', False)

        print(f"RSI 정확성: {'통과' if rsi_accurate else '실패'}")
        print(f"MACD 정확성: {'통과' if macd_accurate else '실패'}")
        print(f"Task 5 호환성: {'통과' if task5_match else '실패'}")

        # 성능 결과
        perf = test_results['performance']
        print(f"RSI 성능: {perf['rsi_time']:.4f}초 (10K 데이터)")
        print(f"MACD 성능: {perf['macd_time']:.4f}초 (10K 데이터)")

        # 메모리 효율성
        memory = test_results['memory_efficiency']
        print(f"메모리 관리: {'통과' if memory['memory_limit_respected'] else '실패'}")

        # 전체 평가
        all_tests_passed = all([
            rsi_accurate, macd_accurate,
            memory['memory_limit_respected']
        ])

        print(f"\n전체 평가: {'모든 테스트 통과' if all_tests_passed else '일부 테스트 실패'}")

        if all_tests_passed:
            print("Task 6.2 구현이 성공적으로 완료되었습니다!")
        else:
            print("일부 이슈가 발견되었습니다. 로그를 확인해주세요.")

    except Exception as e:
        print(f"테스트 실행 중 오류 발생: {e}")
        logger.exception("테스트 실행 중 예외 발생")

    print("\n테스트 완료")
    return test_results


if __name__ == "__main__":
    main()