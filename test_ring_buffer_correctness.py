"""
링 버퍼 기반 메모리 최적화 구조 정확성 테스트

링 버퍼의 정확성, 기존 지표들과의 결과 비교, 에지 케이스 처리를 검증합니다.
"""

import unittest
import numpy as np
import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from technical_indicators.ring_buffer import (
    RingBuffer, SlidingWindow, CachedIndicatorEngine,
    OptimizedTechnicalIndicatorEngine, sma_function, ema_function
)
from technical_indicators.moving_averages import SimpleMovingAverage, ExponentialMovingAverage


class TestRingBuffer(unittest.TestCase):
    """링 버퍼 기본 기능 테스트"""

    def setUp(self):
        self.ring_buffer = RingBuffer(capacity=10)

    def test_initialization(self):
        """초기화 테스트"""
        self.assertEqual(self.ring_buffer.capacity, 10)
        self.assertEqual(self.ring_buffer.size, 0)
        self.assertFalse(self.ring_buffer.is_full())

    def test_append_single_value(self):
        """단일 값 추가 테스트"""
        self.ring_buffer.append(1.0)
        self.assertEqual(self.ring_buffer.size, 1)
        self.assertEqual(self.ring_buffer.get_latest(), 1.0)

    def test_append_multiple_values(self):
        """다중 값 추가 테스트"""
        values = [1, 2, 3, 4, 5]
        for value in values:
            self.ring_buffer.append(value)

        self.assertEqual(self.ring_buffer.size, 5)
        data = self.ring_buffer.get_data()
        np.testing.assert_array_equal(data, values)

    def test_circular_behavior(self):
        """순환 동작 테스트 (용량 초과)"""
        # 용량보다 많은 데이터 추가
        for i in range(15):
            self.ring_buffer.append(i)

        self.assertTrue(self.ring_buffer.is_full())
        self.assertEqual(self.ring_buffer.size, 10)

        # 최근 10개 값만 보존되어야 함 (5~14)
        expected = list(range(5, 15))
        data = self.ring_buffer.get_data()
        np.testing.assert_array_equal(data, expected)

    def test_get_data_with_count(self):
        """특정 개수 데이터 조회 테스트"""
        for i in range(10):
            self.ring_buffer.append(i)

        # 최근 5개 데이터 조회
        data = self.ring_buffer.get_data(5)
        expected = [5, 6, 7, 8, 9]
        np.testing.assert_array_equal(data, expected)

    def test_get_window(self):
        """윈도우 조회 테스트"""
        for i in range(10):
            self.ring_buffer.append(i)

        # 인덱스 2~5 윈도우 조회
        window = self.ring_buffer.get_window(2, 6)
        expected = [2, 3, 4, 5]
        np.testing.assert_array_equal(window, expected)

        # 음수 인덱스 테스트
        window = self.ring_buffer.get_window(-3, None)
        expected = [7, 8, 9]
        np.testing.assert_array_equal(window, expected)

    def test_clear(self):
        """초기화 테스트"""
        for i in range(5):
            self.ring_buffer.append(i)

        self.ring_buffer.clear()
        self.assertEqual(self.ring_buffer.size, 0)
        self.assertIsNone(self.ring_buffer.get_latest())

    def test_memory_usage(self):
        """메모리 사용량 정보 테스트"""
        memory_info = self.ring_buffer.memory_usage()

        self.assertIn('buffer_size_bytes', memory_info)
        self.assertIn('buffer_size_mb', memory_info)
        self.assertIn('capacity', memory_info)
        self.assertIn('current_size', memory_info)
        self.assertIn('utilization_percent', memory_info)

        self.assertEqual(memory_info['capacity'], 10)
        self.assertEqual(memory_info['current_size'], 0)

    def test_thread_safety_basic(self):
        """기본적인 스레드 안전성 테스트"""
        import threading

        def add_data():
            for i in range(100):
                self.ring_buffer.append(i)

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=add_data)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # 데이터가 정상적으로 추가되었는지 확인
        self.assertEqual(self.ring_buffer.size, 10)  # 용량만큼만 저장


class TestSlidingWindow(unittest.TestCase):
    """슬라이딩 윈도우 테스트"""

    def setUp(self):
        self.ring_buffer = RingBuffer(capacity=20)
        self.sliding_window = SlidingWindow(self.ring_buffer, window_size=5)

    def test_initialization(self):
        """초기화 테스트"""
        self.assertEqual(self.sliding_window.window_size, 5)
        self.assertEqual(self.sliding_window.available_size, 0)
        self.assertFalse(self.sliding_window.is_ready())

    def test_window_ready(self):
        """윈도우 준비 상태 테스트"""
        # 충분한 데이터가 없을 때
        for i in range(3):
            self.ring_buffer.append(i)
        self.assertFalse(self.sliding_window.is_ready())

        # 충분한 데이터가 있을 때
        for i in range(3, 7):
            self.ring_buffer.append(i)
        self.assertTrue(self.sliding_window.is_ready())

    def test_current_window(self):
        """현재 윈도우 테스트"""
        for i in range(10):
            self.ring_buffer.append(i)

        window = self.sliding_window.get_current_window()
        expected = [5, 6, 7, 8, 9]  # 최근 5개
        np.testing.assert_array_equal(window, expected)

    def test_window_at_offset(self):
        """오프셋 윈도우 테스트"""
        for i in range(10):
            self.ring_buffer.append(i)

        # 2스텝 이전 윈도우
        window = self.sliding_window.get_window_at_offset(-2)
        expected = [3, 4, 5, 6, 7]
        np.testing.assert_array_equal(window, expected)

    def test_available_size(self):
        """사용 가능한 크기 테스트"""
        # 데이터가 윈도우 크기보다 적을 때
        for i in range(3):
            self.ring_buffer.append(i)
        self.assertEqual(self.sliding_window.available_size, 3)

        # 데이터가 윈도우 크기보다 많을 때
        for i in range(3, 10):
            self.ring_buffer.append(i)
        self.assertEqual(self.sliding_window.available_size, 5)


class TestCachedIndicatorEngine(unittest.TestCase):
    """캐시된 지표 엔진 테스트"""

    def setUp(self):
        self.ring_buffer = RingBuffer(capacity=100)
        self.cache_engine = CachedIndicatorEngine(self.ring_buffer)

    def test_register_indicator(self):
        """지표 등록 테스트"""
        self.cache_engine.register_indicator('sma_5', sma_function, 5)

        stats = self.cache_engine.get_cache_stats()
        self.assertIn('sma_5', stats['registered_indicators'])

    def test_calculate_indicator_insufficient_data(self):
        """데이터 부족 시 지표 계산 테스트"""
        self.cache_engine.register_indicator('sma_10', sma_function, 10)

        # 데이터가 부족한 경우
        for i in range(5):
            self.ring_buffer.append(i)

        result = self.cache_engine.calculate_indicator('sma_10')
        self.assertIsNone(result)

    def test_calculate_indicator_sufficient_data(self):
        """충분한 데이터로 지표 계산 테스트"""
        self.cache_engine.register_indicator('sma_5', sma_function, 5)

        # 충분한 데이터 추가
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for value in test_data:
            self.ring_buffer.append(value)

        result = self.cache_engine.calculate_indicator('sma_5')
        expected = np.mean([6, 7, 8, 9, 10])  # 최근 5개의 평균
        self.assertAlmostEqual(result, expected, places=6)

    def test_cache_hit_miss(self):
        """캐시 히트/미스 테스트"""
        self.cache_engine.register_indicator('sma_3', sma_function, 3)

        for i in range(5):
            self.ring_buffer.append(i)

        # 첫 번째 계산 (캐시 미스)
        result1 = self.cache_engine.calculate_indicator('sma_3')

        # 두 번째 계산 (캐시 히트 - 데이터 변화 없음)
        result2 = self.cache_engine.calculate_indicator('sma_3')

        self.assertEqual(result1, result2)

        stats = self.cache_engine.get_cache_stats()
        self.assertGreater(stats['cache_hits'], 0)

    def test_indicator_history(self):
        """지표 히스토리 테스트"""
        self.cache_engine.register_indicator('sma_3', sma_function, 3)

        for i in range(10):
            self.ring_buffer.append(i)
            if i >= 2:  # 3개 이상일 때부터 계산
                self.cache_engine.calculate_indicator('sma_3')

        history = self.cache_engine.get_indicator_history('sma_3')
        self.assertGreater(len(history), 0)

    def test_clear_cache(self):
        """캐시 초기화 테스트"""
        self.cache_engine.register_indicator('sma_3', sma_function, 3)

        for i in range(5):
            self.ring_buffer.append(i)

        self.cache_engine.calculate_indicator('sma_3')

        # 특정 지표 캐시 초기화
        self.cache_engine.clear_cache('sma_3')

        # 전체 캐시 초기화
        self.cache_engine.clear_cache()

        stats = self.cache_engine.get_cache_stats()
        self.assertEqual(stats['cache_hits'], 0)
        self.assertEqual(stats['cache_misses'], 0)


class TestOptimizedTechnicalIndicatorEngine(unittest.TestCase):
    """최적화된 기술적 지표 엔진 테스트"""

    def setUp(self):
        self.engine = OptimizedTechnicalIndicatorEngine(capacity=50)

    def test_initialization(self):
        """초기화 테스트"""
        status = self.engine.get_status()
        self.assertEqual(status['buffer_capacity'], 50)
        self.assertEqual(status['buffer_size'], 0)
        self.assertGreater(len(status['registered_indicators']), 0)

    def test_add_data(self):
        """데이터 추가 테스트"""
        test_data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        for value in test_data:
            results = self.engine.add_data(value)
            if len(test_data) >= 5:  # 최소 데이터 확보 후
                self.assertIsInstance(results, dict)

    def test_get_indicator(self):
        """특정 지표 조회 테스트"""
        # 충분한 데이터 추가
        for i in range(25):
            self.engine.add_data(i)

        sma_5 = self.engine.get_indicator('sma_5')
        sma_20 = self.engine.get_indicator('sma_20')

        self.assertIsNotNone(sma_5)
        self.assertIsNotNone(sma_20)
        self.assertIsInstance(sma_5, float)

    def test_get_indicator_history(self):
        """지표 히스토리 조회 테스트"""
        # 충분한 데이터 추가
        for i in range(30):
            self.engine.add_data(i)

        history = self.engine.get_indicator_history('sma_10', count=5)
        self.assertEqual(len(history), 5)

    def test_register_custom_indicator(self):
        """사용자 정의 지표 등록 테스트"""
        def custom_max(data, **kwargs):
            return np.max(data)

        self.engine.register_custom_indicator('max_5', custom_max, 5)

        # 테스트 데이터 추가
        test_data = [1, 5, 3, 8, 2, 9, 4]
        for value in test_data:
            self.engine.add_data(value)

        result = self.engine.get_indicator('max_5')
        self.assertIsNotNone(result)

    def test_clear(self):
        """초기화 테스트"""
        # 데이터 추가
        for i in range(10):
            self.engine.add_data(i)

        # 초기화
        self.engine.clear()

        status = self.engine.get_status()
        self.assertEqual(status['buffer_size'], 0)


class TestAccuracyComparison(unittest.TestCase):
    """기존 구현과 정확성 비교 테스트"""

    def test_sma_accuracy(self):
        """SMA 정확성 비교"""
        # 테스트 데이터
        test_data = np.random.random(100) * 100
        period = 20

        # 기존 구현
        sma_original = SimpleMovingAverage(period=period)
        original_results = []
        for value in test_data:
            result = sma_original.add_data_incremental(value)
            if result is not None:
                original_results.append(result)

        # 최적화된 구현
        optimized_engine = OptimizedTechnicalIndicatorEngine(capacity=200)
        optimized_results = []
        for value in test_data:
            results = optimized_engine.add_data(value)
            if results[f'sma_{period}'] is not None:
                optimized_results.append(results[f'sma_{period}'])

        # 결과 비교 (길이가 다를 수 있으므로 최소 길이만큼 비교)
        min_length = min(len(original_results), len(optimized_results))
        if min_length > 0:
            original_subset = original_results[-min_length:]
            optimized_subset = optimized_results[-min_length:]

            # 상대 오차 확인 (1% 이내)
            for orig, opt in zip(original_subset, optimized_subset):
                relative_error = abs(orig - opt) / abs(orig) if orig != 0 else abs(opt)
                self.assertLess(relative_error, 0.01,
                               f"SMA 정확성 오차 초과: {orig} vs {opt}")

    def test_ema_accuracy(self):
        """EMA 정확성 비교"""
        # 테스트 데이터
        test_data = np.random.random(100) * 100
        period = 12

        # 기존 구현
        ema_original = ExponentialMovingAverage(period=period)
        original_results = []
        for value in test_data:
            result = ema_original.add_data_incremental(value)
            if result is not None:
                original_results.append(result)

        # 최적화된 구현
        optimized_engine = OptimizedTechnicalIndicatorEngine(capacity=200)
        optimized_results = []
        for value in test_data:
            results = optimized_engine.add_data(value)
            if results[f'ema_{period}'] is not None:
                optimized_results.append(results[f'ema_{period}'])

        # 결과 비교
        min_length = min(len(original_results), len(optimized_results))
        if min_length > 10:  # EMA는 초기값이 다를 수 있으므로 충분한 데이터 후 비교
            original_subset = original_results[-min_length+10:]
            optimized_subset = optimized_results[-min_length+10:]

            # 상대 오차 확인 (10% 이내 - EMA는 초기화 방식에 따라 차이 가능)
            for orig, opt in zip(original_subset, optimized_subset):
                relative_error = abs(orig - opt) / abs(orig) if orig != 0 else abs(opt)
                self.assertLess(relative_error, 0.10,
                               f"EMA 정확성 오차 초과: {orig} vs {opt}")


class TestEdgeCases(unittest.TestCase):
    """에지 케이스 테스트"""

    def test_invalid_capacity(self):
        """잘못된 용량으로 초기화 테스트"""
        with self.assertRaises(ValueError):
            RingBuffer(capacity=0)

        with self.assertRaises(ValueError):
            RingBuffer(capacity=-1)

    def test_invalid_window_size(self):
        """잘못된 윈도우 크기 테스트"""
        ring_buffer = RingBuffer(capacity=10)

        with self.assertRaises(ValueError):
            SlidingWindow(ring_buffer, window_size=0)

        with self.assertRaises(ValueError):
            SlidingWindow(ring_buffer, window_size=15)  # 버퍼 용량 초과

    def test_unregistered_indicator(self):
        """등록되지 않은 지표 계산 테스트"""
        ring_buffer = RingBuffer(capacity=10)
        cache_engine = CachedIndicatorEngine(ring_buffer)

        with self.assertRaises(ValueError):
            cache_engine.calculate_indicator('unknown_indicator')

    def test_empty_data_handling(self):
        """빈 데이터 처리 테스트"""
        ring_buffer = RingBuffer(capacity=10)

        # 빈 상태에서 조회
        self.assertEqual(len(ring_buffer.get_data()), 0)
        self.assertIsNone(ring_buffer.get_latest())

        # 빈 윈도우 조회
        window = ring_buffer.get_window(0, 5)
        self.assertEqual(len(window), 0)

    def test_nan_and_inf_handling(self):
        """NaN과 무한값 처리 테스트"""
        engine = OptimizedTechnicalIndicatorEngine(capacity=10)

        # 정상 데이터 추가
        for i in range(5):
            engine.add_data(i)

        # NaN 값 추가 시도 (처리 방식에 따라 다를 수 있음)
        try:
            engine.add_data(float('nan'))
        except (ValueError, TypeError):
            pass  # 예상되는 동작

        # 무한값 추가 시도
        try:
            engine.add_data(float('inf'))
        except (ValueError, TypeError):
            pass  # 예상되는 동작


def run_correctness_tests():
    """정확성 테스트 실행"""
    # 테스트 스위트 생성
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 테스트 클래스들 추가
    test_classes = [
        TestRingBuffer,
        TestSlidingWindow,
        TestCachedIndicatorEngine,
        TestOptimizedTechnicalIndicatorEngine,
        TestAccuracyComparison,
        TestEdgeCases
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 결과 요약
    print(f"\n{'='*60}")
    print("테스트 결과 요약")
    print(f"{'='*60}")
    print(f"실행된 테스트: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"오류: {len(result.errors)}")

    if result.failures:
        print(f"\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print(f"\n오류가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")

    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) /
                   result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\n성공률: {success_rate:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    print("링 버퍼 기반 메모리 최적화 구조 정확성 테스트 시작")
    print("="*60)

    success = run_correctness_tests()

    if success:
        print("\n모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n일부 테스트가 실패했습니다. 위의 결과를 확인하세요.")
        sys.exit(1)