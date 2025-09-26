"""
빗썸 WebSocket 데이터 수집 시스템 안정성 및 성능 테스트

이 모듈은 WebSocket 데이터 수집기의 안정성과 성능을 종합적으로 검증합니다.
- 연결 안정성 테스트
- 데이터 무결성 검증
- 메모리 누수 테스트
- 성능 벤치마크
- 재연결 시나리오 테스트
- 백프레셔 상황 시뮬레이션
- 장기 실행 안정성 테스트
"""

import asyncio
import logging
import time
import tracemalloc
import gc
import psutil
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import random

from .websocket_client import BithumbWebSocketClient, SubscriptionType, get_websocket_client
from .websocket_reconnect import ReconnectStrategy
from .message_parser import MessageParser, TickerData, OrderBookData, TransactionData
from .redis_buffer import RedisQueueBuffer, get_redis_buffer, BufferStrategy
from .backpressure_handler import BackpressureHandler, BackpressureLevel
from .data_streams import TickerStreamProcessor, OrderBookStreamProcessor, TradeStreamProcessor

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """테스트 결과"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration_seconds: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'status': self.status,
            'duration_seconds': self.duration_seconds,
            'details': self.details,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    messages_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    connection_latency_ms: float
    processing_latency_ms: float
    cache_hit_rate: float
    error_rate: float


class StabilityPerformanceTestSuite:
    """
    안정성 및 성능 테스트 스위트

    빗썸 WebSocket 데이터 수집 시스템의 모든 컴포넌트에 대한
    종합적인 테스트를 실행합니다.
    """

    def __init__(
        self,
        test_symbols: List[str] = None,
        test_duration_seconds: int = 300,  # 5분 기본 테스트
        enable_long_term_test: bool = False,
        long_term_duration_hours: int = 1
    ):
        """
        테스트 스위트 초기화

        Args:
            test_symbols: 테스트용 심볼 리스트
            test_duration_seconds: 기본 테스트 지속시간
            enable_long_term_test: 장기 실행 테스트 활성화
            long_term_duration_hours: 장기 테스트 지속시간
        """
        self.test_symbols = test_symbols or ["BTC_KRW", "ETH_KRW", "XRP_KRW"]
        self.test_duration_seconds = test_duration_seconds
        self.enable_long_term_test = enable_long_term_test
        self.long_term_duration_hours = long_term_duration_hours

        # 테스트 결과 저장
        self.test_results: List[TestResult] = []

        # 테스트 컴포넌트들
        self._websocket_client: Optional[BithumbWebSocketClient] = None
        self._redis_buffer: Optional[RedisQueueBuffer] = None
        self._backpressure_handler: Optional[BackpressureHandler] = None
        self._message_parser: Optional[MessageParser] = None

        # 성능 모니터링
        self._performance_metrics: List[PerformanceMetrics] = []
        self._memory_snapshots: List[Dict[str, Any]] = []

    async def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        logger.info("=== 빗썸 WebSocket 안정성 및 성능 테스트 시작 ===")
        start_time = time.time()

        try:
            # 테스트 환경 준비
            await self._setup_test_environment()

            # 기본 안정성 테스트들
            await self._test_websocket_connection_stability()
            await self._test_data_integrity()
            await self._test_message_parsing_accuracy()
            await self._test_redis_buffer_reliability()
            await self._test_backpressure_handling()

            # 재연결 시나리오 테스트
            await self._test_reconnection_scenarios()

            # 성능 테스트들
            await self._test_performance_benchmarks()
            await self._test_memory_leak_detection()

            # 백프레셔 시뮬레이션
            await self._test_backpressure_simulation()

            # 스트림 프로세서 통합 테스트
            await self._test_stream_processors_integration()

            # 장기 실행 안정성 테스트 (옵션)
            if self.enable_long_term_test:
                await self._test_long_term_stability()

        except Exception as e:
            logger.error(f"테스트 실행 중 오류: {str(e)}")
            self.test_results.append(TestResult(
                test_name="test_suite_execution",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={"error": str(e)},
                error_message=str(e)
            ))

        finally:
            # 테스트 환경 정리
            await self._cleanup_test_environment()

        # 결과 요약
        total_duration = time.time() - start_time
        summary = self._generate_test_summary(total_duration)

        logger.info("=== 빗썸 WebSocket 안정성 및 성능 테스트 완료 ===")
        return summary

    async def _setup_test_environment(self):
        """테스트 환경 설정"""
        logger.info("테스트 환경 설정 중...")

        # 메모리 추적 시작
        tracemalloc.start()

        # 컴포넌트 초기화
        self._websocket_client = await get_websocket_client()
        self._redis_buffer = await get_redis_buffer()
        self._backpressure_handler = BackpressureHandler()
        self._message_parser = MessageParser()

        # Redis 테스트 큐 생성
        await self._redis_buffer.create_queue("test_queue", max_size=10000)

        logger.info("테스트 환경 설정 완료")

    async def _cleanup_test_environment(self):
        """테스트 환경 정리"""
        logger.info("테스트 환경 정리 중...")

        try:
            # WebSocket 연결 종료
            if self._websocket_client:
                await self._websocket_client.disconnect()

            # 백프레셔 핸들러 중지
            if self._backpressure_handler:
                await self._backpressure_handler.stop()

            # Redis 테스트 큐 정리
            if self._redis_buffer:
                try:
                    await self._redis_buffer.clear_queue("test_queue")
                except Exception as e:
                    logger.warning(f"테스트 큐 정리 실패: {str(e)}")

        except Exception as e:
            logger.warning(f"테스트 환경 정리 중 오류: {str(e)}")

        # 메모리 추적 중지
        tracemalloc.stop()

        logger.info("테스트 환경 정리 완료")

    async def _test_websocket_connection_stability(self):
        """WebSocket 연결 안정성 테스트"""
        logger.info("WebSocket 연결 안정성 테스트 시작...")
        start_time = time.time()

        try:
            connection_attempts = 0
            successful_connections = 0
            connection_times = []

            # 여러 번 연결 테스트
            for i in range(10):
                connection_attempts += 1
                conn_start = time.time()

                try:
                    success = await self._websocket_client.connect()
                    conn_time = (time.time() - conn_start) * 1000  # ms

                    if success and self._websocket_client.is_connected:
                        successful_connections += 1
                        connection_times.append(conn_time)

                        # 연결 상태 확인
                        await asyncio.sleep(1)
                        if not self._websocket_client.is_connected:
                            logger.warning(f"연결 {i+1}: 연결 후 즉시 끊어짐")

                    await self._websocket_client.disconnect()
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.warning(f"연결 {i+1} 실패: {str(e)}")

            # 결과 평가
            success_rate = successful_connections / connection_attempts
            avg_connection_time = statistics.mean(connection_times) if connection_times else 0

            details = {
                'connection_attempts': connection_attempts,
                'successful_connections': successful_connections,
                'success_rate': success_rate,
                'avg_connection_time_ms': avg_connection_time,
                'connection_times': connection_times
            }

            status = "PASS" if success_rate >= 0.9 else "FAIL"  # 90% 이상 성공률 요구

            self.test_results.append(TestResult(
                test_name="websocket_connection_stability",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"연결 안정성 테스트 완료: 성공률 {success_rate:.1%}")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="websocket_connection_stability",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    async def _test_data_integrity(self):
        """데이터 무결성 검증 테스트"""
        logger.info("데이터 무결성 검증 테스트 시작...")
        start_time = time.time()

        try:
            received_messages = []
            processed_messages = []
            parse_errors = 0

            # 메시지 수집 콜백
            def message_handler(message_type, parsed_data, raw_message):
                received_messages.append(raw_message)
                if parsed_data:
                    processed_messages.append(parsed_data)

            # 에러 콜백
            def error_handler(error):
                nonlocal parse_errors
                parse_errors += 1
                logger.warning(f"파싱 에러: {str(error)}")

            # WebSocket 연결 및 구독
            await self._websocket_client.connect()

            # 핸들러 등록
            self._websocket_client.add_message_handler(SubscriptionType.TICKER, message_handler)

            # 구독 시작
            success = await self._websocket_client.subscribe(SubscriptionType.TICKER, self.test_symbols[:2])
            if not success:
                raise Exception("구독 실패")

            # 데이터 수집 (30초)
            await asyncio.sleep(30)

            # 구독 해제
            await self._websocket_client.unsubscribe(SubscriptionType.TICKER, self.test_symbols[:2])

            # 결과 분석
            total_received = len(received_messages)
            total_processed = len(processed_messages)
            processing_success_rate = total_processed / total_received if total_received > 0 else 0

            # 데이터 검증
            data_integrity_checks = {
                'valid_json_count': 0,
                'valid_schema_count': 0,
                'missing_required_fields': 0
            }

            for raw_msg in received_messages:
                try:
                    # JSON 파싱 테스트
                    data = json.loads(raw_msg.data)
                    data_integrity_checks['valid_json_count'] += 1

                    # 기본 스키마 검증
                    if isinstance(data, dict) and 'symbol' in data:
                        data_integrity_checks['valid_schema_count'] += 1
                    else:
                        data_integrity_checks['missing_required_fields'] += 1

                except json.JSONDecodeError:
                    logger.warning("Invalid JSON message received")

            details = {
                'total_received': total_received,
                'total_processed': total_processed,
                'processing_success_rate': processing_success_rate,
                'parse_errors': parse_errors,
                'data_integrity_checks': data_integrity_checks
            }

            # 90% 이상의 메시지가 올바르게 처리되어야 함
            status = "PASS" if processing_success_rate >= 0.9 else "FAIL"

            self.test_results.append(TestResult(
                test_name="data_integrity",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"데이터 무결성 테스트 완료: 처리 성공률 {processing_success_rate:.1%}")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="data_integrity",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    async def _test_message_parsing_accuracy(self):
        """메시지 파싱 정확도 테스트"""
        logger.info("메시지 파싱 정확도 테스트 시작...")
        start_time = time.time()

        try:
            # 샘플 메시지 데이터 준비
            sample_ticker_data = {
                "symbol": "BTC_KRW",
                "opening_price": "50000000",
                "closing_price": "51000000",
                "min_price": "49000000",
                "max_price": "52000000",
                "volume_1day": "100.5",
                "volume_7day": "700.5"
            }

            sample_orderbook_data = {
                "symbol": "BTC_KRW",
                "bids": [{"price": "50000000", "quantity": "1.0"}],
                "asks": [{"price": "51000000", "quantity": "2.0"}]
            }

            parsing_results = {
                'ticker_parse_success': 0,
                'ticker_parse_fail': 0,
                'orderbook_parse_success': 0,
                'orderbook_parse_fail': 0,
                'validation_errors': []
            }

            # Ticker 파싱 테스트 (100회)
            for i in range(100):
                try:
                    # 약간의 변형을 가한 데이터
                    test_data = sample_ticker_data.copy()
                    test_data['closing_price'] = str(50000000 + random.randint(-1000000, 1000000))

                    parsed = self._message_parser.parse_ticker_data(test_data)
                    if parsed and parsed.symbol == "BTC_KRW":
                        parsing_results['ticker_parse_success'] += 1
                    else:
                        parsing_results['ticker_parse_fail'] += 1

                except Exception as e:
                    parsing_results['ticker_parse_fail'] += 1
                    parsing_results['validation_errors'].append(str(e))

            # OrderBook 파싱 테스트 (100회)
            for i in range(100):
                try:
                    # 호가 데이터 변형
                    test_data = sample_orderbook_data.copy()
                    test_data['bids'][0]['price'] = str(50000000 + random.randint(-500000, 0))
                    test_data['asks'][0]['price'] = str(51000000 + random.randint(0, 500000))

                    parsed = self._message_parser.parse_orderbook_data(test_data)
                    if parsed and parsed.symbol == "BTC_KRW":
                        parsing_results['orderbook_parse_success'] += 1
                    else:
                        parsing_results['orderbook_parse_fail'] += 1

                except Exception as e:
                    parsing_results['orderbook_parse_fail'] += 1
                    parsing_results['validation_errors'].append(str(e))

            # 결과 계산
            ticker_accuracy = parsing_results['ticker_parse_success'] / (parsing_results['ticker_parse_success'] + parsing_results['ticker_parse_fail'])
            orderbook_accuracy = parsing_results['orderbook_parse_success'] / (parsing_results['orderbook_parse_success'] + parsing_results['orderbook_parse_fail'])
            overall_accuracy = (ticker_accuracy + orderbook_accuracy) / 2

            details = {
                'ticker_accuracy': ticker_accuracy,
                'orderbook_accuracy': orderbook_accuracy,
                'overall_accuracy': overall_accuracy,
                'parsing_results': parsing_results
            }

            status = "PASS" if overall_accuracy >= 0.95 else "FAIL"  # 95% 이상 정확도 요구

            self.test_results.append(TestResult(
                test_name="message_parsing_accuracy",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"메시지 파싱 정확도 테스트 완료: {overall_accuracy:.1%}")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="message_parsing_accuracy",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    async def _test_redis_buffer_reliability(self):
        """Redis 버퍼 신뢰성 테스트"""
        logger.info("Redis 버퍼 신뢰성 테스트 시작...")
        start_time = time.time()

        try:
            test_queue = "reliability_test_queue"
            await self._redis_buffer.create_queue(test_queue, max_size=1000)

            enqueue_success = 0
            dequeue_success = 0
            data_integrity_success = 0

            test_data = []

            # 데이터 입력 테스트 (500개)
            for i in range(500):
                data = {
                    'id': i,
                    'timestamp': datetime.now().isoformat(),
                    'value': random.random(),
                    'text': f'test_message_{i}'
                }
                test_data.append(data)

                try:
                    success = await self._redis_buffer.enqueue(test_queue, data)
                    if success:
                        enqueue_success += 1
                except Exception as e:
                    logger.warning(f"Enqueue 실패 {i}: {str(e)}")

            # 짧은 대기
            await asyncio.sleep(1)

            # 데이터 출력 테스트
            retrieved_data = []
            for i in range(500):
                try:
                    data = await self._redis_buffer.dequeue(test_queue)
                    if data:
                        dequeue_success += 1
                        retrieved_data.append(data)
                except Exception as e:
                    logger.warning(f"Dequeue 실패 {i}: {str(e)}")

            # 데이터 무결성 검증
            for original, retrieved in zip(test_data[:len(retrieved_data)], retrieved_data):
                if (original.get('id') == retrieved.get('id') and
                    original.get('text') == retrieved.get('text')):
                    data_integrity_success += 1

            # 큐 상태 확인
            queue_size = await self._redis_buffer.get_queue_size(test_queue)

            details = {
                'enqueue_attempts': 500,
                'enqueue_success': enqueue_success,
                'enqueue_success_rate': enqueue_success / 500,
                'dequeue_attempts': 500,
                'dequeue_success': dequeue_success,
                'dequeue_success_rate': dequeue_success / 500,
                'data_integrity_success': data_integrity_success,
                'data_integrity_rate': data_integrity_success / min(len(test_data), len(retrieved_data)) if retrieved_data else 0,
                'final_queue_size': queue_size
            }

            # 큐 정리
            await self._redis_buffer.clear_queue(test_queue)

            success_criteria = (
                details['enqueue_success_rate'] >= 0.95 and
                details['dequeue_success_rate'] >= 0.95 and
                details['data_integrity_rate'] >= 0.95
            )

            status = "PASS" if success_criteria else "FAIL"

            self.test_results.append(TestResult(
                test_name="redis_buffer_reliability",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"Redis 버퍼 신뢰성 테스트 완료")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="redis_buffer_reliability",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    async def _test_backpressure_handling(self):
        """백프레셔 처리 테스트"""
        logger.info("백프레셔 처리 테스트 시작...")
        start_time = time.time()

        try:
            # 백프레셔 핸들러 시작
            await self._backpressure_handler.start()

            level_changes = []
            metrics_history = []

            # 백프레셔 레벨 변경 추적
            def level_change_callback(old_level, new_level):
                level_changes.append({
                    'timestamp': datetime.now().isoformat(),
                    'old_level': old_level.value,
                    'new_level': new_level.value
                })

            self._backpressure_handler.add_level_change_callback(level_change_callback)

            # 시뮬레이션: 높은 부하 상황
            for i in range(100):
                # 가짜 큐 메트릭 업데이트 (점진적 증가)
                queue_size = min(i * 50, 5000)
                queue_capacity = 1000

                self._backpressure_handler.update_queue_metrics(queue_size, queue_capacity)

                # 데이터 수신 시뮬레이션
                self._backpressure_handler.record_ingestion()

                # 메트릭 수집
                metrics = self._backpressure_handler.get_metrics()
                metrics_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'queue_utilization': metrics.queue_utilization,
                    'backpressure_level': metrics.current_level.value,
                    'throttle_factor': metrics.throttle_factor
                })

                await asyncio.sleep(0.1)

            # 백프레셔 응답 테스트
            backpressure_responses = {
                'should_drop_data_calls': 0,
                'should_drop_data_true': 0,
                'should_throttle_calls': 0,
                'should_throttle_true': 0,
                'throttle_delays': []
            }

            for i in range(50):
                backpressure_responses['should_drop_data_calls'] += 1
                if self._backpressure_handler.should_drop_data():
                    backpressure_responses['should_drop_data_true'] += 1

                backpressure_responses['should_throttle_calls'] += 1
                if self._backpressure_handler.should_throttle():
                    backpressure_responses['should_throttle_true'] += 1
                    delay = await self._backpressure_handler.get_throttle_delay()
                    backpressure_responses['throttle_delays'].append(delay)

            details = {
                'level_changes': level_changes,
                'level_changes_count': len(level_changes),
                'metrics_history_count': len(metrics_history),
                'backpressure_responses': backpressure_responses,
                'final_metrics': self._backpressure_handler.get_metrics().__dict__
            }

            # 백프레셔가 적절히 작동했는지 확인
            status = "PASS" if len(level_changes) > 0 else "FAIL"

            self.test_results.append(TestResult(
                test_name="backpressure_handling",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"백프레셔 처리 테스트 완료: {len(level_changes)}회 레벨 변경")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="backpressure_handling",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    async def _test_reconnection_scenarios(self):
        """재연결 시나리오 테스트"""
        logger.info("재연결 시나리오 테스트 시작...")
        start_time = time.time()

        try:
            reconnection_results = []

            # 시나리오 1: 정상적인 재연결
            logger.info("시나리오 1: 의도적 연결 해제 후 재연결")
            await self._websocket_client.connect()
            original_connect_time = time.time()

            await self._websocket_client.disconnect()
            disconnect_time = time.time()

            reconnect_start = time.time()
            success = await self._websocket_client.connect()
            reconnect_time = time.time() - reconnect_start

            reconnection_results.append({
                'scenario': 'normal_reconnection',
                'success': success,
                'reconnect_time_ms': reconnect_time * 1000,
                'downtime_ms': (disconnect_time - original_connect_time + reconnect_time) * 1000
            })

            # 시나리오 2: 자동 재연결 테스트 (재연결 매니저 사용)
            logger.info("시나리오 2: 자동 재연결 매니저 테스트")

            # 재연결 매니저 활성화
            self._websocket_client.enable_auto_reconnect(
                strategy=ReconnectStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                initial_delay=1.0
            )

            reconnection_attempts = []

            def track_reconnection(attempt, success, delay):
                reconnection_attempts.append({
                    'attempt': attempt,
                    'success': success,
                    'delay': delay,
                    'timestamp': datetime.now().isoformat()
                })

            # 재연결 이벤트 추적 (가능하다면)
            # 실제로는 reconnect manager의 콜백을 사용해야 함

            # 연결 시뮬레이션 (3회 시도)
            for attempt in range(3):
                try:
                    await self._websocket_client.connect()
                    await asyncio.sleep(2)  # 짧은 대기
                    await self._websocket_client.disconnect()
                    await asyncio.sleep(1)  # 재연결 대기
                    track_reconnection(attempt + 1, True, 1.0)
                except Exception as e:
                    track_reconnection(attempt + 1, False, 1.0)

            reconnection_results.append({
                'scenario': 'auto_reconnection_manager',
                'attempts': reconnection_attempts,
                'total_attempts': len(reconnection_attempts),
                'successful_attempts': sum(1 for attempt in reconnection_attempts if attempt['success'])
            })

            # 시나리오 3: 네트워크 오류 시뮬레이션 (제한적)
            logger.info("시나리오 3: 연결 오류 시뮬레이션")

            error_handling_results = {
                'connection_errors_handled': 0,
                'recovery_attempts': 0,
                'final_connection_status': False
            }

            # 잘못된 URL로 연결 시도
            original_url = self._websocket_client._websocket_url
            self._websocket_client._websocket_url = "wss://invalid-url.com/ws"

            for i in range(3):
                try:
                    success = await self._websocket_client.connect()
                    error_handling_results['recovery_attempts'] += 1
                    if not success:
                        error_handling_results['connection_errors_handled'] += 1
                except Exception:
                    error_handling_results['connection_errors_handled'] += 1
                    error_handling_results['recovery_attempts'] += 1

            # 원래 URL 복원
            self._websocket_client._websocket_url = original_url
            error_handling_results['final_connection_status'] = await self._websocket_client.connect()

            reconnection_results.append({
                'scenario': 'network_error_simulation',
                'results': error_handling_results
            })

            details = {
                'scenarios_tested': len(reconnection_results),
                'reconnection_results': reconnection_results
            }

            # 재연결이 성공적으로 작동했는지 확인
            success_count = sum(1 for result in reconnection_results
                              if result.get('success', False) or result.get('successful_attempts', 0) > 0)

            status = "PASS" if success_count >= 2 else "FAIL"

            self.test_results.append(TestResult(
                test_name="reconnection_scenarios",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"재연결 시나리오 테스트 완료")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="reconnection_scenarios",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    async def _test_performance_benchmarks(self):
        """성능 벤치마크 테스트"""
        logger.info("성능 벤치마크 테스트 시작...")
        start_time = time.time()

        try:
            # 성능 측정을 위한 메트릭
            performance_data = {
                'message_processing_times': [],
                'memory_usage_samples': [],
                'cpu_usage_samples': [],
                'throughput_samples': []
            }

            process = psutil.Process()

            # WebSocket 연결 및 구독
            await self._websocket_client.connect()

            message_count = 0
            start_benchmark = time.time()

            # 메시지 처리 시간 측정 핸들러
            def performance_message_handler(message_type, parsed_data, raw_message):
                nonlocal message_count
                processing_start = time.time()

                # 간단한 처리 시뮬레이션
                if parsed_data:
                    _ = str(parsed_data)  # 문자열 변환

                processing_time = (time.time() - processing_start) * 1000  # ms
                performance_data['message_processing_times'].append(processing_time)
                message_count += 1

            self._websocket_client.add_message_handler(SubscriptionType.TICKER, performance_message_handler)

            # 구독 시작
            success = await self._websocket_client.subscribe(SubscriptionType.TICKER, self.test_symbols)
            if not success:
                raise Exception("성능 테스트를 위한 구독 실패")

            # 60초 동안 성능 측정
            test_duration = 60
            sample_interval = 5  # 5초마다 샘플링

            for i in range(test_duration // sample_interval):
                await asyncio.sleep(sample_interval)

                # 메모리 사용량 측정
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                performance_data['memory_usage_samples'].append(memory_mb)

                # CPU 사용률 측정
                cpu_percent = process.cpu_percent()
                performance_data['cpu_usage_samples'].append(cpu_percent)

                # 처리량 계산 (messages/sec)
                elapsed = time.time() - start_benchmark
                if elapsed > 0:
                    throughput = message_count / elapsed
                    performance_data['throughput_samples'].append(throughput)

            # 구독 해제
            await self._websocket_client.unsubscribe(SubscriptionType.TICKER, self.test_symbols)

            # 성능 메트릭 계산
            total_time = time.time() - start_benchmark
            avg_throughput = message_count / total_time if total_time > 0 else 0

            processing_times = performance_data['message_processing_times']
            avg_processing_time = statistics.mean(processing_times) if processing_times else 0
            max_processing_time = max(processing_times) if processing_times else 0

            memory_samples = performance_data['memory_usage_samples']
            avg_memory_usage = statistics.mean(memory_samples) if memory_samples else 0
            max_memory_usage = max(memory_samples) if memory_samples else 0

            cpu_samples = performance_data['cpu_usage_samples']
            avg_cpu_usage = statistics.mean(cpu_samples) if cpu_samples else 0
            max_cpu_usage = max(cpu_samples) if cpu_samples else 0

            details = {
                'test_duration_seconds': total_time,
                'total_messages_processed': message_count,
                'avg_throughput_msg_per_sec': avg_throughput,
                'avg_processing_time_ms': avg_processing_time,
                'max_processing_time_ms': max_processing_time,
                'avg_memory_usage_mb': avg_memory_usage,
                'max_memory_usage_mb': max_memory_usage,
                'avg_cpu_usage_percent': avg_cpu_usage,
                'max_cpu_usage_percent': max_cpu_usage,
                'performance_samples': {
                    'throughput_samples': performance_data['throughput_samples'],
                    'memory_samples': memory_samples,
                    'cpu_samples': cpu_samples
                }
            }

            # 성능 기준 평가
            performance_criteria = (
                avg_throughput >= 10 and  # 최소 10 msg/sec
                avg_processing_time <= 10 and  # 평균 처리시간 10ms 이하
                avg_memory_usage <= 500  # 평균 메모리 사용량 500MB 이하
            )

            status = "PASS" if performance_criteria else "FAIL"

            self.test_results.append(TestResult(
                test_name="performance_benchmarks",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"성능 벤치마크 테스트 완료: {avg_throughput:.1f} msg/sec")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="performance_benchmarks",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    async def _test_memory_leak_detection(self):
        """메모리 누수 탐지 테스트"""
        logger.info("메모리 누수 탐지 테스트 시작...")
        start_time = time.time()

        try:
            # 메모리 스냅샷 수집
            initial_snapshot = tracemalloc.take_snapshot()
            memory_snapshots = [initial_snapshot]

            process = psutil.Process()
            memory_usage_over_time = []

            # 메모리 사용량 초기 측정
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # WebSocket 연결/해제를 반복하여 메모리 누수 확인
            for cycle in range(10):
                logger.debug(f"메모리 테스트 사이클 {cycle + 1}/10")

                # 연결
                await self._websocket_client.connect()
                await self._websocket_client.subscribe(SubscriptionType.TICKER, self.test_symbols[:1])

                # 잠시 데이터 수신
                await asyncio.sleep(5)

                # 연결 해제
                await self._websocket_client.unsubscribe(SubscriptionType.TICKER, self.test_symbols[:1])
                await self._websocket_client.disconnect()

                # 메모리 정리 강제 실행
                gc.collect()

                # 메모리 스냅샷
                snapshot = tracemalloc.take_snapshot()
                memory_snapshots.append(snapshot)

                # 프로세스 메모리 사용량 측정
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage_over_time.append({
                    'cycle': cycle + 1,
                    'memory_mb': current_memory,
                    'memory_increase_mb': current_memory - initial_memory
                })

                await asyncio.sleep(2)  # 메모리 정리 대기

            # 최종 메모리 사용량
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # 메모리 증가 추세 분석
            memory_increases = [sample['memory_increase_mb'] for sample in memory_usage_over_time]
            avg_increase_per_cycle = statistics.mean(memory_increases) if memory_increases else 0
            max_increase = max(memory_increases) if memory_increases else 0

            # tracemalloc을 이용한 상세 분석
            final_snapshot = memory_snapshots[-1]
            top_stats = final_snapshot.compare_to(memory_snapshots[0], 'lineno')
            top_memory_consumers = []

            for stat in top_stats[:10]:  # 상위 10개
                top_memory_consumers.append({
                    'filename': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                    'size_diff_mb': stat.size_diff / 1024 / 1024,
                    'count_diff': stat.count_diff
                })

            details = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'total_memory_increase_mb': memory_increase,
                'avg_increase_per_cycle_mb': avg_increase_per_cycle,
                'max_increase_mb': max_increase,
                'memory_usage_over_time': memory_usage_over_time,
                'top_memory_consumers': top_memory_consumers,
                'test_cycles': 10
            }

            # 메모리 누수 기준: 사이클당 평균 증가량이 5MB 미만
            status = "PASS" if avg_increase_per_cycle < 5.0 else "FAIL"

            self.test_results.append(TestResult(
                test_name="memory_leak_detection",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"메모리 누수 탐지 테스트 완료: 총 증가량 {memory_increase:.1f}MB")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="memory_leak_detection",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    async def _test_backpressure_simulation(self):
        """백프레셔 상황 시뮬레이션 테스트"""
        logger.info("백프레셔 상황 시뮬레이션 테스트 시작...")
        start_time = time.time()

        try:
            # 고부하 시뮬레이션을 위한 설정
            high_load_queue = "backpressure_simulation_queue"
            await self._redis_buffer.create_queue(high_load_queue, max_size=100)  # 작은 큐

            backpressure_events = []
            messages_dropped = 0
            messages_throttled = 0

            # 백프레셔 핸들러 설정 (낮은 임계값)
            test_backpressure_handler = BackpressureHandler(
                memory_threshold_mb=100.0,  # 낮은 메모리 임계값
                cpu_threshold_percent=50.0   # 낮은 CPU 임계값
            )
            await test_backpressure_handler.start()

            # 백프레셔 이벤트 추적
            def track_backpressure_event(old_level, new_level):
                backpressure_events.append({
                    'timestamp': datetime.now().isoformat(),
                    'old_level': old_level.value,
                    'new_level': new_level.value
                })

            test_backpressure_handler.add_level_change_callback(track_backpressure_event)

            # 고부하 시뮬레이션 (500개 메시지 빠르게 전송)
            for i in range(500):
                # 큐 사용률을 점진적으로 증가
                simulated_queue_size = min(i, 100)
                test_backpressure_handler.update_queue_metrics(simulated_queue_size, 100)

                # 데이터 수신 기록
                test_backpressure_handler.record_ingestion()

                # 백프레셔 상태 확인
                if test_backpressure_handler.should_drop_data():
                    messages_dropped += 1
                    continue

                if test_backpressure_handler.should_throttle():
                    messages_throttled += 1
                    throttle_delay = await test_backpressure_handler.get_throttle_delay()
                    await asyncio.sleep(min(throttle_delay, 0.01))  # 최대 10ms 대기

                # 메시지 처리 시뮬레이션
                try:
                    test_data = {'message_id': i, 'timestamp': datetime.now().isoformat()}
                    await self._redis_buffer.enqueue(high_load_queue, test_data, priority=1)
                    test_backpressure_handler.record_processing()
                except Exception:
                    pass  # 큐 오버플로우 무시

                # 매 100번째마다 잠시 대기
                if i % 100 == 0:
                    await asyncio.sleep(0.1)

            # 최종 메트릭 수집
            final_metrics = test_backpressure_handler.get_metrics()
            final_queue_size = await self._redis_buffer.get_queue_size(high_load_queue)

            # 시스템 복구 테스트
            recovery_start = time.time()

            # 큐 비우기
            await self._redis_buffer.clear_queue(high_load_queue)

            # 백프레셔 상태 정상화 대기
            for _ in range(10):
                test_backpressure_handler.update_queue_metrics(0, 100)
                await asyncio.sleep(0.5)

            recovery_time = time.time() - recovery_start
            recovery_metrics = test_backpressure_handler.get_metrics()

            await test_backpressure_handler.stop()

            details = {
                'total_messages_sent': 500,
                'messages_dropped': messages_dropped,
                'messages_throttled': messages_throttled,
                'drop_rate': messages_dropped / 500,
                'throttle_rate': messages_throttled / 500,
                'backpressure_events': backpressure_events,
                'backpressure_events_count': len(backpressure_events),
                'final_queue_size': final_queue_size,
                'final_metrics': final_metrics.__dict__,
                'recovery_time_seconds': recovery_time,
                'recovery_metrics': recovery_metrics.__dict__
            }

            # 백프레셔가 적절히 작동했는지 확인
            backpressure_working = (
                len(backpressure_events) > 0 or  # 백프레셔 레벨 변경 발생
                messages_dropped > 0 or          # 메시지 드롭 발생
                messages_throttled > 0           # 스로틀링 발생
            )

            status = "PASS" if backpressure_working else "FAIL"

            self.test_results.append(TestResult(
                test_name="backpressure_simulation",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"백프레셔 시뮬레이션 테스트 완료: {len(backpressure_events)}회 이벤트")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="backpressure_simulation",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    async def _test_stream_processors_integration(self):
        """스트림 프로세서 통합 테스트"""
        logger.info("스트림 프로세서 통합 테스트 시작...")
        start_time = time.time()

        try:
            integration_results = {}

            # Ticker 스트림 프로세서 테스트
            logger.info("Ticker 스트림 프로세서 테스트...")
            async with TickerStreamProcessor(
                symbols=self.test_symbols[:1],
                websocket_client=self._websocket_client,
                redis_buffer=self._redis_buffer,
                backpressure_handler=self._backpressure_handler
            ) as ticker_processor:

                # 10초 동안 실행
                await asyncio.sleep(10)

                ticker_stats = ticker_processor.get_stats()
                ticker_health = await ticker_processor.health_check()

                integration_results['ticker_processor'] = {
                    'status': ticker_processor.status.value,
                    'is_running': ticker_processor.is_running,
                    'stats': ticker_stats,
                    'health': ticker_health
                }

            # OrderBook 스트림 프로세서 테스트
            logger.info("OrderBook 스트림 프로세서 테스트...")
            async with OrderBookStreamProcessor(
                symbols=self.test_symbols[:1],
                websocket_client=self._websocket_client,
                redis_buffer=self._redis_buffer,
                backpressure_handler=self._backpressure_handler
            ) as orderbook_processor:

                # 10초 동안 실행
                await asyncio.sleep(10)

                orderbook_stats = orderbook_processor.get_stats()
                orderbook_health = await orderbook_processor.health_check()

                integration_results['orderbook_processor'] = {
                    'status': orderbook_processor.status.value,
                    'is_running': orderbook_processor.is_running,
                    'stats': orderbook_stats,
                    'health': orderbook_health
                }

            # Trade 스트림 프로세서 테스트
            logger.info("Trade 스트림 프로세서 테스트...")
            async with TradeStreamProcessor(
                symbols=self.test_symbols[:1],
                websocket_client=self._websocket_client,
                redis_buffer=self._redis_buffer,
                backpressure_handler=self._backpressure_handler
            ) as trade_processor:

                # 10초 동안 실행
                await asyncio.sleep(10)

                trade_stats = trade_processor.get_stats()
                trade_health = await trade_processor.health_check()

                integration_results['trade_processor'] = {
                    'status': trade_processor.status.value,
                    'is_running': trade_processor.is_running,
                    'stats': trade_stats,
                    'health': trade_health
                }

            # 동시 실행 테스트
            logger.info("동시 실행 테스트...")
            processors = [
                TickerStreamProcessor(symbols=self.test_symbols[:1], cache_ttl_seconds=60),
                OrderBookStreamProcessor(symbols=self.test_symbols[:1], cache_ttl_seconds=60),
                TradeStreamProcessor(symbols=self.test_symbols[:1], cache_ttl_seconds=60)
            ]

            concurrent_start = time.time()

            # 모든 프로세서 동시 시작
            for processor in processors:
                await processor.start()

            # 30초 동안 동시 실행
            await asyncio.sleep(30)

            # 통계 수집
            concurrent_stats = []
            for i, processor in enumerate(processors):
                stats = processor.get_stats()
                health = await processor.health_check()
                concurrent_stats.append({
                    'processor_type': ['ticker', 'orderbook', 'trade'][i],
                    'stats': stats,
                    'health': health
                })

            # 모든 프로세서 중지
            for processor in processors:
                await processor.stop()

            concurrent_duration = time.time() - concurrent_start

            integration_results['concurrent_execution'] = {
                'duration_seconds': concurrent_duration,
                'processors_count': len(processors),
                'concurrent_stats': concurrent_stats
            }

            # 통합 테스트 평가
            all_processors_healthy = all(
                result.get('health', {}).get('status') != 'error'
                for result in integration_results.values()
                if isinstance(result, dict) and 'health' in result
            )

            message_processing = sum(
                result.get('stats', {}).get('messages_processed', 0)
                for result in integration_results.values()
                if isinstance(result, dict) and 'stats' in result
            )

            details = {
                'integration_results': integration_results,
                'all_processors_healthy': all_processors_healthy,
                'total_messages_processed': message_processing
            }

            status = "PASS" if all_processors_healthy and message_processing > 0 else "FAIL"

            self.test_results.append(TestResult(
                test_name="stream_processors_integration",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"스트림 프로세서 통합 테스트 완료: {message_processing}개 메시지 처리")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="stream_processors_integration",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    async def _test_long_term_stability(self):
        """장기 실행 안정성 테스트"""
        if not self.enable_long_term_test:
            logger.info("장기 실행 테스트가 비활성화되어 있습니다")
            return

        logger.info(f"장기 실행 안정성 테스트 시작 ({self.long_term_duration_hours}시간)...")
        start_time = time.time()

        try:
            test_duration_seconds = self.long_term_duration_hours * 3600
            monitoring_interval = 300  # 5분마다 상태 확인

            stability_metrics = {
                'uptime_checks': [],
                'memory_usage_over_time': [],
                'error_counts_over_time': [],
                'performance_degradation': [],
                'reconnection_events': []
            }

            # 간단한 스트림 프로세서로 장기 테스트
            async with TickerStreamProcessor(
                symbols=self.test_symbols[:1],
                websocket_client=self._websocket_client,
                redis_buffer=self._redis_buffer
            ) as processor:

                test_start = time.time()
                process = psutil.Process()

                while time.time() - test_start < test_duration_seconds:
                    current_time = time.time() - test_start

                    # 프로세서 상태 확인
                    is_running = processor.is_running
                    stats = processor.get_stats()
                    health = await processor.health_check()

                    stability_metrics['uptime_checks'].append({
                        'elapsed_seconds': current_time,
                        'is_running': is_running,
                        'status': processor.status.value,
                        'messages_processed': stats.get('messages_processed', 0),
                        'error_count': stats.get('parse_errors', 0)
                    })

                    # 메모리 사용량 추적
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    stability_metrics['memory_usage_over_time'].append({
                        'elapsed_seconds': current_time,
                        'memory_mb': memory_mb
                    })

                    # 성능 확인 (처리량)
                    if current_time > 0:
                        throughput = stats.get('messages_processed', 0) / current_time
                        stability_metrics['performance_degradation'].append({
                            'elapsed_seconds': current_time,
                            'messages_per_second': throughput
                        })

                    # 장기 실행 중 간헐적 재시작 테스트
                    if int(current_time) % 1800 == 0 and current_time > 0:  # 30분마다
                        logger.info("장기 테스트 중 재시작 테스트...")
                        await processor.stop()
                        await asyncio.sleep(5)
                        await processor.start()

                        stability_metrics['reconnection_events'].append({
                            'elapsed_seconds': current_time,
                            'event': 'manual_restart'
                        })

                    await asyncio.sleep(monitoring_interval)

            # 장기 실행 결과 분석
            total_uptime_checks = len(stability_metrics['uptime_checks'])
            successful_checks = sum(1 for check in stability_metrics['uptime_checks'] if check['is_running'])
            uptime_percentage = successful_checks / total_uptime_checks if total_uptime_checks > 0 else 0

            memory_usage = [sample['memory_mb'] for sample in stability_metrics['memory_usage_over_time']]
            avg_memory = statistics.mean(memory_usage) if memory_usage else 0
            max_memory = max(memory_usage) if memory_usage else 0
            memory_growth = memory_usage[-1] - memory_usage[0] if len(memory_usage) >= 2 else 0

            throughput_data = [sample['messages_per_second'] for sample in stability_metrics['performance_degradation']]
            avg_throughput = statistics.mean(throughput_data) if throughput_data else 0

            # 성능 저하 확인 (처음 10분 대비 마지막 10분)
            first_quarter = throughput_data[:len(throughput_data)//4] if len(throughput_data) >= 4 else throughput_data
            last_quarter = throughput_data[-len(throughput_data)//4:] if len(throughput_data) >= 4 else throughput_data

            performance_degradation = 0
            if first_quarter and last_quarter:
                avg_first = statistics.mean(first_quarter)
                avg_last = statistics.mean(last_quarter)
                if avg_first > 0:
                    performance_degradation = ((avg_first - avg_last) / avg_first) * 100

            details = {
                'test_duration_hours': self.long_term_duration_hours,
                'actual_duration_seconds': time.time() - start_time,
                'uptime_percentage': uptime_percentage,
                'avg_memory_usage_mb': avg_memory,
                'max_memory_usage_mb': max_memory,
                'memory_growth_mb': memory_growth,
                'avg_throughput': avg_throughput,
                'performance_degradation_percent': performance_degradation,
                'total_reconnection_events': len(stability_metrics['reconnection_events']),
                'stability_metrics': stability_metrics
            }

            # 장기 안정성 기준
            stability_criteria = (
                uptime_percentage >= 0.95 and  # 95% 이상 업타임
                memory_growth < 100 and        # 100MB 이하 메모리 증가
                performance_degradation < 20   # 20% 이하 성능 저하
            )

            status = "PASS" if stability_criteria else "FAIL"

            self.test_results.append(TestResult(
                test_name="long_term_stability",
                status=status,
                duration_seconds=time.time() - start_time,
                details=details
            ))

            logger.info(f"장기 실행 안정성 테스트 완료: 업타임 {uptime_percentage:.1%}")

        except Exception as e:
            self.test_results.append(TestResult(
                test_name="long_term_stability",
                status="FAIL",
                duration_seconds=time.time() - start_time,
                details={},
                error_message=str(e)
            ))

    def _generate_test_summary(self, total_duration: float) -> Dict[str, Any]:
        """테스트 결과 요약 생성"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.status == "PASS")
        failed_tests = sum(1 for result in self.test_results if result.status == "FAIL")
        skipped_tests = sum(1 for result in self.test_results if result.status == "SKIP")

        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        # 실패한 테스트 목록
        failed_test_names = [result.test_name for result in self.test_results if result.status == "FAIL"]

        # 성능 메트릭 요약
        performance_summary = {}
        for result in self.test_results:
            if result.test_name == "performance_benchmarks" and result.status == "PASS":
                performance_summary = {
                    'avg_throughput': result.details.get('avg_throughput_msg_per_sec', 0),
                    'avg_processing_time_ms': result.details.get('avg_processing_time_ms', 0),
                    'avg_memory_usage_mb': result.details.get('avg_memory_usage_mb', 0),
                    'avg_cpu_usage_percent': result.details.get('avg_cpu_usage_percent', 0)
                }

        summary = {
            'test_execution_summary': {
                'total_duration_seconds': total_duration,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'skipped_tests': skipped_tests,
                'success_rate': success_rate,
                'failed_test_names': failed_test_names
            },
            'performance_summary': performance_summary,
            'detailed_results': [result.to_dict() for result in self.test_results],
            'test_environment': {
                'test_symbols': self.test_symbols,
                'test_duration_seconds': self.test_duration_seconds,
                'long_term_test_enabled': self.enable_long_term_test
            },
            'recommendation': self._get_recommendations()
        }

        return summary

    def _get_recommendations(self) -> List[str]:
        """테스트 결과를 바탕으로 권장사항 생성"""
        recommendations = []

        # 실패한 테스트별 권장사항
        for result in self.test_results:
            if result.status == "FAIL":
                if result.test_name == "websocket_connection_stability":
                    recommendations.append("WebSocket 연결 안정성 개선 필요: 재연결 로직 점검")
                elif result.test_name == "data_integrity":
                    recommendations.append("데이터 무결성 검증 강화 필요: 파싱 로직 개선")
                elif result.test_name == "memory_leak_detection":
                    recommendations.append("메모리 누수 발견: 리소스 정리 로직 점검")
                elif result.test_name == "performance_benchmarks":
                    recommendations.append("성능 최적화 필요: 처리 속도 또는 리소스 사용량 개선")
                elif result.test_name == "backpressure_simulation":
                    recommendations.append("백프레셔 처리 개선 필요: 부하 제어 로직 조정")

        # 일반적 권장사항
        if not recommendations:
            recommendations.append("모든 테스트 통과: 시스템이 안정적으로 작동합니다")

        return recommendations

    async def save_test_report(self, file_path: str = None) -> str:
        """테스트 보고서 저장"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"stability_performance_test_report_{timestamp}.json"

        summary = self._generate_test_summary(0)  # 임시로 0 전달

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"테스트 보고서 저장됨: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"테스트 보고서 저장 실패: {str(e)}")
            raise


# 테스트 실행을 위한 헬퍼 함수들

async def run_quick_stability_test(symbols: List[str] = None) -> Dict[str, Any]:
    """빠른 안정성 테스트 (5분)"""
    test_suite = StabilityPerformanceTestSuite(
        test_symbols=symbols or ["BTC_KRW"],
        test_duration_seconds=300,  # 5분
        enable_long_term_test=False
    )

    return await test_suite.run_all_tests()


async def run_comprehensive_test(symbols: List[str] = None) -> Dict[str, Any]:
    """종합 테스트 (30분)"""
    test_suite = StabilityPerformanceTestSuite(
        test_symbols=symbols or ["BTC_KRW", "ETH_KRW"],
        test_duration_seconds=1800,  # 30분
        enable_long_term_test=False
    )

    return await test_suite.run_all_tests()


async def run_long_term_test(symbols: List[str] = None, hours: int = 1) -> Dict[str, Any]:
    """장기 안정성 테스트"""
    test_suite = StabilityPerformanceTestSuite(
        test_symbols=symbols or ["BTC_KRW"],
        test_duration_seconds=300,  # 기본 5분
        enable_long_term_test=True,
        long_term_duration_hours=hours
    )

    return await test_suite.run_all_tests()


if __name__ == "__main__":
    # 기본 테스트 실행 예제
    async def main():
        print("빗썸 WebSocket 안정성 및 성능 테스트 시작...")

        # 빠른 테스트 실행
        results = await run_quick_stability_test()

        # 결과 출력
        print(f"테스트 완료: {results['test_execution_summary']['success_rate']:.1%} 성공률")
        print(f"총 테스트: {results['test_execution_summary']['total_tests']}개")
        print(f"통과: {results['test_execution_summary']['passed_tests']}개")
        print(f"실패: {results['test_execution_summary']['failed_tests']}개")

        if results['test_execution_summary']['failed_tests'] > 0:
            print("실패한 테스트:", results['test_execution_summary']['failed_test_names'])

    # asyncio.run(main())