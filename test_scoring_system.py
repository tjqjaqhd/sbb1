#!/usr/bin/env python3
"""
가중 점수 시스템 종합 테스트 스크립트

ScoringSystem의 모든 기능을 테스트하고 결과를 검증하는 스크립트입니다.
실제 빗썸 API 데이터를 사용하여 종합 점수 계산을 테스트합니다.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Dict, Any, List

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath('.'))

from src.database.config import DatabaseConfig
from src.services.scoring_system import ScoringSystem, get_scoring_system, close_scoring_system

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scoring_system_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class ScoringSystemTester:
    """가중 점수 시스템 테스터"""

    # 테스트할 심볼들
    TEST_SYMBOLS = [
        "BTC_KRW",  # 비트코인 - 높은 유동성
        "ETH_KRW",  # 이더리움 - 높은 유동성
        "ADA_KRW",  # 에이다 - 중간 유동성
        "DOGE_KRW", # 도지코인 - 변동성 높음
        "XRP_KRW"   # 리플 - 안정적인 거래량
    ]

    def __init__(self):
        self.db_config = None
        self.scoring_system = None
        self.test_results = []

    async def setup(self):
        """테스트 환경 설정"""
        try:
            logger.info("가중 점수 시스템 테스트 환경 설정 시작")

            # 데이터베이스 설정
            self.db_config = DatabaseConfig()
            await self.db_config.initialize()

            # 데이터베이스 연결 확인
            if not await self.db_config.health_check():
                raise Exception("데이터베이스 연결 실패")

            # 가중 점수 시스템 초기화
            self.scoring_system = await get_scoring_system(self.db_config)

            logger.info("테스트 환경 설정 완료")

        except Exception as e:
            logger.error(f"테스트 환경 설정 실패: {str(e)}")
            raise

    async def cleanup(self):
        """테스트 환경 정리"""
        try:
            logger.info("테스트 환경 정리 중")

            if self.scoring_system:
                await close_scoring_system()

            if self.db_config:
                await self.db_config.close()

            logger.info("테스트 환경 정리 완료")

        except Exception as e:
            logger.error(f"테스트 환경 정리 중 오류: {str(e)}")

    async def test_health_check(self) -> bool:
        """시스템 상태 확인 테스트"""
        try:
            logger.info("=== 시스템 상태 확인 테스트 ===")

            health_status = await self.scoring_system.health_check()

            logger.info(f"서비스명: {health_status.get('service_name')}")
            logger.info(f"데이터베이스 연결: {health_status.get('database_connected')}")
            logger.info(f"모든 서비스 정상: {health_status.get('all_services_healthy')}")

            # 각 분석 서비스 상태
            analyzer_services = health_status.get('analyzer_services', {})
            for service_name, status in analyzer_services.items():
                logger.info(f"  {service_name}: {'정상' if status else '비정상'}")

            # 통계 정보
            stats = health_status.get('stats', {})
            logger.info(f"통계 정보: {stats}")

            success = health_status.get('all_services_healthy', False)
            logger.info(f"상태 확인 테스트: {'성공' if success else '실패'}")

            return success

        except Exception as e:
            logger.error(f"상태 확인 테스트 실패: {str(e)}")
            return False

    async def test_basic_scoring(self, symbol: str) -> Dict[str, Any]:
        """기본 점수 계산 테스트"""
        try:
            logger.info(f"=== {symbol} 기본 점수 계산 테스트 ===")

            # 기본 가중치로 종합 점수 계산
            result = await self.scoring_system.calculate_comprehensive_score(symbol)

            if result:
                logger.info(f"종합 점수: {result['comprehensive_score']:.2f}")
                logger.info(f"신뢰도: {result['is_reliable']}")
                logger.info(f"권고: {result['recommendation']['action']}")

                # 개별 점수들
                individual_scores = result.get('individual_scores', {})
                for indicator, score in individual_scores.items():
                    logger.info(f"  {indicator}: {score:.2f}")

                # 사용된 가중치
                weights_used = result.get('weights_used', {})
                for indicator, weight in weights_used.items():
                    logger.info(f"  {indicator} 가중치: {weight:.3f}")

                # 품질 평가
                quality = result.get('quality_assessment', {})
                logger.info(f"품질 점수: {quality.get('overall_quality', 0):.2f}")
                logger.info(f"지표 커버리지: {quality.get('coverage_score', 0):.2f}")

                return {
                    'symbol': symbol,
                    'success': True,
                    'score': result['comprehensive_score'],
                    'reliability': result['is_reliable'],
                    'quality': quality.get('overall_quality', 0),
                    'coverage': quality.get('coverage_score', 0)
                }
            else:
                logger.error(f"{symbol} 기본 점수 계산 실패")
                return {'symbol': symbol, 'success': False}

        except Exception as e:
            logger.error(f"{symbol} 기본 점수 계산 테스트 실패: {str(e)}")
            return {'symbol': symbol, 'success': False, 'error': str(e)}

    async def test_custom_weights(self, symbol: str) -> Dict[str, Any]:
        """사용자 정의 가중치 테스트"""
        try:
            logger.info(f"=== {symbol} 사용자 정의 가중치 테스트 ===")

            # 거래량 중심 가중치
            volume_focused_weights = {
                'volume': 0.50,    # 거래량 50%
                'atr': 0.20,       # ATR 20%
                'rsi': 0.15,       # RSI 15%
                'bollinger': 0.10, # 볼린저밴드 10%
                'spread': 0.05     # 스프레드 5%
            }

            result = await self.scoring_system.calculate_comprehensive_score(
                symbol,
                custom_weights=volume_focused_weights,
                enable_dynamic_weights=False  # 동적 가중치 비활성화
            )

            if result:
                logger.info(f"거래량 중심 종합 점수: {result['comprehensive_score']:.2f}")

                # 사용된 가중치 확인
                weights_used = result.get('weights_used', {})
                logger.info("사용된 가중치:")
                for indicator, weight in weights_used.items():
                    logger.info(f"  {indicator}: {weight:.3f}")

                return {
                    'symbol': symbol,
                    'success': True,
                    'score': result['comprehensive_score'],
                    'weights_correct': abs(weights_used.get('volume', 0) - 0.50) < 0.01
                }
            else:
                logger.error(f"{symbol} 사용자 정의 가중치 테스트 실패")
                return {'symbol': symbol, 'success': False}

        except Exception as e:
            logger.error(f"{symbol} 사용자 정의 가중치 테스트 실패: {str(e)}")
            return {'symbol': symbol, 'success': False, 'error': str(e)}

    async def test_dynamic_weights(self, symbol: str) -> Dict[str, Any]:
        """동적 가중치 조정 테스트"""
        try:
            logger.info(f"=== {symbol} 동적 가중치 조정 테스트 ===")

            # 동적 가중치 활성화
            result = await self.scoring_system.calculate_comprehensive_score(
                symbol,
                enable_dynamic_weights=True
            )

            if result:
                logger.info(f"동적 조정 종합 점수: {result['comprehensive_score']:.2f}")

                # 사용된 가중치
                weights_used = result.get('weights_used', {})
                logger.info("동적 조정된 가중치:")
                for indicator, weight in weights_used.items():
                    logger.info(f"  {indicator}: {weight:.3f}")

                # 분석 결과에서 시장 상황 파악
                analysis_results = result.get('analysis_results', {})
                atr_data = analysis_results.get('atr', {})
                spread_data = analysis_results.get('spread', {})

                market_condition = "일반"
                if atr_data:
                    atr_percentage = atr_data.get('atr_percentage', 0)
                    if atr_percentage > 8.0:
                        market_condition = "고변동성"
                    elif atr_percentage < 3.0:
                        market_condition = "저변동성"

                if spread_data:
                    current_analysis = spread_data.get('current_analysis', {})
                    spread_rate = current_analysis.get('spread_rate', 0)
                    if spread_rate > 0.005:
                        market_condition = "저유동성"

                logger.info(f"감지된 시장 상황: {market_condition}")

                return {
                    'symbol': symbol,
                    'success': True,
                    'score': result['comprehensive_score'],
                    'market_condition': market_condition,
                    'weights_adjusted': weights_used != self.scoring_system.DEFAULT_WEIGHTS
                }
            else:
                logger.error(f"{symbol} 동적 가중치 테스트 실패")
                return {'symbol': symbol, 'success': False}

        except Exception as e:
            logger.error(f"{symbol} 동적 가중치 테스트 실패: {str(e)}")
            return {'symbol': symbol, 'success': False, 'error': str(e)}

    async def test_outlier_filtering(self, symbol: str) -> Dict[str, Any]:
        """이상치 필터링 테스트"""
        try:
            logger.info(f"=== {symbol} 이상치 필터링 테스트 ===")

            # 여러 번 계산하여 점수 이력 생성
            scores = []
            for i in range(5):
                result = await self.scoring_system.calculate_comprehensive_score(
                    symbol,
                    filter_outliers=False  # 첫 번째는 필터링 없이
                )
                if result:
                    scores.append(result['comprehensive_score'])
                await asyncio.sleep(1)  # 1초 간격

            if len(scores) >= 3:
                # 이상치 필터링 활성화하여 재계산
                filtered_result = await self.scoring_system.calculate_comprehensive_score(
                    symbol,
                    filter_outliers=True
                )

                if filtered_result:
                    filtered_score = filtered_result['comprehensive_score']
                    logger.info(f"점수 이력: {scores}")
                    logger.info(f"필터링된 점수: {filtered_score:.2f}")

                    # 통계 정보
                    stats = self.scoring_system.get_stats()
                    logger.info(f"필터링된 이상치 수: {stats.get('filtered_outliers', 0)}")

                    return {
                        'symbol': symbol,
                        'success': True,
                        'score_history': scores,
                        'filtered_score': filtered_score,
                        'outliers_filtered': stats.get('filtered_outliers', 0)
                    }

            logger.warning(f"{symbol} 이상치 필터링 테스트를 위한 데이터 부족")
            return {'symbol': symbol, 'success': False, 'reason': 'insufficient_data'}

        except Exception as e:
            logger.error(f"{symbol} 이상치 필터링 테스트 실패: {str(e)}")
            return {'symbol': symbol, 'success': False, 'error': str(e)}

    async def test_performance_benchmark(self) -> Dict[str, Any]:
        """성능 벤치마크 테스트"""
        try:
            logger.info("=== 성능 벤치마크 테스트 ===")

            start_time = datetime.now()

            # 여러 심볼 동시 계산
            tasks = []
            for symbol in self.TEST_SYMBOLS[:3]:  # 처음 3개만 테스트
                task = self.scoring_system.calculate_comprehensive_score(symbol)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            successful_results = [r for r in results if not isinstance(r, Exception) and r is not None]

            logger.info(f"동시 계산 시간: {duration:.2f}초")
            logger.info(f"성공한 계산: {len(successful_results)}/{len(tasks)}")
            logger.info(f"평균 처리 시간: {duration/len(tasks):.2f}초/심볼")

            # 시스템 통계
            stats = self.scoring_system.get_stats()
            logger.info(f"총 계산 횟수: {stats.get('total_calculations', 0)}")
            logger.info(f"성공한 계산: {stats.get('successful_calculations', 0)}")
            logger.info(f"오류 수: {stats.get('errors', 0)}")

            return {
                'success': True,
                'duration': duration,
                'successful_calculations': len(successful_results),
                'total_attempts': len(tasks),
                'avg_time_per_symbol': duration / len(tasks),
                'stats': stats
            }

        except Exception as e:
            logger.error(f"성능 벤치마크 테스트 실패: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def run_comprehensive_test(self):
        """종합 테스트 실행"""
        try:
            logger.info("===== 가중 점수 시스템 종합 테스트 시작 =====")

            await self.setup()

            # 1. 시스템 상태 확인
            health_result = await self.test_health_check()
            self.test_results.append(('health_check', health_result))

            if not health_result:
                logger.error("시스템 상태가 불량하여 테스트를 중단합니다")
                return

            # 2. 각 심볼별 기본 테스트
            for symbol in self.TEST_SYMBOLS:
                logger.info(f"\n--- {symbol} 테스트 시작 ---")

                # 기본 점수 계산 테스트
                basic_result = await self.test_basic_scoring(symbol)
                self.test_results.append(('basic_scoring', basic_result))

                if basic_result.get('success'):
                    # 사용자 정의 가중치 테스트
                    custom_result = await self.test_custom_weights(symbol)
                    self.test_results.append(('custom_weights', custom_result))

                    # 동적 가중치 테스트
                    dynamic_result = await self.test_dynamic_weights(symbol)
                    self.test_results.append(('dynamic_weights', dynamic_result))

                    # 이상치 필터링 테스트 (첫 번째 심볼만)
                    if symbol == self.TEST_SYMBOLS[0]:
                        outlier_result = await self.test_outlier_filtering(symbol)
                        self.test_results.append(('outlier_filtering', outlier_result))

                await asyncio.sleep(2)  # API 호출 간격

            # 3. 성능 벤치마크
            performance_result = await self.test_performance_benchmark()
            self.test_results.append(('performance', performance_result))

            # 4. 결과 요약
            await self.print_test_summary()

        except Exception as e:
            logger.error(f"종합 테스트 실행 중 오류: {str(e)}")
        finally:
            await self.cleanup()

    async def print_test_summary(self):
        """테스트 결과 요약 출력"""
        try:
            logger.info("\n===== 테스트 결과 요약 =====")

            # 테스트 분류별 결과 집계
            test_categories = {}
            for test_type, result in self.test_results:
                if test_type not in test_categories:
                    test_categories[test_type] = {'total': 0, 'success': 0, 'results': []}

                test_categories[test_type]['total'] += 1
                if result.get('success', False):
                    test_categories[test_type]['success'] += 1
                test_categories[test_type]['results'].append(result)

            # 분류별 결과 출력
            for category, data in test_categories.items():
                success_rate = (data['success'] / data['total']) * 100
                logger.info(f"{category}: {data['success']}/{data['total']} ({success_rate:.1f}%)")

            # 전체 성공률
            total_tests = len(self.test_results)
            successful_tests = sum(1 for _, result in self.test_results if result.get('success', False))
            overall_success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0

            logger.info(f"\n전체 테스트: {successful_tests}/{total_tests} ({overall_success_rate:.1f}%)")

            # 주요 지표 요약
            basic_scores = [r.get('score', 0) for _, r in self.test_results
                           if _[0] == 'basic_scoring' and r.get('success')]
            if basic_scores:
                avg_score = sum(basic_scores) / len(basic_scores)
                logger.info(f"평균 종합 점수: {avg_score:.2f}")

            # 성능 정보
            performance_data = next((r for t, r in self.test_results if t == 'performance'), {})
            if performance_data.get('success'):
                logger.info(f"평균 처리 시간: {performance_data.get('avg_time_per_symbol', 0):.2f}초/심볼")

            # 최종 판정
            if overall_success_rate >= 80:
                logger.info("결론: 가중 점수 시스템이 정상적으로 작동합니다 ✓")
            elif overall_success_rate >= 60:
                logger.info("결론: 가중 점수 시스템이 대체로 정상 작동하나 일부 개선이 필요합니다")
            else:
                logger.info("결론: 가중 점수 시스템에 문제가 있어 점검이 필요합니다")

        except Exception as e:
            logger.error(f"테스트 결과 요약 생성 중 오류: {str(e)}")


async def main():
    """메인 실행 함수"""
    try:
        tester = ScoringSystemTester()
        await tester.run_comprehensive_test()

    except KeyboardInterrupt:
        logger.info("사용자에 의해 테스트가 중단되었습니다")
    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    # Windows에서 ProactorEventLoop 사용
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    exit_code = asyncio.run(main())
    sys.exit(exit_code)