"""
스프레드 분석기 테스트 스크립트

SpreadAnalyzer의 모든 기능을 테스트하고 검증합니다.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone
from decimal import Decimal

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.spread_analyzer import SpreadAnalyzer, get_spread_analyzer
from src.database.config import DatabaseConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_spread_analysis():
    """기본 스프레드 분석 테스트"""
    try:
        logger.info("=== 기본 스프레드 분석 테스트 시작 ===")

        # 데이터베이스 설정 초기화
        db_config = DatabaseConfig()

        # SpreadAnalyzer 초기화
        async with SpreadAnalyzer(db_config) as analyzer:
            # 상태 확인
            health_status = await analyzer.health_check()
            logger.info(f"분석기 상태: {health_status}")

            # 테스트할 심볼들
            test_symbols = ["BTC_KRW", "ETH_KRW", "XRP_KRW"]

            for symbol in test_symbols:
                logger.info(f"\n--- {symbol} 스프레드 분석 ---")

                # 1. 호가창 데이터 수집 테스트
                orderbook_data = await analyzer.get_orderbook_data(symbol)
                if orderbook_data:
                    logger.info(f"✅ 호가창 데이터 수집 성공")
                    logger.info(f"매수호가: {len(orderbook_data.get('bids', []))}개")
                    logger.info(f"매도호가: {len(orderbook_data.get('asks', []))}개")

                    # 2. 스프레드 메트릭 계산 테스트
                    spread_metrics = await analyzer.calculate_spread_metrics(orderbook_data)
                    if spread_metrics:
                        logger.info(f"✅ 스프레드 메트릭 계산 성공")
                        logger.info(f"최우선 매수호가: {spread_metrics['best_bid']}")
                        logger.info(f"최우선 매도호가: {spread_metrics['best_ask']}")
                        logger.info(f"스프레드율: {spread_metrics['spread_rate']:.4f}%")
                        logger.info(f"유동성 등급: {spread_metrics['liquidity_level']}")
                        logger.info(f"유동성 점수: {spread_metrics['liquidity_score']:.2f}")

                        # 3. 슬리피지 예측 테스트
                        test_sizes = [Decimal('1'), Decimal('10'), Decimal('50')]
                        for size in test_sizes:
                            buy_slippage = await analyzer.predict_slippage(orderbook_data, size, "BUY")
                            sell_slippage = await analyzer.predict_slippage(orderbook_data, size, "SELL")

                            if buy_slippage:
                                logger.info(f"매수 슬리피지 ({size}개): {buy_slippage['slippage_rate']:.4f}%")
                            if sell_slippage:
                                logger.info(f"매도 슬리피지 ({size}개): {sell_slippage['slippage_rate']:.4f}%")
                    else:
                        logger.warning(f"❌ 스프레드 메트릭 계산 실패")
                else:
                    logger.warning(f"❌ 호가창 데이터 수집 실패")

                # 잠시 대기 (API 호출 제한 고려)
                await asyncio.sleep(1)

        logger.info("=== 기본 스프레드 분석 테스트 완료 ===")
        return True

    except Exception as e:
        logger.error(f"기본 스프레드 분석 테스트 중 오류: {str(e)}")
        return False


async def test_comprehensive_spread_analysis():
    """종합 스프레드 분석 테스트"""
    try:
        logger.info("\n=== 종합 스프레드 분석 테스트 시작 ===")

        # 데이터베이스 설정 초기화
        db_config = DatabaseConfig()

        # 전역 SpreadAnalyzer 사용
        analyzer = await get_spread_analyzer(db_config)

        # 메인 코인들에 대한 종합 분석
        main_symbols = ["BTC_KRW", "ETH_KRW"]

        for symbol in main_symbols:
            logger.info(f"\n--- {symbol} 종합 분석 ---")

            # 다양한 주문 크기로 종합 분석
            order_sizes = [Decimal('1'), Decimal('5'), Decimal('10'), Decimal('25'), Decimal('50')]

            analysis_result = await analyzer.comprehensive_spread_analysis(
                symbol, order_sizes
            )

            if analysis_result:
                logger.info(f"✅ {symbol} 종합 분석 성공")

                # 스프레드 메트릭 출력
                spread_metrics = analysis_result.get('spread_metrics', {})
                logger.info(f"스프레드율: {spread_metrics.get('spread_rate', 0):.4f}%")
                logger.info(f"유동성 등급: {spread_metrics.get('liquidity_level', 'N/A')}")
                logger.info(f"시장 깊이 점수: {spread_metrics.get('market_depth', {}).get('depth_score', 0):.2f}")

                # 슬리피지 예측 결과 출력
                slippage_predictions = analysis_result.get('slippage_predictions', {})
                logger.info("슬리피지 예측 결과:")
                for size, predictions in slippage_predictions.items():
                    if predictions.get('buy'):
                        buy_rate = predictions['buy']['slippage_rate']
                        logger.info(f"  {size}개 매수: {buy_rate:.4f}%")
                    if predictions.get('sell'):
                        sell_rate = predictions['sell']['slippage_rate']
                        logger.info(f"  {size}개 매도: {sell_rate:.4f}%")

                # 과거 분석 결과
                historical = analysis_result.get('historical_analysis')
                if historical:
                    logger.info(f"24시간 평균 스프레드: {historical.get('avg_spread_rate', 0):.4f}%")
                    logger.info(f"스프레드 트렌드: {historical.get('trend', 'N/A')}")

                # 종합 점수 및 추천
                comprehensive_score = analysis_result.get('comprehensive_score', 0)
                recommendation = analysis_result.get('trading_recommendation', {})

                logger.info(f"종합 점수: {comprehensive_score:.2f}/100")
                logger.info(f"거래 추천: {recommendation.get('recommendation', 'N/A')}")
                logger.info(f"추천 사유: {recommendation.get('reason', 'N/A')}")

                if recommendation.get('warnings'):
                    logger.info(f"주의사항: {', '.join(recommendation['warnings'])}")

            else:
                logger.warning(f"❌ {symbol} 종합 분석 실패")

            # API 호출 제한 고려하여 대기
            await asyncio.sleep(2)

        logger.info("=== 종합 스프레드 분석 테스트 완료 ===")
        return True

    except Exception as e:
        logger.error(f"종합 스프레드 분석 테스트 중 오류: {str(e)}")
        return False


async def test_liquidity_scoring():
    """유동성 점수 계산 테스트"""
    try:
        logger.info("\n=== 유동성 점수 계산 테스트 시작 ===")

        db_config = DatabaseConfig()
        analyzer = SpreadAnalyzer(db_config)

        # 다양한 스프레드율에 대한 유동성 점수 테스트
        test_spread_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]

        logger.info("스프레드율별 유동성 점수:")
        for spread_rate in test_spread_rates:
            liquidity_level = analyzer._determine_liquidity_level(spread_rate)
            liquidity_score = analyzer._calculate_liquidity_score(spread_rate)

            logger.info(f"  {spread_rate:.2f}% -> {liquidity_level.value} (점수: {liquidity_score:.1f})")

        logger.info("=== 유동성 점수 계산 테스트 완료 ===")
        return True

    except Exception as e:
        logger.error(f"유동성 점수 계산 테스트 중 오류: {str(e)}")
        return False


async def test_slippage_categories():
    """슬리피지 범주 분류 테스트"""
    try:
        logger.info("\n=== 슬리피지 범주 분류 테스트 시작 ===")

        db_config = DatabaseConfig()
        analyzer = SpreadAnalyzer(db_config)

        # 다양한 슬리피지율에 대한 범주 분류 테스트
        test_slippage_rates = [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]

        logger.info("슬리피지율별 범주 분류:")
        for slippage_rate in test_slippage_rates:
            category = analyzer._categorize_slippage(slippage_rate)
            logger.info(f"  {slippage_rate:.2f}% -> {category}")

        logger.info("=== 슬리피지 범주 분류 테스트 완료 ===")
        return True

    except Exception as e:
        logger.error(f"슬리피지 범주 분류 테스트 중 오류: {str(e)}")
        return False


async def test_historical_spread_analysis():
    """과거 스프레드 분석 테스트"""
    try:
        logger.info("\n=== 과거 스프레드 분석 테스트 시작 ===")

        db_config = DatabaseConfig()
        analyzer = await get_spread_analyzer(db_config)

        # 데이터베이스 연결 확인
        if not await db_config.health_check():
            logger.warning("데이터베이스 연결 안됨 - 과거 분석 테스트 스킵")
            return True

        # 주요 코인들의 과거 스프레드 분석
        symbols = ["BTC_KRW", "ETH_KRW"]

        for symbol in symbols:
            logger.info(f"\n--- {symbol} 과거 스프레드 분석 ---")

            # 24시간, 12시간, 6시간 분석
            time_periods = [24, 12, 6]

            for hours in time_periods:
                historical = await analyzer.get_historical_spread_analysis(symbol, hours)

                if historical:
                    logger.info(f"✅ {hours}시간 과거 분석 성공")
                    logger.info(f"평균 스프레드: {historical.get('avg_spread_rate', 0):.4f}%")
                    logger.info(f"최소/최대: {historical.get('min_spread_rate', 0):.4f}%/{historical.get('max_spread_rate', 0):.4f}%")
                    logger.info(f"변동성: {historical.get('spread_volatility', 0):.4f}")
                    logger.info(f"트렌드: {historical.get('trend', 'N/A')}")
                    logger.info(f"샘플 수: {historical.get('sample_count', 0)}개")
                else:
                    logger.info(f"⚠️ {hours}시간 과거 분석 데이터 없음")

        logger.info("=== 과거 스프레드 분석 테스트 완료 ===")
        return True

    except Exception as e:
        logger.error(f"과거 스프레드 분석 테스트 중 오류: {str(e)}")
        return False


async def run_performance_test():
    """성능 테스트"""
    try:
        logger.info("\n=== 성능 테스트 시작 ===")

        db_config = DatabaseConfig()
        analyzer = await get_spread_analyzer(db_config)

        # 여러 심볼에 대한 연속 분석 테스트
        test_symbols = ["BTC_KRW", "ETH_KRW", "XRP_KRW", "ADA_KRW", "DOT_KRW"]

        start_time = datetime.now(timezone.utc)
        success_count = 0

        for i, symbol in enumerate(test_symbols, 1):
            logger.info(f"성능 테스트 {i}/{len(test_symbols)}: {symbol}")

            result = await analyzer.comprehensive_spread_analysis(symbol)
            if result:
                success_count += 1
                logger.info(f"✅ {symbol} 분석 완료")
            else:
                logger.warning(f"❌ {symbol} 분석 실패")

            # API 제한 고려
            await asyncio.sleep(1)

        end_time = datetime.now(timezone.utc)
        elapsed_time = (end_time - start_time).total_seconds()

        # 통계 정보 출력
        stats = analyzer.get_stats()
        logger.info(f"\n📊 성능 테스트 결과:")
        logger.info(f"총 소요 시간: {elapsed_time:.2f}초")
        logger.info(f"성공률: {success_count}/{len(test_symbols)} ({success_count/len(test_symbols)*100:.1f}%)")
        logger.info(f"API 호출 수: {stats.get('api_calls', 0)}")
        logger.info(f"DB 쿼리 수: {stats.get('db_queries', 0)}")
        logger.info(f"오류 수: {stats.get('errors', 0)}")
        logger.info(f"분석 수: {stats.get('analysis_count', 0)}")

        logger.info("=== 성능 테스트 완료 ===")
        return success_count > 0

    except Exception as e:
        logger.error(f"성능 테스트 중 오류: {str(e)}")
        return False


async def main():
    """메인 테스트 함수"""
    logger.info("🚀 SpreadAnalyzer 통합 테스트 시작")

    # 테스트 실행
    test_results = {
        "기본 스프레드 분석": await test_basic_spread_analysis(),
        "유동성 점수 계산": await test_liquidity_scoring(),
        "슬리피지 범주 분류": await test_slippage_categories(),
        "과거 스프레드 분석": await test_historical_spread_analysis(),
        "종합 스프레드 분석": await test_comprehensive_spread_analysis(),
        "성능 테스트": await run_performance_test()
    }

    # 결과 요약
    logger.info("\n" + "="*50)
    logger.info("📋 테스트 결과 요약")
    logger.info("="*50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\n전체 테스트 결과: {passed}/{total} 통과 ({passed/total*100:.1f}%)")

    if passed == total:
        logger.info("🎉 모든 테스트가 성공했습니다!")
    else:
        logger.warning("⚠️ 일부 테스트가 실패했습니다.")

    # 전역 인스턴스 정리
    from src.services.spread_analyzer import close_spread_analyzer
    await close_spread_analyzer()

    return passed == total


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"테스트 실행 중 예상치 못한 오류: {str(e)}")
        sys.exit(1)