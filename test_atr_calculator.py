"""
ATR Calculator 테스트 스크립트

ATRCalculator의 기능을 테스트하고 검증하는 스크립트입니다.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone

# 프로젝트 경로 설정
sys.path.append('.')

from src.services.atr_calculator import get_atr_calculator, close_atr_calculator
from src.database.config import DatabaseConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_atr_calculator():
    """ATR Calculator 종합 테스트"""
    print("=" * 60)
    print("ATR Calculator 테스트 시작")
    print("=" * 60)

    # 데이터베이스 설정
    db_config = DatabaseConfig()

    # ATR Calculator 인스턴스 생성
    atr_calculator = await get_atr_calculator(db_config)

    try:
        # 1. 서비스 상태 확인
        print("\n1. 서비스 상태 확인")
        print("-" * 40)
        health_status = await atr_calculator.health_check()
        print(f"서비스 상태: {health_status}")

        if not health_status.get('http_client_available', False):
            print("경고: HTTP 클라이언트가 비활성화 상태입니다.")

        # 2. 테스트 심볼 목록
        test_symbols = ['BTC_KRW', 'ETH_KRW', 'XRP_KRW']

        for symbol in test_symbols:
            print(f"\n2. {symbol} ATR 분석 테스트")
            print("-" * 40)

            try:
                # 종합 ATR 분석 실행
                analysis_result = await atr_calculator.comprehensive_atr_analysis(symbol)

                if analysis_result:
                    print(f"분석 완료: {symbol}")
                    print(f"  - 현재 가격: {analysis_result.get('current_price', 'N/A')}")
                    print(f"  - ATR 값: {analysis_result.get('atr_value', 'N/A'):.6f}")
                    print(f"  - ATR 퍼센티지: {analysis_result.get('atr_percentage', 'N/A'):.2f}%")
                    print(f"  - 변동성 점수: {analysis_result.get('volatility_score', 'N/A'):.2f}/10")

                    # 데이트레이딩 적합성 정보
                    suitability = analysis_result.get('suitability_analysis', {})
                    print(f"  - 변동성 수준: {suitability.get('volatility_level', 'N/A')}")
                    print(f"  - 적합성 점수: {suitability.get('suitability_score', 'N/A'):.1f}/10")
                    print(f"  - 데이트레이딩 적합: {'예' if suitability.get('is_suitable', False) else '아니오'}")
                    print(f"  - 위험도: {suitability.get('risk_level', 'N/A')}")
                    print(f"  - 권장사항: {suitability.get('recommendation', 'N/A')}")

                    # 분석 품질 정보
                    quality = analysis_result.get('analysis_quality', {})
                    print(f"  - 분석 품질: {quality.get('grade', 'N/A')} ({quality.get('score', 0):.1f}/10)")
                    print(f"  - 사용 데이터: {quality.get('data_points', 'N/A')}개")

                    # 추가 통계
                    stats = analysis_result.get('additional_stats', {})
                    if stats:
                        print(f"  - 평균 일일 변동: {stats.get('avg_daily_change_pct', 0):.2f}%")
                        print(f"  - 최대 일일 변동: {stats.get('max_daily_change_pct', 0):.2f}%")
                else:
                    print(f"분석 실패: {symbol}")

            except Exception as e:
                print(f"오류 발생 ({symbol}): {str(e)}")
                logger.error(f"ATR 분석 테스트 오류 ({symbol}): {str(e)}")

        # 3. 통계 정보 출력
        print(f"\n3. 서비스 통계")
        print("-" * 40)
        stats = atr_calculator.get_stats()
        print(f"ATR 계산 횟수: {stats.get('atr_calculations', 0)}")
        print(f"API 호출 횟수: {stats.get('api_calls', 0)}")
        print(f"DB 쿼리 횟수: {stats.get('db_queries', 0)}")
        print(f"오류 발생 횟수: {stats.get('errors', 0)}")
        print(f"마지막 계산: {stats.get('last_calculation', '없음')}")

        # 4. 개별 기능 테스트
        print(f"\n4. 개별 기능 테스트")
        print("-" * 40)

        # 변동성 점수 정규화 테스트
        test_atr_percentages = [0.01, 0.03, 0.05, 0.08, 0.12, 0.20]
        print("ATR 퍼센티지별 변동성 점수:")
        for atr_pct in test_atr_percentages:
            score = atr_calculator.normalize_volatility_score(atr_pct)
            print(f"  {atr_pct*100:5.1f}% -> {score:4.1f}/10")

        # 데이트레이딩 적합성 평가 테스트
        print("\nATR 퍼센티지별 데이트레이딩 적합성:")
        for atr_pct in test_atr_percentages:
            evaluation = atr_calculator.evaluate_daytrading_suitability(atr_pct)
            print(f"  {atr_pct*100:5.1f}% -> {evaluation['volatility_level']:10s} (적합: {'예' if evaluation['is_suitable'] else '아니오'})")

        print(f"\n테스트 완료!")

    except Exception as e:
        logger.error(f"테스트 실행 중 오류: {str(e)}")
        print(f"테스트 실행 중 오류: {str(e)}")

    finally:
        # 리소스 정리
        await close_atr_calculator()
        await db_config.close()


async def test_simple_atr():
    """간단한 ATR 계산 테스트 (가상 데이터 사용)"""
    print("\n" + "=" * 60)
    print("간단한 ATR 계산 테스트 (가상 데이터)")
    print("=" * 60)

    db_config = DatabaseConfig()
    atr_calculator = await get_atr_calculator(db_config)

    try:
        # 가상 OHLCV 데이터 생성
        import numpy as np

        # 30일간 가상 가격 데이터 생성
        days = 30
        base_price = 50000000  # 5천만원 (BTC 가격 기준)

        historical_data = []
        current_price = base_price

        for i in range(days):
            # 일일 변동률 (-3% ~ +3%)
            daily_change = np.random.uniform(-0.03, 0.03)
            new_price = current_price * (1 + daily_change)

            # OHLC 생성
            high = new_price * (1 + abs(np.random.normal(0, 0.01)))
            low = new_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = low + (high - low) * np.random.random()
            close = new_price
            volume = np.random.uniform(1000, 5000)

            historical_data.append({
                'timestamp': datetime.now(timezone.utc),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

            current_price = new_price

        print(f"가상 데이터 생성 완료: {len(historical_data)}일치")
        print(f"시작 가격: {historical_data[0]['close']:,.0f}원")
        print(f"종료 가격: {historical_data[-1]['close']:,.0f}원")

        # ATR 계산
        atr_value = atr_calculator.calculate_atr(historical_data, 14)
        if atr_value:
            print(f"ATR 값: {atr_value:,.0f}원")

            # ATR 퍼센티지 계산
            atr_percentage = atr_calculator.calculate_atr_percentage(atr_value, current_price)
            if atr_percentage:
                print(f"ATR 퍼센티지: {atr_percentage * 100:.2f}%")

                # 변동성 점수
                volatility_score = atr_calculator.normalize_volatility_score(atr_percentage)
                print(f"변동성 점수: {volatility_score:.2f}/10")

                # 데이트레이딩 적합성
                suitability = atr_calculator.evaluate_daytrading_suitability(atr_percentage)
                print(f"데이트레이딩 적합성: {suitability['volatility_level']} ({'적합' if suitability['is_suitable'] else '부적합'})")
                print(f"권장사항: {suitability['recommendation']}")
        else:
            print("ATR 계산 실패")

    except Exception as e:
        logger.error(f"간단한 ATR 테스트 중 오류: {str(e)}")
        print(f"테스트 실행 중 오류: {str(e)}")

    finally:
        await close_atr_calculator()
        await db_config.close()


if __name__ == "__main__":
    try:
        # 전체 테스트 실행
        asyncio.run(test_atr_calculator())

        # 간단한 ATR 계산 테스트
        asyncio.run(test_simple_atr())

    except KeyboardInterrupt:
        print("\n테스트 중단됨")
    except Exception as e:
        logger.error(f"메인 테스트 실행 오류: {str(e)}")
        print(f"테스트 실행 오류: {str(e)}")