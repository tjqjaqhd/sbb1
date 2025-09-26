#!/usr/bin/env python3
"""
거래량 분석기 테스트 스크립트

VolumeAnalyzer 클래스의 기능을 테스트합니다.
"""

import asyncio
import logging
from decimal import Decimal
from src.services.volume_analyzer import VolumeAnalyzer, get_volume_analyzer
from src.database.config import DatabaseConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)


async def test_volume_analyzer():
    """거래량 분석기 종합 테스트"""
    print("=" * 60)
    print("거래량 분석기 테스트 시작")
    print("=" * 60)

    # 데이터베이스 설정
    db_config = DatabaseConfig()

    # VolumeAnalyzer 생성
    async with VolumeAnalyzer(db_config) as analyzer:
        print("\n1. 거래량 분석기 상태 확인...")
        health_status = await analyzer.health_check()
        print(f"상태 확인 결과: {health_status}")

        test_symbols = ["BTC_KRW", "ETH_KRW"]

        for symbol in test_symbols:
            print(f"\n{'=' * 40}")
            print(f"{symbol} 분석 테스트")
            print(f"{'=' * 40}")

            # 2. 24시간 거래량 데이터 수집 테스트
            print(f"\n2. {symbol} 24시간 거래량 데이터 수집...")
            volume_data = await analyzer.get_24h_volume_data(symbol)
            if volume_data:
                print(f"✓ 24시간 거래량: {volume_data.get('volume_24h')}")
                print(f"✓ 거래금액: {volume_data.get('volume_value_24h')}")
                print(f"✓ 현재가: {volume_data.get('closing_price')}")
            else:
                print(f"✗ {symbol} 24시간 거래량 데이터 수집 실패")
                continue

            # 3. 과거 평균 거래량 계산 테스트
            print(f"\n3. {symbol} 과거 평균 거래량 계산...")
            avg_7d = await analyzer.get_historical_volume_avg(symbol, 7)
            avg_30d = await analyzer.get_historical_volume_avg(symbol, 30)
            print(f"✓ 7일 평균 거래량: {avg_7d}")
            print(f"✓ 30일 평균 거래량: {avg_30d}")

            # 4. 거래량 급등 점수 계산 테스트
            if volume_data and volume_data.get('volume_24h'):
                print(f"\n4. {symbol} 거래량 급등 점수 계산...")
                surge_score = await analyzer.calculate_volume_surge_score(
                    symbol, volume_data['volume_24h']
                )
                print(f"✓ 급등 점수 (0-10): {surge_score}")
            else:
                print(f"\n4. {symbol} 거래량 데이터 부족으로 급등 점수 계산 건너뛰기")

            # 5. 시간대별 거래량 패턴 분석 테스트
            print(f"\n5. {symbol} 시간대별 거래량 패턴 분석...")
            pattern_analysis = await analyzer.analyze_hourly_volume_pattern(symbol)
            if pattern_analysis:
                print(f"✓ 총 거래량: {pattern_analysis.get('total_volume')}")
                print(f"✓ 시간당 평균 거래량: {pattern_analysis.get('avg_hourly_volume')}")
                print(f"✓ 정규화 점수: {pattern_analysis.get('normalized_score'):.2f}")
                print(f"✓ 피크 시간대 수: {len(pattern_analysis.get('peak_hours', []))}")
            else:
                print(f"✗ {symbol} 시간대별 패턴 분석 실패")

            # 6. 종합 거래량 분석 테스트
            print(f"\n6. {symbol} 종합 거래량 분석...")
            comprehensive_analysis = await analyzer.comprehensive_volume_analysis(symbol)
            if comprehensive_analysis:
                print(f"✓ 현재 24시간 거래량: {comprehensive_analysis.get('current_volume_24h')}")
                print(f"✓ 7일 평균 대비 비율: {comprehensive_analysis.get('volume_ratios', {}).get('vs_7d_avg')}")
                print(f"✓ 30일 평균 대비 비율: {comprehensive_analysis.get('volume_ratios', {}).get('vs_30d_avg')}")
                print(f"✓ 급등 점수: {comprehensive_analysis.get('surge_score')}")
                print(f"✓ 종합 점수: {comprehensive_analysis.get('comprehensive_score'):.2f}")
            else:
                print(f"✗ {symbol} 종합 분석 실패")

        # 7. 통계 정보 확인
        print(f"\n{'=' * 40}")
        print("거래량 분석기 통계 정보")
        print(f"{'=' * 40}")
        stats = analyzer.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")

    print(f"\n{'=' * 60}")
    print("거래량 분석기 테스트 완료")
    print(f"{'=' * 60}")


async def test_api_only():
    """API 호출만 테스트 (DB 없이)"""
    print("\n" + "=" * 40)
    print("API 전용 테스트 (DB 연결 없음)")
    print("=" * 40)

    db_config = DatabaseConfig()
    analyzer = VolumeAnalyzer(db_config)

    # 24시간 거래량 데이터만 테스트
    symbols = ["BTC_KRW", "ETH_KRW", "XRP_KRW"]
    for symbol in symbols:
        print(f"\n{symbol} 24시간 거래량 데이터 수집 테스트...")
        volume_data = await analyzer.get_24h_volume_data(symbol)
        if volume_data:
            print(f"✓ 거래량: {volume_data.get('volume_24h')}")
            print(f"✓ 현재가: {volume_data.get('closing_price')}")
            print(f"✓ 24시간 변동률: {volume_data.get('fluctate_rate_24h')}%")
        else:
            print(f"✗ {symbol} 데이터 수집 실패")

    await analyzer.__aexit__(None, None, None)


if __name__ == "__main__":
    print("거래량 분석기 테스트를 시작합니다...")

    try:
        # 먼저 API만 테스트
        asyncio.run(test_api_only())

        # 그다음 전체 기능 테스트 (DB 연결 필요)
        print("\n데이터베이스 연결이 필요한 전체 테스트를 진행합니다...")
        print("(DB 연결 오류가 발생하면 API 테스트 결과만 확인하세요)")
        asyncio.run(test_volume_analyzer())

    except KeyboardInterrupt:
        print("\n테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {str(e)}")
        logging.exception("테스트 실행 중 예외 발생")