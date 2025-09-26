#!/usr/bin/env python3
"""
거래량 분석기 간단한 테스트 스크립트

API 호출 기능만 테스트합니다.
"""

import asyncio
import logging
from src.services.volume_analyzer import VolumeAnalyzer
from src.database.config import DatabaseConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def simple_api_test():
    """간단한 API 테스트"""
    print("=" * 50)
    print("거래량 분석기 간단 테스트")
    print("=" * 50)

    # 데이터베이스 설정 (연결은 하지 않음)
    db_config = DatabaseConfig()

    # VolumeAnalyzer 생성
    analyzer = VolumeAnalyzer(db_config)

    try:
        symbols = ["BTC_KRW", "ETH_KRW", "XRP_KRW"]

        for symbol in symbols:
            print(f"\n[{symbol}] 24시간 거래량 데이터 수집 테스트...")

            # 1. 24시간 거래량 데이터 수집
            volume_data = await analyzer.get_24h_volume_data(symbol)

            if volume_data:
                print(f"  ✓ 현재 거래량: {volume_data.get('volume_24h')}")
                print(f"  ✓ 거래 금액: {volume_data.get('volume_value_24h')}")
                print(f"  ✓ 현재가: {volume_data.get('closing_price'):,} 원")
                print(f"  ✓ 24시간 변동: {volume_data.get('fluctate_24h')}")
                print(f"  ✓ 24시간 변동률: {volume_data.get('fluctate_rate_24h')}%")

                # 거래량을 KRW로 환산한 금액 확인
                if volume_data.get('volume_value_24h'):
                    volume_krw = int(float(volume_data['volume_value_24h']))
                    print(f"  ✓ 거래 금액: {volume_krw:,} 원")

                # 간단한 점수 계산 (DB 없이)
                if volume_data.get('volume_24h'):
                    # 간단한 기준: 거래량이 높을수록 높은 점수
                    volume_float = float(volume_data['volume_24h'])

                    # 심볼별 기준값 설정
                    base_volumes = {
                        'BTC_KRW': 500,    # BTC 기준 거래량
                        'ETH_KRW': 20000,  # ETH 기준 거래량
                        'XRP_KRW': 30000000  # XRP 기준 거래량
                    }

                    base_volume = base_volumes.get(symbol, 1000)
                    volume_score = min(100, (volume_float / base_volume) * 100)

                    print(f"  ✓ 거래량 점수: {volume_score:.1f}/100")
            else:
                print(f"  ✗ {symbol} 데이터 수집 실패")

        # 통계 확인
        print(f"\n{'=' * 30}")
        print("분석기 통계")
        print(f"{'=' * 30}")
        stats = analyzer.get_stats()
        print(f"API 호출 횟수: {stats['api_calls']}")
        print(f"오류 발생 횟수: {stats['errors']}")
        print(f"마지막 분석 시간: {stats['last_analysis']}")

    finally:
        # 리소스 정리
        await analyzer.__aexit__(None, None, None)

    print(f"\n{'=' * 50}")
    print("테스트 완료")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(simple_api_test())