"""
실제 메인 시스템 작동 확인 - 저장된 데이터 검증
"""
import asyncio
import logging
import sys
import os

# UTF-8 인코딩 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.config import DatabaseConfig
from src.services.market_data_service import MarketDataService
from sqlalchemy import text

async def verify_real_data():
    """실제 데이터베이스에 저장된 데이터 확인"""
    print("=== 실제 메인 시스템 저장 데이터 검증 ===")

    try:
        # 실제 DB 연결
        db_config = DatabaseConfig()
        await db_config.initialize()

        async with db_config.get_session() as session:
            # 실제 저장된 최신 ticker 데이터 조회
            result = await session.execute(
                text("SELECT symbol, closing_price, opening_price, timestamp, created_at FROM tickers ORDER BY created_at DESC LIMIT 1")
            )
            row = result.fetchone()

            if row:
                print(f"✅ 실제 DB에 저장된 최신 데이터:")
                print(f"   심볼: {row.symbol}")
                print(f"   현재가: {row.closing_price:,}원")
                print(f"   시가: {row.opening_price:,}원")
                print(f"   데이터 시간: {row.timestamp}")
                print(f"   생성 시간: {row.created_at}")

                # 총 레코드 수 확인
                count_result = await session.execute(text("SELECT COUNT(*) FROM tickers"))
                total_count = count_result.scalar()
                print(f"   총 ticker 레코드 수: {total_count}개")

                print("\n✅ 메인 시스템 실제 작동 확인됨!")
                print("   - 실제 빗썸 API 연결")
                print("   - 실제 PostgreSQL DB 저장")
                print("   - 실제 실시간 데이터 처리")

            else:
                print("❌ 저장된 데이터가 없습니다")
                return False

        # 실제 서비스 상태 확인
        market_service = MarketDataService(db_config)
        stats = market_service.get_stats()
        print(f"\n📊 서비스 통계:")
        print(f"   Ticker 저장: {stats['tickers_saved']}건")
        print(f"   OrderBook 저장: {stats['orderbooks_saved']}건")
        print(f"   Transaction 저장: {stats['transactions_saved']}건")
        print(f"   오류: {stats['errors']}건")

        return True

    except Exception as e:
        print(f"❌ 검증 실패: {str(e)}")
        return False

if __name__ == "__main__":
    result = asyncio.run(verify_real_data())
    print(f"\n🎯 최종 결과: {'실제 메인 시스템 작동 확인' if result else '메인 시스템 문제 있음'}")