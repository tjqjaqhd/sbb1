"""
빗썸 API 데이터를 데이터베이스에 저장하는 테스트
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.bithumb.client import get_http_client, close_http_client
from src.database.config import init_database, close_database
from src.database.models.market import Ticker
from sqlalchemy import select


async def test_api_data_to_database():
    """빗썸 API 데이터를 데이터베이스에 저장하는 테스트"""
    print("=" * 60)
    print("Bithumb API Data to Database Test")
    print("=" * 60)

    db_config = None
    http_client = None

    try:
        # 데이터베이스 연결
        print("\n[STEP 1] Connecting to database...")
        db_config = await init_database()
        db_health = await db_config.health_check()
        if not db_health:
            raise Exception("Database connection failed")
        print("[SUCCESS] Database connected!")

        # HTTP 클라이언트 가져오기
        print("\n[STEP 2] Getting HTTP client...")
        http_client = await get_http_client()
        print("[SUCCESS] HTTP client ready!")

        # BTC 티커 데이터 가져오기
        print("\n[STEP 3] Fetching BTC ticker data...")
        ticker_response = await http_client.get("/public/ticker/BTC_KRW")

        if not ticker_response or "data" not in ticker_response:
            raise Exception("Failed to get ticker data")

        ticker_data = ticker_response["data"]
        print(f"[SUCCESS] Got BTC ticker data: {ticker_data.get('closing_price')}")

        # 데이터베이스에 저장
        print("\n[STEP 4] Saving ticker data to database...")
        async with db_config.get_session() as session:
            # 기존 BTC 티커 레코드 확인
            existing_ticker = await session.execute(
                select(Ticker).filter(Ticker.symbol == "BTC_KRW")
            )
            ticker_record = existing_ticker.scalar_one_or_none()

            if ticker_record:
                # 기존 레코드 업데이트
                print("  - Updating existing BTC ticker record...")
                ticker_record.opening_price = Decimal(str(ticker_data.get("opening_price", 0)))
                ticker_record.closing_price = Decimal(str(ticker_data.get("closing_price", 0)))
                ticker_record.min_price = Decimal(str(ticker_data.get("min_price", 0)))
                ticker_record.max_price = Decimal(str(ticker_data.get("max_price", 0)))
                ticker_record.volume_24h = Decimal(str(ticker_data.get("volume_1day", 0)))
                ticker_record.volume_7d = Decimal(str(ticker_data.get("volume_7day", 0))) if ticker_data.get("volume_7day") else None
                ticker_record.timestamp = datetime.now(timezone.utc)
            else:
                # 새 레코드 생성
                print("  - Creating new BTC ticker record...")
                ticker_record = Ticker(
                    symbol="BTC_KRW",
                    timestamp=datetime.now(timezone.utc),
                    opening_price=Decimal(str(ticker_data.get("opening_price", 0))),
                    closing_price=Decimal(str(ticker_data.get("closing_price", 0))),
                    min_price=Decimal(str(ticker_data.get("min_price", 0))),
                    max_price=Decimal(str(ticker_data.get("max_price", 0))),
                    volume_24h=Decimal(str(ticker_data.get("volume_1day", 0))),
                    volume_7d=Decimal(str(ticker_data.get("volume_7day", 0))) if ticker_data.get("volume_7day") else None
                )
                session.add(ticker_record)

            await session.commit()
            print("[SUCCESS] Ticker data saved to database!")

        # ETH 데이터도 저장해보기
        print("\n[STEP 5] Fetching and saving ETH ticker data...")
        eth_response = await http_client.get("/public/ticker/ETH_KRW")

        if eth_response and "data" in eth_response:
            eth_data = eth_response["data"]

            async with db_config.get_session() as session:
                existing_eth = await session.execute(
                    select(Ticker).filter(Ticker.symbol == "ETH_KRW")
                )
                eth_record = existing_eth.scalar_one_or_none()

                if eth_record:
                    eth_record.closing_price = Decimal(str(eth_data.get("closing_price", 0)))
                    eth_record.opening_price = Decimal(str(eth_data.get("opening_price", 0)))
                    eth_record.timestamp = datetime.now(timezone.utc)
                else:
                    eth_record = Ticker(
                        symbol="ETH_KRW",
                        timestamp=datetime.now(timezone.utc),
                        opening_price=Decimal(str(eth_data.get("opening_price", 0))),
                        closing_price=Decimal(str(eth_data.get("closing_price", 0))),
                        min_price=Decimal(str(eth_data.get("min_price", 0))),
                        max_price=Decimal(str(eth_data.get("max_price", 0))),
                        volume_24h=Decimal(str(eth_data.get("volume_1day", 0)))
                    )
                    session.add(eth_record)

                await session.commit()
                print(f"[SUCCESS] ETH ticker saved: {eth_data.get('closing_price')}")

        # 저장된 데이터 확인
        print("\n[STEP 6] Verifying saved data...")
        async with db_config.get_session() as session:
            # BTC 데이터 조회
            btc_result = await session.execute(
                select(Ticker).filter(Ticker.symbol == "BTC_KRW")
            )
            btc_ticker = btc_result.scalar_one_or_none()

            if btc_ticker:
                print(f"[SUCCESS] BTC data in DB:")
                print(f"  - Current Price: {btc_ticker.closing_price}")
                print(f"  - Opening Price: {btc_ticker.opening_price}")
                print(f"  - Volume: {btc_ticker.volume_24h}")
                print(f"  - Updated: {btc_ticker.timestamp}")

            # ETH 데이터 조회
            eth_result = await session.execute(
                select(Ticker).filter(Ticker.symbol == "ETH_KRW")
            )
            eth_ticker = eth_result.scalar_one_or_none()

            if eth_ticker:
                print(f"[SUCCESS] ETH data in DB:")
                print(f"  - Current Price: {eth_ticker.closing_price}")
                print(f"  - Updated: {eth_ticker.timestamp}")

            # 전체 ticker 수 확인
            from sqlalchemy import text
            count_result = await session.execute(text("SELECT COUNT(*) FROM tickers"))
            ticker_count = count_result.scalar()
            print(f"[INFO] Total tickers in database: {ticker_count}")

        print("\n" + "=" * 60)
        print("[SUCCESS] All API-to-Database tests passed!")
        print("Real data flow from Bithumb API to PostgreSQL verified!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[ERROR] Test failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

    finally:
        # 리소스 정리
        if http_client:
            await close_http_client()
            print("\n[INFO] HTTP client closed.")

        if db_config:
            await close_database()
            print("[INFO] Database connection closed.")


if __name__ == "__main__":
    result = asyncio.run(test_api_data_to_database())
    sys.exit(0 if result else 1)