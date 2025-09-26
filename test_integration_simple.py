"""
간단한 최종 통합 테스트: WebSocket -> Database
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from decimal import Decimal

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.config import init_database, close_database
from src.database.models.market import Ticker
from sqlalchemy import select
import websockets


async def simple_integration_test():
    """간단한 통합 테스트"""
    print("Simple Integration Test: WebSocket to Database")
    print("=" * 50)

    db_config = None
    websocket = None

    try:
        # 1. 데이터베이스 연결
        print("Step 1: Connecting to database...")
        db_config = await init_database()
        if not await db_config.health_check():
            raise Exception("Database connection failed")
        print("Database connected successfully!")

        # 2. WebSocket 연결 및 데이터 수신
        print("\nStep 2: WebSocket data collection...")
        WEBSOCKET_URL = "wss://pubwss.bithumb.com/pub/ws"
        websocket = await websockets.connect(WEBSOCKET_URL)
        print("WebSocket connected!")

        # 올바른 구독
        ticker_sub = {"type": "ticker", "symbols": ["BTC_KRW"], "tickTypes": ["24H"]}
        await websocket.send(json.dumps(ticker_sub))
        print("Ticker subscription sent")

        # 3. 데이터 수신 및 저장
        data_saved = 0
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < 20 and data_saved < 3:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                parsed = json.loads(message)

                # 상태 메시지 스킵
                if 'status' in parsed:
                    print(f"Status: {parsed.get('resmsg', 'Unknown')}")
                    continue

                # Ticker 데이터 처리
                if parsed.get('type') == 'ticker':
                    content = parsed.get('content', {})
                    symbol = content.get('symbol', 'BTC_KRW')
                    price = content.get('closePrice')

                    if price:
                        print(f"Received ticker: {symbol} = {price}")

                        # 데이터베이스에 저장
                        async with db_config.get_session() as session:
                            existing = await session.execute(
                                select(Ticker).filter(Ticker.symbol == symbol)
                            )
                            ticker = existing.scalar_one_or_none()

                            if ticker:
                                ticker.closing_price = Decimal(str(price))
                                ticker.timestamp = datetime.now()
                            else:
                                ticker = Ticker(
                                    symbol=symbol,
                                    timestamp=datetime.now(),
                                    closing_price=Decimal(str(price)),
                                    opening_price=Decimal(str(price))
                                )
                                session.add(ticker)

                            await session.commit()
                            data_saved += 1
                            print(f"Saved to database! (Count: {data_saved})")

            except asyncio.TimeoutError:
                print("Waiting for data...")
                continue
            except Exception as e:
                print(f"Processing error: {str(e)}")
                continue

        # 4. 검증
        print(f"\nStep 3: Verification...")
        async with db_config.get_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT COUNT(*) FROM tickers"))
            count = result.scalar()
            print(f"Total tickers in database: {count}")

            if count > 0:
                latest = await session.execute(
                    text("SELECT symbol, closing_price, timestamp FROM tickers ORDER BY timestamp DESC LIMIT 1")
                )
                row = latest.fetchone()
                print(f"Latest: {row.symbol} = {row.closing_price} at {row.timestamp}")

        success = data_saved > 0

        if success:
            print("\n" + "=" * 50)
            print("SUCCESS! Complete pipeline verified:")
            print("- WebSocket connection: OK")
            print("- Real-time data reception: OK")
            print("- Database storage: OK")
            print("- Data verification: OK")
            print("=" * 50)
        else:
            print("\nFailed to save data to database")

        return success

    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

    finally:
        if websocket:
            await websocket.close()
        if db_config:
            await close_database()
        print("Resources cleaned up")


if __name__ == "__main__":
    result = asyncio.run(simple_integration_test())
    print(f"\nResult: {'SUCCESS' if result else 'FAILED'}")
    sys.exit(0 if result else 1)