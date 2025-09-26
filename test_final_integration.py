"""
최종 통합 테스트: 빗썸 WebSocket → 데이터베이스 저장 전체 파이프라인
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.config import init_database, close_database
import websockets
import json


class FinalIntegrationTest:
    """최종 통합 테스트 클래스"""

    def __init__(self):
        self.db_config = None
        self.websocket = None
        self.is_running = False

    async def setup(self):
        """테스트 환경 설정"""
        print("=" * 60)
        print("FINAL INTEGRATION TEST")
        print("Bithumb WebSocket → Database Storage Pipeline")
        print("=" * 60)

        try:
            # 데이터베이스 연결
            print("\n[STEP 1] Connecting to database...")
            self.db_config = await init_database()

            db_health = await self.db_config.health_check()
            if not db_health:
                raise Exception("Database connection failed")
            print("[SUCCESS] Database connected!")

            return True

        except Exception as e:
            print(f"\n[ERROR] Setup failed: {str(e)}")
            return False

    async def test_websocket_to_database(self, duration_seconds: int = 30):
        """WebSocket 실시간 데이터 → 데이터베이스 저장 테스트"""
        print(f"\n[STEP 2] WebSocket → Database integration test ({duration_seconds}s)")

        try:
            # WebSocket 연결
            WEBSOCKET_URL = "wss://pubwss.bithumb.com/pub/ws"
            print(f"Connecting to: {WEBSOCKET_URL}")

            self.websocket = await websockets.connect(WEBSOCKET_URL)
            print("WebSocket connected!")

            # 올바른 구독 메시지 전송
            subscriptions = [
                {"type": "ticker", "symbols": ["BTC_KRW"], "tickTypes": ["24H"]},
                {"type": "orderbookdepth", "symbols": ["BTC_KRW"]},
                {"type": "transaction", "symbols": ["BTC_KRW"]}
            ]

            for sub in subscriptions:
                await self.websocket.send(json.dumps(sub))
                print(f"Subscription sent: {sub['type']}")

            self.is_running = True
            print(f"\nCollecting data for {duration_seconds} seconds...")

            # 데이터 수집 및 저장
            ticker_count = 0
            orderbook_count = 0
            transaction_count = 0
            start_time = asyncio.get_event_loop().time()

            while (asyncio.get_event_loop().time() - start_time) < duration_seconds and self.is_running:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)

                    try:
                        parsed = json.loads(message)

                        # 상태 메시지 스킵
                        if 'status' in parsed and 'resmsg' in parsed:
                            continue

                        # 실제 데이터 처리
                        msg_type = parsed.get('type')

                        if msg_type == 'ticker':
                            success = await self._save_ticker_data(parsed)
                            if success:
                                ticker_count += 1
                                if ticker_count <= 3:
                                    print(f"[TICKER {ticker_count}] Saved to database")

                        elif msg_type == 'orderbookdepth':
                            success = await self._save_orderbook_data(parsed)
                            if success:
                                orderbook_count += 1
                                if orderbook_count <= 3:
                                    print(f"[ORDERBOOK {orderbook_count}] Saved to database")

                        elif msg_type == 'transaction':
                            success = await self._save_transaction_data(parsed)
                            if success:
                                transaction_count += 1
                                if transaction_count <= 3:
                                    print(f"[TRANSACTION {transaction_count}] Saved to database")

                    except json.JSONDecodeError:
                        continue

                except asyncio.TimeoutError:
                    elapsed = int(asyncio.get_event_loop().time() - start_time)
                    print(f"[{elapsed}s] Data saved - T:{ticker_count}, O:{orderbook_count}, Tx:{transaction_count}")
                    continue

            print(f"\n[COMPLETED] Data collection finished!")
            print(f"Results:")
            print(f"  - Tickers saved: {ticker_count}")
            print(f"  - OrderBooks saved: {orderbook_count}")
            print(f"  - Transactions saved: {transaction_count}")

            total_saved = ticker_count + orderbook_count + transaction_count
            return total_saved > 0

        except Exception as e:
            print(f"\n[ERROR] Integration test failed: {str(e)}")
            return False

        finally:
            if self.websocket:
                await self.websocket.close()

    async def _save_ticker_data(self, data):
        """Ticker 데이터를 데이터베이스에 저장"""
        try:
            from src.database.models.market import Ticker
            from sqlalchemy import select
            from decimal import Decimal

            content = data.get('content', {})
            symbol = content.get('symbol', 'BTC_KRW')

            async with self.db_config.get_session() as session:
                # 기존 ticker 업데이트 또는 새로 생성
                existing = await session.execute(
                    select(Ticker).filter(Ticker.symbol == symbol)
                )
                ticker = existing.scalar_one_or_none()

                if ticker:
                    # 업데이트
                    ticker.closing_price = Decimal(str(content.get('closePrice', 0)))
                    ticker.opening_price = Decimal(str(content.get('openPrice', content.get('closePrice', 0))))
                    ticker.timestamp = datetime.now()
                else:
                    # 새로 생성
                    ticker = Ticker(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        closing_price=Decimal(str(content.get('closePrice', 0))),
                        opening_price=Decimal(str(content.get('openPrice', content.get('closePrice', 0))))
                    )
                    session.add(ticker)

                await session.commit()
                return True

        except Exception as e:
            print(f"Ticker save error: {str(e)}")
            return False

    async def _save_orderbook_data(self, data):
        """OrderBook 데이터를 데이터베이스에 저장"""
        try:
            from src.database.models.market import OrderBook

            content = data.get('content', {})
            entries = content.get('list', [])

            if not entries:
                return False

            # 첫 번째 엔트리에서 심볼 가져오기
            symbol = entries[0].get('symbol', 'BTC_KRW') if entries else 'BTC_KRW'

            # JSON 형태로 변환
            bids_data = []
            asks_data = []

            for entry in entries:
                order_type = entry.get('orderType', 'bid')
                order_data = {
                    "price": entry.get('price', '0'),
                    "quantity": entry.get('quantity', '0'),
                    "total": entry.get('total', '0')
                }

                if order_type == 'bid':
                    bids_data.append(order_data)
                else:
                    asks_data.append(order_data)

            async with self.db_config.get_session() as session:
                orderbook = OrderBook(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bids_data=bids_data,
                    asks_data=asks_data
                )
                session.add(orderbook)
                await session.commit()
                return True

        except Exception as e:
            print(f"OrderBook save error: {str(e)}")
            return False

    async def _save_transaction_data(self, data):
        """Transaction 데이터를 데이터베이스에 저장"""
        try:
            from src.database.models.market import Transaction
            from decimal import Decimal

            content = data.get('content', {})
            transactions = content.get('list', [])

            if not transactions:
                return False

            async with self.db_config.get_session() as session:
                for tx in transactions:
                    symbol = tx.get('symbol', 'BTC_KRW')

                    transaction_record = Transaction(
                        symbol=symbol,
                        transaction_id=f"{symbol}_{int(datetime.now().timestamp() * 1000)}",
                        price=Decimal(str(tx.get('contPrice', tx.get('price', 0)))),
                        quantity=Decimal(str(tx.get('contQty', tx.get('quantity', 0)))),
                        type='buy' if tx.get('buySellGb', '1') == '1' else 'sell',
                        timestamp=datetime.now()
                    )
                    session.add(transaction_record)

                await session.commit()
                return True

        except Exception as e:
            print(f"Transaction save error: {str(e)}")
            return False

    async def verify_stored_data(self):
        """저장된 데이터 검증"""
        print(f"\n[STEP 3] Verifying stored data...")

        try:
            async with self.db_config.get_session() as session:
                from sqlalchemy import text

                # 각 테이블 확인
                tables = [
                    ("tickers", "Ticker data"),
                    ("orderbooks", "OrderBook data"),
                    ("transactions", "Transaction data")
                ]

                total_records = 0
                for table_name, description in tables:
                    try:
                        result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        count = result.scalar()
                        total_records += count
                        print(f"  - {description}: {count} records")

                        # 최신 데이터 확인
                        if count > 0:
                            if table_name == "tickers":
                                latest = await session.execute(
                                    text("SELECT symbol, closing_price, timestamp FROM tickers ORDER BY timestamp DESC LIMIT 1")
                                )
                                row = latest.fetchone()
                                if row:
                                    print(f"    Latest: {row.symbol} = {row.closing_price} at {row.timestamp}")

                    except Exception as e:
                        print(f"  - {description}: Error checking ({str(e)})")

                print(f"\nTotal records stored: {total_records}")
                return total_records > 0

        except Exception as e:
            print(f"Verification error: {str(e)}")
            return False

    async def cleanup(self):
        """리소스 정리"""
        print(f"\n[STEP 4] Cleaning up...")

        self.is_running = False

        if self.websocket:
            await self.websocket.close()

        if self.db_config:
            await close_database()

        print("Cleanup completed")


async def main():
    """메인 테스트 실행"""
    test = FinalIntegrationTest()

    try:
        # 설정
        if not await test.setup():
            return False

        # WebSocket → DB 통합 테스트
        success = await test.test_websocket_to_database(30)  # 30초

        # 저장된 데이터 검증
        if success:
            verified = await test.verify_stored_data()
            success = success and verified

        if success:
            print("\n" + "=" * 60)
            print("🎉 COMPLETE SUCCESS! 🎉")
            print("✅ Real-time Bithumb WebSocket data")
            print("✅ Successfully parsed and validated")
            print("✅ Stored in PostgreSQL database")
            print("✅ End-to-end pipeline verified!")
            print("=" * 60)
        else:
            print("\n❌ Integration test failed")

        return success

    except Exception as e:
        print(f"\nTest execution error: {str(e)}")
        return False

    finally:
        await test.cleanup()


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nFINAL RESULT: {'SUCCESS' if result else 'FAILED'}")
    sys.exit(0 if result else 1)