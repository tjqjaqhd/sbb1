"""
실제 빗썸 WebSocket 연결 및 실시간 데이터 저장 테스트
"""

import asyncio
import sys
from pathlib import Path
import signal
from datetime import datetime

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.config import init_database, close_database
from src.services.market_data_service import MarketDataService


class WebSocketRealTest:
    """실제 WebSocket 데이터 수신 및 저장 테스트"""

    def __init__(self):
        self.db_config = None
        self.market_service = None
        self.is_running = False

    async def setup(self):
        """테스트 환경 설정"""
        print("=" * 60)
        print("Real Bithumb WebSocket Data Collection Test")
        print("=" * 60)

        try:
            # 데이터베이스 연결
            print("\n[STEP 1] Connecting to database...")
            self.db_config = await init_database()

            db_health = await self.db_config.health_check()
            if not db_health:
                raise Exception("Database connection failed")
            print("[SUCCESS] Database connected!")

            # 시장 데이터 서비스 초기화
            print("\n[STEP 2] Initializing market data service...")
            self.market_service = MarketDataService(self.db_config)
            print("[SUCCESS] Market data service ready!")

            return True

        except Exception as e:
            print(f"\n[ERROR] Setup failed: {str(e)}")
            return False

    async def test_websocket_connection(self):
        """WebSocket 연결 테스트"""
        print("\n[STEP 3] Testing WebSocket connection...")

        try:
            # 테스트용 심볼
            test_symbols = ["BTC_KRW", "ETH_KRW"]

            # 시장 데이터 서비스 시작
            success = await self.market_service.start(test_symbols)

            if not success:
                raise Exception("Market data service failed to start")

            print(f"[SUCCESS] WebSocket connected and subscribed to: {test_symbols}")
            self.is_running = True

            return True

        except Exception as e:
            print(f"[ERROR] WebSocket connection failed: {str(e)}")
            return False

    async def collect_real_data(self, duration_seconds: int = 60):
        """실시간 데이터 수집"""
        print(f"\n[STEP 4] Collecting real-time data for {duration_seconds} seconds...")
        print("Press Ctrl+C to stop early")

        try:
            start_time = datetime.now()
            last_stats_time = start_time

            while (datetime.now() - start_time).seconds < duration_seconds and self.is_running:
                await asyncio.sleep(5)  # 5초마다 상태 확인

                # 10초마다 통계 출력
                if (datetime.now() - last_stats_time).seconds >= 10:
                    stats = self.market_service.get_stats()
                    elapsed = (datetime.now() - start_time).seconds

                    print(f"\n  === Progress: {elapsed}s ===")
                    print(f"  Tickers saved: {stats['tickers_saved']}")
                    print(f"  OrderBooks saved: {stats['orderbooks_saved']}")
                    print(f"  Transactions saved: {stats['transactions_saved']}")
                    print(f"  Errors: {stats['errors']}")
                    if stats['last_update']:
                        print(f"  Last update: {stats['last_update']}")

                    last_stats_time = datetime.now()

            # 최종 통계
            final_stats = self.market_service.get_stats()
            print(f"\n[COMPLETED] Data collection finished!")
            print(f"  Total duration: {(datetime.now() - start_time).seconds} seconds")
            print(f"  Final stats:")
            print(f"    - Tickers: {final_stats['tickers_saved']}")
            print(f"    - OrderBooks: {final_stats['orderbooks_saved']}")
            print(f"    - Transactions: {final_stats['transactions_saved']}")
            print(f"    - Errors: {final_stats['errors']}")

            return True

        except Exception as e:
            print(f"[ERROR] Data collection failed: {str(e)}")
            return False

    async def verify_stored_data(self):
        """저장된 데이터 검증"""
        print("\n[STEP 5] Verifying stored data in database...")

        try:
            async with self.db_config.get_session() as session:
                from sqlalchemy import text

                # 각 테이블의 레코드 수 확인
                tables = [
                    ("tickers", "Real-time ticker data"),
                    ("orderbooks", "Order book snapshots"),
                    ("transactions", "Trade transactions")
                ]

                total_records = 0

                for table_name, description in tables:
                    try:
                        result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        count = result.scalar()
                        total_records += count
                        print(f"  - {description}: {count} records")

                        # 최신 데이터 샘플 확인
                        if count > 0:
                            if table_name == "tickers":
                                sample = await session.execute(
                                    text("SELECT symbol, closing_price, timestamp FROM tickers ORDER BY timestamp DESC LIMIT 2")
                                )
                                print(f"    Latest samples:")
                                for row in sample:
                                    print(f"      {row.symbol}: {row.closing_price} at {row.timestamp}")

                            elif table_name == "transactions":
                                sample = await session.execute(
                                    text("SELECT symbol, price, quantity, type FROM transactions ORDER BY timestamp DESC LIMIT 3")
                                )
                                print(f"    Latest samples:")
                                for row in sample:
                                    print(f"      {row.symbol}: {row.type} {row.quantity} at {row.price}")

                    except Exception as e:
                        print(f"  - {description}: Error checking ({str(e)})")

                print(f"\n[INFO] Total records stored: {total_records}")

                if total_records == 0:
                    print("[WARNING] No data was stored! WebSocket may not have received data.")
                    return False
                else:
                    print("[SUCCESS] Real-time data successfully stored in database!")
                    return True

        except Exception as e:
            print(f"[ERROR] Data verification failed: {str(e)}")
            return False

    async def cleanup(self):
        """리소스 정리"""
        print("\n[STEP 6] Cleaning up resources...")

        self.is_running = False

        try:
            if self.market_service:
                await self.market_service.stop()
                print("  - Market data service stopped")

            if self.db_config:
                await close_database()
                print("  - Database connection closed")

        except Exception as e:
            print(f"  [WARNING] Cleanup error: {str(e)}")


async def main():
    """메인 테스트 실행"""
    test = WebSocketRealTest()

    # 신호 핸들러 설정 (Ctrl+C 처리)
    def signal_handler(signum, frame):
        print("\n\n[INTERRUPTED] User requested stop...")
        test.is_running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # 설정
        if not await test.setup():
            return False

        # WebSocket 연결 테스트
        if not await test.test_websocket_connection():
            return False

        # 실시간 데이터 수집 (60초)
        print("\n[INFO] Starting real-time data collection...")
        print("       This will collect live data from Bithumb WebSocket")

        await test.collect_real_data(60)

        # 저장된 데이터 검증
        result = await test.verify_stored_data()

        if result:
            print("\n" + "=" * 60)
            print("[SUCCESS] COMPLETE END-TO-END VERIFICATION!")
            print("✅ Bithumb WebSocket → Parsing → Database storage WORKS!")
            print("✅ Real-time trading data pipeline is operational!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("[WARNING] WebSocket connected but no data received")
            print("This might be due to Bithumb WebSocket protocol changes")
            print("=" * 60)

        return result

    except KeyboardInterrupt:
        print("\n\n[STOPPED] User interrupted the test")
        return False
    except Exception as e:
        print(f"\n[ERROR] Test execution failed: {str(e)}")
        return False
    finally:
        await test.cleanup()


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        sys.exit(1)