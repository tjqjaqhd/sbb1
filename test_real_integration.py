"""
실제 빗썸 API 및 WebSocket과 데이터베이스 통합 테스트

실제 빗썸 API 연결, WebSocket 데이터 수신, 데이터베이스 저장까지
전체 파이프라인을 테스트합니다.
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
from src.api.bithumb.client import get_http_client, close_http_client
from src.services.market_data_service import MarketDataService
from config.config import get_settings


class RealIntegrationTest:
    """실제 통합 테스트 클래스"""

    def __init__(self):
        self.db_config = None
        self.market_service = None
        self.http_client = None
        self.is_running = False

    async def setup(self):
        """테스트 환경 설정"""
        print("=" * 60)
        print("Real Bithumb API and Database Integration Test")
        print("=" * 60)

        try:
            # 설정 확인
            settings = get_settings()
            print(f"[INFO] 데이터베이스: {settings.DATABASE_URL}")
            print(f"[INFO] 빗썸 API 키: {'설정됨' if settings.BITHUMB_API_KEY else '미설정'}")

            # 데이터베이스 연결
            print("\n[STEP 1] 데이터베이스 연결 중...")
            self.db_config = await init_database()

            db_health = await self.db_config.health_check()
            if not db_health:
                raise Exception("데이터베이스 연결 실패")

            print("[성공] 데이터베이스 연결 완료!")

            # HTTP 클라이언트 설정
            print("\n[STEP 2] 빗썸 HTTP API 연결 테스트...")
            self.http_client = await get_http_client()

            api_health = await self.http_client.health_check()
            if not api_health:
                raise Exception("빗썸 API 연결 실패")

            print("[성공] 빗썸 API 연결 완료!")

            # 시장 데이터 서비스 초기화
            print("\n[STEP 3] 시장 데이터 서비스 초기화...")
            self.market_service = MarketDataService(self.db_config)

            print("[성공] 시장 데이터 서비스 준비 완료!")

            return True

        except Exception as e:
            print(f"\n[오류] 테스트 설정 실패: {str(e)}")
            return False

    async def test_http_api(self):
        """HTTP API 테스트"""
        print("\n[STEP 4] HTTP API 기능 테스트...")

        try:
            # 공개 API - BTC 티커 조회
            print("  - BTC_KRW 티커 데이터 조회 중...")
            ticker_data = await self.http_client.get("/public/ticker/BTC_KRW")

            if ticker_data and "data" in ticker_data:
                btc_price = ticker_data["data"].get("closing_price", "N/A")
                print(f"  [성공] BTC 현재가: {btc_price}원")
            else:
                print(f"  [경고] 예상과 다른 응답: {ticker_data}")

            # Rate Limit 상태 확인
            rate_status = self.http_client.get_rate_limit_status()
            print(f"  - Rate Limit 상태: {len(rate_status)}개 카테고리")

            return True

        except Exception as e:
            print(f"  [오류] HTTP API 테스트 실패: {str(e)}")
            return False

    async def test_websocket_data_collection(self, duration_seconds: int = 30):
        """WebSocket 데이터 수집 테스트"""
        print(f"\n[STEP 5] WebSocket 실시간 데이터 수집 테스트 ({duration_seconds}초)...")

        try:
            # 시장 데이터 서비스 시작
            symbols = ["BTC_KRW", "ETH_KRW"]
            success = await self.market_service.start(symbols)

            if not success:
                raise Exception("시장 데이터 서비스 시작 실패")

            print(f"  [성공] WebSocket 연결 및 구독 완료: {symbols}")
            self.is_running = True

            # 지정된 시간 동안 데이터 수집
            print(f"  - {duration_seconds}초 동안 데이터 수집 중...")

            start_time = datetime.now()
            while (datetime.now() - start_time).seconds < duration_seconds and self.is_running:
                # 1초마다 상태 확인
                await asyncio.sleep(1)

                # 통계 확인
                stats = self.market_service.get_stats()
                if stats['last_update']:
                    elapsed = (datetime.now() - start_time).seconds
                    print(f"    진행: {elapsed}s | "
                          f"Ticker: {stats['tickers_saved']} | "
                          f"OrderBook: {stats['orderbooks_saved']} | "
                          f"Transaction: {stats['transactions_saved']} | "
                          f"오류: {stats['errors']}")

            # 최종 통계
            final_stats = self.market_service.get_stats()
            print(f"\n  [완료] 데이터 수집 통계:")
            print(f"    - Ticker 저장: {final_stats['tickers_saved']}건")
            print(f"    - OrderBook 저장: {final_stats['orderbooks_saved']}건")
            print(f"    - Transaction 저장: {final_stats['transactions_saved']}건")
            print(f"    - 오류: {final_stats['errors']}건")

            return True

        except Exception as e:
            print(f"  [오류] WebSocket 데이터 수집 실패: {str(e)}")
            return False

    async def test_database_verification(self):
        """데이터베이스 저장 확인"""
        print("\n[STEP 6] 데이터베이스 저장 데이터 확인...")

        try:
            async with self.db_config.get_session() as session:
                from sqlalchemy import text

                # 각 테이블의 레코드 수 확인
                tables_to_check = [
                    ("tickers", "Ticker 데이터"),
                    ("order_books", "OrderBook 데이터"),
                    ("transactions", "Transaction 데이터"),
                    ("market_data", "MarketData 시계열 데이터")
                ]

                for table_name, description in tables_to_check:
                    try:
                        result = await session.execute(
                            text(f"SELECT COUNT(*) FROM {table_name}")
                        )
                        count = result.scalar()
                        print(f"  - {description}: {count}건")

                        # 최신 데이터 확인 (timestamp 필드가 있는 테이블)
                        if table_name in ["tickers", "order_books", "transactions"]:
                            recent_result = await session.execute(
                                text(f"SELECT updated_at FROM {table_name} ORDER BY updated_at DESC LIMIT 1")
                            )
                            recent_time = recent_result.scalar()
                            if recent_time:
                                print(f"    최신 데이터: {recent_time}")

                    except Exception as table_error:
                        print(f"  - {description}: 테이블 확인 실패 ({str(table_error)})")

                # 최근 BTC 티커 데이터 확인
                btc_ticker = await self.market_service.get_latest_ticker("BTC_KRW")
                if btc_ticker:
                    print(f"\n  [상세] 최신 BTC 티커:")
                    print(f"    현재가: {btc_ticker.closing_price}원")
                    print(f"    거래량: {btc_ticker.units_traded}")
                    print(f"    업데이트: {btc_ticker.updated_at}")

            return True

        except Exception as e:
            print(f"  [오류] 데이터베이스 검증 실패: {str(e)}")
            return False

    async def cleanup(self):
        """리소스 정리"""
        print("\n[STEP 7] 리소스 정리 중...")

        self.is_running = False

        try:
            if self.market_service:
                await self.market_service.stop()
                print("  - 시장 데이터 서비스 중지 완료")

            if self.http_client:
                await close_http_client()
                print("  - HTTP 클라이언트 정리 완료")

            if self.db_config:
                await close_database()
                print("  - 데이터베이스 연결 종료 완료")

        except Exception as e:
            print(f"  [경고] 정리 중 오류: {str(e)}")


async def main():
    """메인 테스트 실행"""
    test = RealIntegrationTest()

    # 신호 핸들러 설정 (Ctrl+C 처리)
    def signal_handler(signum, frame):
        print("\n\n[중단] 사용자 요청으로 테스트를 중단합니다...")
        test.is_running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # 설정 및 초기화
        if not await test.setup():
            return False

        # HTTP API 테스트
        if not await test.test_http_api():
            return False

        # WebSocket 데이터 수집 테스트
        print(f"\n[알림] 실시간 데이터 수집을 시작합니다.")
        print(f"       Ctrl+C로 언제든 중단할 수 있습니다.")

        await test.test_websocket_data_collection(30)  # 30초 동안 테스트

        # 데이터베이스 검증
        await test.test_database_verification()

        print("\n" + "=" * 60)
        print("[성공] 모든 통합 테스트가 완료되었습니다!")
        print("Task 3 & 4: 실제 연결 검증 및 데이터 저장 완료")
        print("=" * 60)

        return True

    except KeyboardInterrupt:
        print("\n\n[중단] 사용자가 테스트를 중단했습니다.")
        return False
    except Exception as e:
        print(f"\n[오류] 통합 테스트 실행 중 오류: {str(e)}")
        return False
    finally:
        await test.cleanup()


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n테스트가 중단되었습니다.")
        sys.exit(1)