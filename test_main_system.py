"""
메인 시스템 end-to-end 검증 테스트

수정된 메인 코드들이 실제로 작동하는지 검증
"""
import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# UTF-8 인코딩 설정
import locale
import codecs

# Windows 터미널 UTF-8 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.config import DatabaseConfig
from src.services.market_data_service import MarketDataService

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def test_main_system():
    """메인 시스템 end-to-end 테스트"""
    print("[INFO] 메인 시스템 검증 시작...")

    try:
        # 1. 데이터베이스 설정
        db_config = DatabaseConfig()

        print("[INFO] 데이터베이스 초기화...")
        await db_config.initialize()

        print("[INFO] 데이터베이스 연결 테스트...")
        is_healthy = await db_config.health_check()
        if not is_healthy:
            print("[ERROR] 데이터베이스 연결 실패")
            return False

        print("[SUCCESS] 데이터베이스 연결 성공")

        # 2. 시장 데이터 서비스 시작
        print("[INFO] 시장 데이터 서비스 초기화...")
        market_service = MarketDataService(db_config)

        # 3. WebSocket 연결 및 구독
        print("[INFO] WebSocket 연결 및 데이터 구독...")
        symbols = ["BTC_KRW"]
        success = await market_service.start(symbols)

        if not success:
            print("[ERROR] 시장 데이터 서비스 시작 실패")
            return False

        print("[SUCCESS] 시장 데이터 서비스 시작 완료")

        # 4. 15초간 데이터 수집
        print("[INFO] 15초간 실시간 데이터 수집...")
        await asyncio.sleep(15)

        # 5. 통계 확인
        stats = market_service.get_stats()
        print(f"[INFO] 수집 통계:")
        print(f"  - Ticker 데이터: {stats['tickers_saved']}건")
        print(f"  - OrderBook 데이터: {stats['orderbooks_saved']}건")
        print(f"  - Transaction 데이터: {stats['transactions_saved']}건")
        print(f"  - 오류: {stats['errors']}건")
        print(f"  - 마지막 업데이트: {stats['last_update']}")

        # 6. 최신 Ticker 데이터 조회
        print("[INFO] 저장된 최신 Ticker 데이터 조회...")
        ticker = await market_service.get_latest_ticker("BTC_KRW")
        if ticker:
            print(f"[SUCCESS] BTC 현재 가격: {ticker.closing_price:,}원")
            print(f"  - 종가: {ticker.closing_price:,}원")
            print(f"  - 시가: {ticker.opening_price:,}원")
            print(f"  - 고가: {ticker.max_price:,}원")
            print(f"  - 저가: {ticker.min_price:,}원")
            print(f"  - 거래량(24H): {ticker.volume_1day}")
            print(f"  - 업데이트: {ticker.updated_at}")
        else:
            print("[WARNING] Ticker 데이터를 찾을 수 없음")

        # 7. 서비스 상태 확인
        print("[INFO] 서비스 상태 확인...")
        health = await market_service.health_check()
        print(f"[INFO] 서비스 상태:")
        print(f"  - 서비스 실행 중: {health['service_running']}")
        print(f"  - 데이터베이스 연결: {health['database_connected']}")
        print(f"  - WebSocket 상태: {health['websocket']}")

        # 8. 서비스 중지
        print("[INFO] 서비스 중지...")
        await market_service.stop()

        # 검증 결과
        total_data = stats['tickers_saved'] + stats['orderbooks_saved'] + stats['transactions_saved']

        if total_data > 0 and stats['errors'] == 0:
            print(f"[COMPLETE] 메인 시스템 검증 성공! 총 {total_data}건 데이터 수집")
            return True
        elif total_data > 0:
            print(f"[WARNING] 데이터 수집은 됐지만 {stats['errors']}개 오류 발생")
            return True
        else:
            print("[FAILED] 데이터 수집 실패")
            return False

    except Exception as e:
        print(f"[ERROR] 메인 시스템 테스트 중 오류: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_main_system())