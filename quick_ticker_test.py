"""
빠른 ticker 메시지 구조 확인
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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def quick_test():
    """빠른 테스트 - 1개 메시지만 확인"""
    print("[INFO] 빠른 ticker 메시지 구조 확인...")

    try:
        db_config = DatabaseConfig()
        await db_config.initialize()

        market_service = MarketDataService(db_config)
        success = await market_service.start(["BTC_KRW"])

        if success:
            print("[INFO] 5초간 데이터 수집...")
            await asyncio.sleep(5)

        await market_service.stop()
    except Exception as e:
        print(f"[ERROR] 테스트 오류: {str(e)}")

if __name__ == "__main__":
    asyncio.run(quick_test())