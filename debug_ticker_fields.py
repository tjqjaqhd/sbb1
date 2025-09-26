"""
빗썸 ticker 메시지의 실제 필드명을 확인하는 디버그 스크립트
"""
import asyncio
import logging
import sys
import os
import json
from datetime import datetime, timezone

# UTF-8 인코딩 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api.bithumb.websocket_client import BithumbWebSocketClient, SubscriptionType, get_websocket_client

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 받은 메시지 출력을 위한 핸들러
def debug_ticker_handler(ticker_data, message):
    """Ticker 메시지 디버그 출력"""
    print("[DEBUG] Ticker 메시지 원본:")
    print(json.dumps(message, indent=2, ensure_ascii=False, default=str))
    print("=" * 50)

def debug_orderbook_handler(orderbook_data, message):
    """OrderBook 메시지 디버그 출력"""
    print("[DEBUG] OrderBook 메시지 원본:")
    print(json.dumps(message, indent=2, ensure_ascii=False, default=str))
    print("=" * 50)

def debug_transaction_handler(transaction_data, message):
    """Transaction 메시지 디버그 출력"""
    print("[DEBUG] Transaction 메시지 원본:")
    print(json.dumps(message, indent=2, ensure_ascii=False, default=str))
    print("=" * 50)

async def debug_messages():
    """실제 메시지 구조 디버깅"""
    print("[INFO] 빗썸 메시지 구조 디버깅 시작...")

    try:
        # WebSocket 클라이언트 생성
        client = await get_websocket_client()

        # 디버그 핸들러 등록
        client.add_message_handler(SubscriptionType.TICKER, debug_ticker_handler)
        client.add_message_handler(SubscriptionType.ORDERBOOK, debug_orderbook_handler)
        client.add_message_handler(SubscriptionType.TRANSACTION, debug_transaction_handler)

        # 연결 및 구독
        await client.connect()
        symbols = ["BTC_KRW"]

        await client.subscribe(SubscriptionType.TICKER, symbols)
        await client.subscribe(SubscriptionType.ORDERBOOK, symbols)
        await client.subscribe(SubscriptionType.TRANSACTION, symbols)

        # 10초간 메시지 수신
        print("[INFO] 10초간 메시지 수신...")
        await asyncio.sleep(10)

        # 연결 해제
        await client.disconnect()

    except Exception as e:
        print(f"[ERROR] 디버깅 중 오류: {str(e)}")

if __name__ == "__main__":
    asyncio.run(debug_messages())