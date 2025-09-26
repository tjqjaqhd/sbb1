"""
빗썸 WebSocket 메시지 구조 상세 분석
"""

import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import websockets


async def analyze_bithumb_messages():
    """빗썸 WebSocket 메시지 구조 분석"""
    print("=" * 60)
    print("Bithumb WebSocket Message Structure Analysis")
    print("=" * 60)

    websocket = None

    try:
        # 연결
        WEBSOCKET_URL = "wss://pubwss.bithumb.com/pub/ws"
        print(f"Connecting to: {WEBSOCKET_URL}")

        websocket = await websockets.connect(WEBSOCKET_URL)
        print("Connected successfully!")

        # 다양한 구독 시도
        print("\n=== Testing different subscription formats ===")

        # 형태 1: 기본 ticker
        subscription1 = {"type": "ticker", "symbols": ["BTC_KRW"]}
        await websocket.send(json.dumps(subscription1))
        print(f"Sent: {subscription1}")

        # 형태 2: 거래 데이터
        subscription2 = {"type": "transaction", "symbols": ["BTC_KRW"]}
        await websocket.send(json.dumps(subscription2))
        print(f"Sent: {subscription2}")

        # 형태 3: 호가 데이터
        subscription3 = {"type": "orderbook", "symbols": ["BTC_KRW"]}
        await websocket.send(json.dumps(subscription3))
        print(f"Sent: {subscription3}")

        print("\n=== Analyzing received messages ===")

        message_count = 0
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < 20 and message_count < 10:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                message_count += 1

                print(f"\n--- MESSAGE {message_count} ---")

                try:
                    parsed = json.loads(message)
                    print(f"Type: {type(parsed).__name__}")

                    if isinstance(parsed, dict):
                        print("Keys:", list(parsed.keys()))

                        # 각 키의 값 타입과 간단한 내용 출력
                        for key, value in parsed.items():
                            if isinstance(value, dict):
                                print(f"  {key} (dict): keys={list(value.keys())[:5]}")
                                # 가격이나 심볼 정보 찾기
                                for subkey in ['symbol', 'closing_price', 'closePrice', 'price', 'contPrice']:
                                    if subkey in value:
                                        print(f"    -> {subkey}: {value[subkey]}")
                            elif isinstance(value, list):
                                print(f"  {key} (list): length={len(value)}")
                                if len(value) > 0:
                                    print(f"    first item: {str(value[0])[:100]}")
                            else:
                                print(f"  {key}: {str(value)[:100]}")

                    elif isinstance(parsed, list):
                        print(f"List with {len(parsed)} items")
                        if len(parsed) > 0:
                            print(f"First item: {str(parsed[0])[:200]}")

                    else:
                        print(f"Raw content (first 300 chars): {str(parsed)[:300]}")

                except json.JSONDecodeError:
                    print(f"Non-JSON message (first 200 chars): {message[:200]}")

            except asyncio.TimeoutError:
                print("No message received in 3 seconds")
                break

        print(f"\n=== Analysis Complete ===")
        print(f"Total messages received: {message_count}")

        return message_count > 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

    finally:
        if websocket:
            await websocket.close()
            print("Connection closed")


if __name__ == "__main__":
    result = asyncio.run(analyze_bithumb_messages())
    print(f"\nResult: {'SUCCESS' if result else 'FAILED'}")
    sys.exit(0 if result else 1)