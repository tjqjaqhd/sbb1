"""
수정된 빗썸 WebSocket 구독 메시지로 실제 데이터 수신 테스트 (Simple Version)
"""

import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import websockets


async def test_fixed_websocket():
    """올바른 프로토콜로 빗썸 WebSocket 테스트"""
    print("Fixed Bithumb WebSocket Protocol Test")
    print("=" * 50)

    websocket = None

    try:
        # 연결
        WEBSOCKET_URL = "wss://pubwss.bithumb.com/pub/ws"
        print(f"Connecting to: {WEBSOCKET_URL}")

        websocket = await websockets.connect(WEBSOCKET_URL)
        print("Connected successfully!")

        print("\nSending correct subscription messages...")

        # 1. Ticker 구독 (tickTypes 포함)
        ticker_sub = {
            "type": "ticker",
            "symbols": ["BTC_KRW"],
            "tickTypes": ["24H"]
        }
        await websocket.send(json.dumps(ticker_sub))
        print(f"Ticker subscription sent: {ticker_sub}")

        # 2. Orderbook Depth 구독
        orderbook_sub = {
            "type": "orderbookdepth",
            "symbols": ["BTC_KRW"]
        }
        await websocket.send(json.dumps(orderbook_sub))
        print(f"Orderbook depth subscription sent: {orderbook_sub}")

        # 3. Transaction 구독
        transaction_sub = {
            "type": "transaction",
            "symbols": ["BTC_KRW"]
        }
        await websocket.send(json.dumps(transaction_sub))
        print(f"Transaction subscription sent: {transaction_sub}")

        print("\nListening for messages (60 seconds)...")

        message_count = 0
        data_count = 0
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < 60:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                message_count += 1

                try:
                    parsed = json.loads(message)

                    # 상태 메시지
                    if 'status' in parsed and 'resmsg' in parsed:
                        print(f"[STATUS] {parsed['status']}: {parsed['resmsg']}")
                        continue

                    # 데이터 메시지
                    data_count += 1
                    msg_type = parsed.get('type', 'unknown')

                    print(f"\n[DATA {data_count}] Type: {msg_type}")

                    if 'content' in parsed:
                        content = parsed['content']
                        symbol = content.get('symbol', 'N/A')
                        print(f"  Symbol: {symbol}")

                        if msg_type == 'ticker':
                            price = content.get('closePrice', content.get('closing_price', 'N/A'))
                            print(f"  Price: {price}")

                        elif msg_type in ['orderbookdepth', 'orderbooksnapshot']:
                            if 'list' in content:
                                print(f"  Entries: {len(content['list'])}")

                        elif msg_type == 'transaction':
                            if 'list' in content:
                                print(f"  Trades: {len(content['list'])}")

                    if data_count <= 5:  # 처음 5개만 상세히
                        print(f"  Raw: {str(parsed)[:150]}...")

                except json.JSONDecodeError:
                    print(f"[RAW {message_count}] Non-JSON: {message[:100]}...")

            except asyncio.TimeoutError:
                elapsed = int(asyncio.get_event_loop().time() - start_time)
                print(f"[{elapsed}s] Waiting... Messages: {message_count}, Data: {data_count}")
                continue

        print(f"\n" + "=" * 50)
        print(f"Test completed!")
        print(f"Total messages: {message_count}")
        print(f"Data messages: {data_count}")

        success = data_count > 0

        if success:
            print("SUCCESS: Real-time data received!")
        else:
            print("Issue: No data messages received")

        return success

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

    finally:
        if websocket:
            await websocket.close()
            print("Connection closed")


if __name__ == "__main__":
    result = asyncio.run(test_fixed_websocket())
    print(f"\nResult: {'SUCCESS' if result else 'FAILED'}")
    sys.exit(0 if result else 1)