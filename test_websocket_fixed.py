"""
수정된 빗썸 WebSocket 구독 메시지로 실제 데이터 수신 테스트
"""

import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import websockets


async def test_fixed_websocket_protocol():
    """올바른 프로토콜로 빗썸 WebSocket 테스트"""
    print("=" * 60)
    print("Fixed Bithumb WebSocket Protocol Test")
    print("=" * 60)

    websocket = None

    try:
        # 연결
        WEBSOCKET_URL = "wss://pubwss.bithumb.com/pub/ws"
        print(f"Connecting to: {WEBSOCKET_URL}")

        websocket = await websockets.connect(WEBSOCKET_URL)
        print("Connected successfully!")

        # 올바른 구독 메시지들 전송
        print("\n=== Sending Correct Subscription Messages ===")

        # 1. Ticker 구독 (tickTypes 포함)
        ticker_subscription = {
            "type": "ticker",
            "symbols": ["BTC_KRW", "ETH_KRW"],
            "tickTypes": ["30M", "1H", "12H", "24H", "MID"]
        }
        await websocket.send(json.dumps(ticker_subscription))
        print(f"✓ Ticker subscription sent: {ticker_subscription}")

        # 2. Orderbook Depth 구독
        orderbook_subscription = {
            "type": "orderbookdepth",
            "symbols": ["BTC_KRW", "ETH_KRW"]
        }
        await websocket.send(json.dumps(orderbook_subscription))
        print(f"✓ Orderbook depth subscription sent: {orderbook_subscription}")

        # 3. Transaction 구독
        transaction_subscription = {
            "type": "transaction",
            "symbols": ["BTC_KRW", "ETH_KRW"]
        }
        await websocket.send(json.dumps(transaction_subscription))
        print(f"✓ Transaction subscription sent: {transaction_subscription}")

        # 4. Orderbook Snapshot도 추가로 구독
        snapshot_subscription = {
            "type": "orderbooksnapshot",
            "symbols": ["BTC_KRW"]
        }
        await websocket.send(json.dumps(snapshot_subscription))
        print(f"✓ Orderbook snapshot subscription sent: {snapshot_subscription}")

        print("\n=== Listening for Real Data (90 seconds) ===")

        message_count = 0
        data_messages = 0
        status_messages = 0
        start_time = asyncio.get_event_loop().time()

        ticker_received = False
        orderbook_received = False
        transaction_received = False

        while (asyncio.get_event_loop().time() - start_time) < 90:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                message_count += 1

                try:
                    parsed = json.loads(message)

                    # 상태 메시지 처리
                    if isinstance(parsed, dict) and 'status' in parsed and 'resmsg' in parsed:
                        status_messages += 1
                        status = parsed['status']
                        msg = parsed['resmsg']

                        print(f"\n[STATUS {status_messages}] {status}: {msg}")

                        if status == "0000":
                            if "Connected" in msg:
                                print("  → WebSocket connection established!")
                            elif "Filter Registered" in msg:
                                print("  → Subscription successful!")
                        else:
                            print(f"  → ⚠️  Subscription error: {msg}")
                        continue

                    # 실제 데이터 메시지 처리
                    data_messages += 1

                    # 메시지 타입 확인
                    msg_type = parsed.get('type', 'unknown')

                    if msg_type == 'ticker':
                        ticker_received = True
                        print(f"\n🎯 [TICKER {data_messages}] Real-time ticker data received!")

                        if 'content' in parsed:
                            content = parsed['content']
                            symbol = content.get('symbol', 'Unknown')
                            closing_price = content.get('closePrice', 'N/A')
                            volume = content.get('volume', 'N/A')
                            print(f"     Symbol: {symbol}")
                            print(f"     Price: {closing_price}")
                            print(f"     Volume: {volume}")

                    elif msg_type == 'orderbookdepth':
                        orderbook_received = True
                        print(f"\n📊 [ORDERBOOK {data_messages}] Orderbook depth update!")

                        if 'content' in parsed:
                            content = parsed['content']
                            symbol = content.get('symbol', 'Unknown')
                            print(f"     Symbol: {symbol}")
                            if 'list' in content:
                                updates = len(content['list'])
                                print(f"     Updates: {updates} entries")

                    elif msg_type == 'transaction':
                        transaction_received = True
                        print(f"\n💰 [TRANSACTION {data_messages}] Trade execution!")

                        if 'content' in parsed:
                            content = parsed['content']
                            symbol = content.get('symbol', 'Unknown')
                            print(f"     Symbol: {symbol}")
                            if 'list' in content:
                                trades = len(content['list'])
                                print(f"     Trades: {trades} executions")

                    else:
                        print(f"\n[DATA {data_messages}] Unknown message type: {msg_type}")
                        print(f"     Keys: {list(parsed.keys())}")

                    # 처음 10개 데이터 메시지만 자세히 출력
                    if data_messages <= 10:
                        print(f"     Raw: {str(parsed)[:200]}...")

                except json.JSONDecodeError:
                    print(f"\n[RAW {message_count}] Non-JSON: {message[:100]}...")

            except asyncio.TimeoutError:
                elapsed = int(asyncio.get_event_loop().time() - start_time)
                print(f"[{elapsed}s] Waiting for data... Status: {status_messages}, Data: {data_messages}")
                continue

        # 결과 분석
        print(f"\n" + "=" * 60)
        print("[TEST RESULTS]")
        print(f"Total messages: {message_count}")
        print(f"Status messages: {status_messages}")
        print(f"Data messages: {data_messages}")
        print(f"\nData Types Received:")
        print(f"  ✅ Ticker: {'YES' if ticker_received else 'NO'}")
        print(f"  ✅ Orderbook: {'YES' if orderbook_received else 'NO'}")
        print(f"  ✅ Transaction: {'YES' if transaction_received else 'NO'}")

        success = data_messages > 0 and ticker_received

        if success:
            print(f"\n🎉 SUCCESS: Fixed WebSocket protocol is working!")
            print(f"✅ Real-time data successfully received!")
        else:
            print(f"\n❌ Issue: Still not receiving expected data")

        return success

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

    finally:
        if websocket:
            await websocket.close()
            print("\nConnection closed")


if __name__ == "__main__":
    try:
        result = asyncio.run(test_fixed_websocket_protocol())
        print(f"\nFinal Result: {'SUCCESS' if result else 'NEEDS MORE WORK'}")
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)