"""
빗썸 WebSocket 장시간 테스트 - 실제 데이터 수신 확인
"""

import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import websockets


async def long_term_websocket_test():
    """빗썸 WebSocket 장시간 테스트"""
    print("=" * 60)
    print("Bithumb WebSocket Long-term Data Reception Test")
    print("=" * 60)

    websocket = None

    try:
        # 연결
        WEBSOCKET_URL = "wss://pubwss.bithumb.com/pub/ws"
        print(f"Connecting to: {WEBSOCKET_URL}")

        websocket = await websockets.connect(WEBSOCKET_URL)
        print("Connected successfully!")

        # 성공했던 구독 메시지만 전송 (orderbook이 성공했음)
        print("\nSending successful subscription...")
        subscription = {"type": "orderbook", "symbols": ["BTC_KRW"]}
        await websocket.send(json.dumps(subscription))
        print(f"Sent: {subscription}")

        # 추가로 다른 형태도 시도
        print("\nTrying alternative subscription formats...")

        # 형태 1: 단순 ticker
        alt1 = {"type": "ticker", "symbols": ["BTC_KRW"]}
        await websocket.send(json.dumps(alt1))

        # 형태 2: 다른 형태 시도
        alt2 = {"cmd": "subscribe", "args": ["ticker:BTC_KRW"]}
        await websocket.send(json.dumps(alt2))

        # 형태 3: 또 다른 형태
        alt3 = {"event": "bts:subscribe", "data": {"channel": "ticker_BTC_KRW"}}
        await websocket.send(json.dumps(alt3))

        print("\nListening for data messages (2 minutes)...")
        print("We will filter out status messages and look for actual data...")

        message_count = 0
        data_messages = 0
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < 120:  # 2분
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                message_count += 1

                try:
                    parsed = json.loads(message)

                    # 상태 메시지가 아닌 데이터 메시지인지 확인
                    if isinstance(parsed, dict):
                        # 상태 메시지 필터링
                        if 'status' in parsed and 'resmsg' in parsed:
                            # 상태 메시지는 간단히 출력
                            print(f"Status [{message_count}]: {parsed['resmsg']}")
                            continue

                        # 실제 데이터 메시지인 경우
                        data_messages += 1
                        print(f"\n=== DATA MESSAGE {data_messages} (total: {message_count}) ===")
                        print(f"Keys: {list(parsed.keys())}")

                        # 중요한 필드들 확인
                        data_fields = ['type', 'content', 'data', 'symbol', 'price', 'timestamp']
                        for field in data_fields:
                            if field in parsed:
                                print(f"{field}: {str(parsed[field])[:100]}")

                        # content가 있으면 자세히 분석
                        if 'content' in parsed and isinstance(parsed['content'], dict):
                            content = parsed['content']
                            print(f"Content keys: {list(content.keys())}")

                            # 가격 정보 찾기
                            price_fields = ['closePrice', 'closing_price', 'price', 'contPrice']
                            for pf in price_fields:
                                if pf in content:
                                    print(f"  PRICE ({pf}): {content[pf]}")

                        print("=" * 40)

                except json.JSONDecodeError:
                    print(f"Non-JSON message [{message_count}]: {message[:200]}")

            except asyncio.TimeoutError:
                print(f"[{int(asyncio.get_event_loop().time() - start_time)}s] No message for 10 seconds...")
                continue

        print(f"\n=== TEST COMPLETED ===")
        print(f"Total messages received: {message_count}")
        print(f"Data messages (non-status): {data_messages}")

        if data_messages > 0:
            print("\n✅ SUCCESS: Received actual data messages!")
            print("WebSocket is working and we can get real-time data!")
            return True
        else:
            print("\n⚠️  WARNING: Only status messages received")
            print("Need to adjust subscription format or wait longer")
            return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

    finally:
        if websocket:
            await websocket.close()
            print("\nConnection closed")


if __name__ == "__main__":
    try:
        result = asyncio.run(long_term_websocket_test())
        print(f"\nFinal Result: {'SUCCESS' if result else 'NEEDS WORK'}")
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)