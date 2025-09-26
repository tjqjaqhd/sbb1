"""
빗썸 WebSocket 간단 연결 테스트
"""

import asyncio
import json
import sys
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import websockets
    from websockets import connect
except ImportError:
    print("websockets library not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'websockets'])
    import websockets
    from websockets import connect


async def test_bithumb_websocket():
    """빗썸 WebSocket 직접 연결 테스트"""
    print("=" * 60)
    print("Simple Bithumb WebSocket Connection Test")
    print("=" * 60)

    websocket = None

    try:
        # 빗썸 WebSocket URL
        WEBSOCKET_URL = "wss://pubwss.bithumb.com/pub/ws"

        print(f"\n[STEP 1] Connecting to: {WEBSOCKET_URL}")

        # WebSocket 연결
        websocket = await connect(
            WEBSOCKET_URL,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10
        )

        print("[SUCCESS] WebSocket connected!")

        # 구독 메시지 전송 (빗썸 API 문서 기준)
        print("\n[STEP 2] Sending subscription message...")

        # 빗썸 WebSocket 구독 메시지 형태
        subscription_message = {
            "type": "ticker",
            "symbols": ["BTC_KRW", "ETH_KRW"]
        }

        await websocket.send(json.dumps(subscription_message))
        print(f"[SUCCESS] Subscription sent: {subscription_message}")

        # 메시지 수신 테스트
        print("\n[STEP 3] Listening for messages (30 seconds)...")

        message_count = 0
        start_time = asyncio.get_event_loop().time()

        try:
            while (asyncio.get_event_loop().time() - start_time) < 30:
                # 타임아웃과 함께 메시지 받기
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    message_count += 1

                    print(f"\n[MESSAGE {message_count}] Received:")

                    try:
                        # JSON 파싱 시도
                        parsed = json.loads(message)
                        print(f"  Type: {parsed.get('type', 'unknown')}")

                        # 내용 요약
                        if 'content' in parsed:
                            content = parsed['content']
                            if isinstance(content, dict):
                                print(f"  Content keys: {list(content.keys())}")
                                # 가격 정보가 있으면 출력
                                for key in ['closePrice', 'closing_price', 'price']:
                                    if key in content:
                                        print(f"  Price ({key}): {content[key]}")
                                        break
                    except json.JSONDecodeError:
                        print(f"  Raw message: {message[:200]}...")

                    # 처음 5개 메시지만 자세히 출력
                    if message_count >= 5:
                        print("  (continuing to receive messages...)")

                except asyncio.TimeoutError:
                    print("  [INFO] No message received in 5 seconds, continuing...")
                    continue

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] User stopped the test")

        print(f"\n[COMPLETED] Received {message_count} messages in total")

        if message_count > 0:
            print("[SUCCESS] WebSocket is working and receiving data!")
            return True
        else:
            print("[WARNING] No messages received - check subscription format")
            return False

    except Exception as e:
        print(f"\n[ERROR] WebSocket test failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

    finally:
        if websocket:
            try:
                await websocket.close()
                print("\n[INFO] WebSocket connection closed")
            except:
                pass


if __name__ == "__main__":
    try:
        result = asyncio.run(test_bithumb_websocket())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        sys.exit(1)