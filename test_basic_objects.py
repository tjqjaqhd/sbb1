#!/usr/bin/env python3
"""기본 객체 생성 테스트"""

import asyncio
import traceback

async def test_basic_objects():
    """핵심 클래스들의 기본 객체 생성 테스트"""
    print("Basic Object Creation Test")
    print("=" * 35)

    tests = []

    # 1. MessageParser 테스트
    try:
        from src.api.bithumb.message_parser import MessageParser
        parser = MessageParser()
        print("  [OK] MessageParser 생성 성공")
        tests.append(("MessageParser", True))
    except Exception as e:
        print(f"  [FAILED] MessageParser: {e}")
        tests.append(("MessageParser", False))

    # 2. TickerData 테스트
    try:
        from src.api.bithumb.message_parser import TickerData
        ticker = TickerData(symbol="BTC_KRW", closing_price=50000000)
        print("  [OK] TickerData 생성 성공")
        tests.append(("TickerData", True))
    except Exception as e:
        print(f"  [FAILED] TickerData: {e}")
        tests.append(("TickerData", False))

    # 3. BackpressureHandler 테스트
    try:
        from src.api.bithumb.backpressure_handler import BackpressureHandler
        handler = BackpressureHandler()
        print("  [OK] BackpressureHandler 생성 성공")
        tests.append(("BackpressureHandler", True))
    except Exception as e:
        print(f"  [FAILED] BackpressureHandler: {e}")
        tests.append(("BackpressureHandler", False))

    # 4. BithumbWebSocketClient 테스트 (연결하지 않고 객체만 생성)
    try:
        from src.api.bithumb.websocket_client import BithumbWebSocketClient
        client = BithumbWebSocketClient()
        print("  [OK] BithumbWebSocketClient 생성 성공")
        tests.append(("BithumbWebSocketClient", True))
    except Exception as e:
        print(f"  [FAILED] BithumbWebSocketClient: {e}")
        tests.append(("BithumbWebSocketClient", False))

    # 5. RedisQueueBuffer 테스트 (연결하지 않고 객체만 생성)
    try:
        from src.api.bithumb.redis_buffer import RedisQueueBuffer
        buffer = RedisQueueBuffer("test_queue")
        print("  [OK] RedisQueueBuffer 생성 성공")
        tests.append(("RedisQueueBuffer", True))
    except Exception as e:
        print(f"  [FAILED] RedisQueueBuffer: {e}")
        tests.append(("RedisQueueBuffer", False))

    # 6. TickerStreamProcessor 테스트
    try:
        from src.api.bithumb.data_streams import TickerStreamProcessor
        processor = TickerStreamProcessor(["BTC_KRW"])
        print("  [OK] TickerStreamProcessor 생성 성공")
        tests.append(("TickerStreamProcessor", True))
    except Exception as e:
        print(f"  [FAILED] TickerStreamProcessor: {e}")
        tests.append(("TickerStreamProcessor", False))

    # 결과 요약
    print("\n" + "=" * 35)
    success_count = sum(1 for _, success in tests if success)
    total_count = len(tests)
    print(f"Object Creation Test Results:")
    print(f"Success: {success_count}/{total_count}")

    if success_count == total_count:
        print("All basic objects created successfully!")
        return True
    else:
        failed = [name for name, success in tests if not success]
        print(f"Failed objects: {failed}")
        return False

if __name__ == "__main__":
    asyncio.run(test_basic_objects())