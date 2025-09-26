#!/usr/bin/env python3
"""Simple object creation test without logging issues"""

import asyncio
import sys
import io

# Set UTF-8 encoding for stdout
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

async def test_objects():
    """Test basic object creation"""
    print("Simple Object Test")
    print("==================")

    results = []

    # Test 1: MessageParser
    try:
        from src.api.bithumb.message_parser import MessageParser
        parser = MessageParser(strict_mode=False)  # Non-strict mode to avoid issues
        print("[OK] MessageParser")
        results.append(True)
    except Exception as e:
        print(f"[FAIL] MessageParser: {e}")
        results.append(False)

    # Test 2: TickerData
    try:
        from src.api.bithumb.message_parser import TickerData
        ticker = TickerData(symbol="BTC_KRW")
        print("[OK] TickerData")
        results.append(True)
    except Exception as e:
        print(f"[FAIL] TickerData: {e}")
        results.append(False)

    # Test 3: BackpressureHandler (without logging)
    try:
        from src.api.bithumb.backpressure_handler import BackpressureHandler
        # Mock logger to avoid encoding issues
        import logging
        logging.disable(logging.CRITICAL)

        handler = BackpressureHandler(
            memory_threshold_mb=1000.0,
            cpu_threshold_percent=80.0
        )
        print("[OK] BackpressureHandler")
        results.append(True)
    except Exception as e:
        print(f"[FAIL] BackpressureHandler: {e}")
        results.append(False)

    # Test 4: WebSocket Client (basic)
    try:
        from src.api.bithumb.websocket_client import BithumbWebSocketClient
        client = BithumbWebSocketClient()
        print("[OK] WebSocketClient")
        results.append(True)
    except Exception as e:
        print(f"[FAIL] WebSocketClient: {e}")
        results.append(False)

    # Test 5: Stream processors
    try:
        from src.api.bithumb.data_streams import TickerStreamProcessor
        processor = TickerStreamProcessor(symbols=["BTC_KRW"], cache_ttl_seconds=60)
        print("[OK] TickerStreamProcessor")
        results.append(True)
    except Exception as e:
        print(f"[FAIL] TickerStreamProcessor: {e}")
        results.append(False)

    print("==================")
    success_count = sum(results)
    total_count = len(results)
    print(f"Results: {success_count}/{total_count} passed")

    return success_count == total_count

if __name__ == "__main__":
    asyncio.run(test_objects())