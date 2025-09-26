#!/usr/bin/env python3
"""
메시지 파서 디버깅
"""

# 인코딩 문제 해결
import sys
import os
import io
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win') and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json

def debug_message_parsing():
    """메시지 파싱 디버깅"""
    print("🐛 메시지 파서 디버깅")
    print("=" * 30)

    # 실제 빗썸 메시지 시뮬레이션
    sample_ticker = {
        "type": "ticker",
        "content": {
            "symbol": "BTC_KRW",
            "tickType": "24H",
            "date": "20241225",
            "time": "153030",
            "openPrice": "50000000",
            "closePrice": "51000000",
            "lowPrice": "49500000",
            "highPrice": "52000000",
            "value": "12345678901234",
            "volume": "123.456789",
            "sellVolume": "60.123",
            "buyVolume": "63.333"
        }
    }

    try:
        from src.api.bithumb.message_parser import MessageParser

        print("1. MessageParser 초기화...")
        parser = MessageParser(strict_mode=True)  # 엄격 모드로 오류 확인

        print("2. 메시지 파싱 시도...")
        message_str = json.dumps(sample_ticker)
        print(f"   원본 메시지: {message_str[:100]}...")

        parsed = parser.parse_message(message_str)

        if parsed:
            print(f"   ✅ 파싱 성공: {type(parsed)}")
            print(f"   메시지 타입: {getattr(parsed, 'message_type', 'N/A')}")
            print(f"   데이터: {getattr(parsed, 'data', 'N/A')}")
        else:
            print("   ❌ 파싱 실패 - None 반환")

        print("\n3. 파서 통계:")
        stats = parser.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

    except Exception as e:
        print(f"   💥 예외 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_message_parsing()