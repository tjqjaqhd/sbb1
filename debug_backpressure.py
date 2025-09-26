#!/usr/bin/env python3
"""
백프레셔 핸들러 디버깅
"""

# 인코딩 문제 해결
import sys
import os
import io
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win') and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio

async def debug_backpressure_handler():
    """백프레셔 핸들러 디버깅"""
    print("🐛 백프레셔 핸들러 디버깅")
    print("=" * 35)

    try:
        from src.api.bithumb.backpressure_handler import BackpressureHandler

        print("1. BackpressureHandler 초기화...")
        handler = BackpressureHandler()

        print("2. 핸들러 시작...")
        await handler.start()

        print("3. 2초 대기 (메트릭 수집 기다림)...")
        await asyncio.sleep(2.0)

        print("4. 현재 메트릭 확인...")
        metrics = handler.get_current_metrics()
        print(f"   메트릭: {metrics}")

        print("5. 통계 확인...")
        stats = handler.get_stats()
        print(f"   통계: {stats}")

        print("6. 메트릭 히스토리 확인...")
        history = handler.get_metrics_history()
        print(f"   히스토리 개수: {len(history) if history else 0}")

        if metrics:
            print(f"   메트릭 레벨: {getattr(metrics, 'level', 'N/A')}")
            print(f"   메트릭 타임스탬프: {getattr(metrics, 'timestamp', 'N/A')}")

        print("7. 핸들러 중지...")
        await handler.stop()

        print("8. 테스트 결과 판정...")
        if metrics or (stats and 'total_updates' in stats):
            print("   ✅ 백프레셔 핸들러 테스트 통과")
            return True
        else:
            print("   ❌ 백프레셔 핸들러 테스트 실패")
            return False

    except Exception as e:
        print(f"   💥 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(debug_backpressure_handler())