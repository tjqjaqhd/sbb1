#!/usr/bin/env python3
"""실제 import 동작 테스트"""

import sys
import traceback

def test_imports():
    """모든 구현 모듈의 실제 import 테스트"""
    print("Real Import Test Started")
    print("=" * 40)

    modules_to_test = [
        "src.api.bithumb.websocket_client",
        "src.api.bithumb.websocket_reconnect",
        "src.api.bithumb.message_parser",
        "src.api.bithumb.redis_buffer",
        "src.api.bithumb.backpressure_handler",
        "src.api.bithumb.data_streams",
        "src.api.bithumb.stability_performance_tests"
    ]

    failed_imports = []

    for module_name in modules_to_test:
        try:
            print(f"Testing import: {module_name}")
            __import__(module_name)
            print(f"  [OK] {module_name}")
        except Exception as e:
            print(f"  [FAILED] {module_name}: {e}")
            print(f"    Error: {traceback.format_exc().split('Traceback')[-1].strip()}")
            failed_imports.append((module_name, str(e)))

    print("\n" + "=" * 40)
    print("Import Test Results:")
    print(f"Success: {len(modules_to_test) - len(failed_imports)}/{len(modules_to_test)}")
    print(f"Failed: {len(failed_imports)}")

    if failed_imports:
        print("\nFailed Imports:")
        for module, error in failed_imports:
            print(f"  - {module}: {error}")
        return False
    else:
        print("\nAll imports successful!")
        return True

if __name__ == "__main__":
    test_imports()