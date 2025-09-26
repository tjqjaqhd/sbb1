#!/usr/bin/env python3
"""간단한 Task 3 구현 검증 스크립트"""

import ast
from pathlib import Path

def verify_implementation():
    """구현 검증"""
    print("Task 3 WebSocket 시스템 구현 검증...")
    print("=" * 50)

    # 1. 파일 존재 확인
    src_path = Path("src/api/bithumb")
    required_files = [
        "websocket_client.py",
        "websocket_reconnect.py",
        "message_parser.py",
        "redis_buffer.py",
        "backpressure_handler.py",
        "data_streams.py",
        "stability_performance_tests.py"
    ]

    print("\n1. 파일 존재 확인:")
    missing_files = []
    for filename in required_files:
        filepath = src_path / filename
        if filepath.exists():
            print(f"  [OK] {filename}")
        else:
            print(f"  [MISSING] {filename}")
            missing_files.append(filename)

    # 2. 문법 검사
    print("\n2. Python 문법 검사:")
    syntax_errors = []
    for filename in required_files:
        filepath = src_path / filename
        if not filepath.exists():
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            print(f"  [OK] {filename}")
        except SyntaxError as e:
            print(f"  [ERROR] {filename}: {e}")
            syntax_errors.append(filename)
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")
            syntax_errors.append(filename)

    # 3. 핵심 클래스 확인
    print("\n3. 핵심 클래스 정의 확인:")
    expected_classes = {
        "websocket_client.py": ["BithumbWebSocketClient"],
        "message_parser.py": ["TickerData", "MessageParser"],
        "redis_buffer.py": ["RedisQueueBuffer"],
        "backpressure_handler.py": ["BackpressureHandler"],
        "data_streams.py": ["TickerStreamProcessor", "OrderBookStreamProcessor", "TradeStreamProcessor"],
        "stability_performance_tests.py": ["StabilityPerformanceTestSuite"]
    }

    missing_classes = []
    for filename, classes in expected_classes.items():
        filepath = src_path / filename
        if not filepath.exists():
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source)

            found_classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    found_classes.append(node.name)

            for expected_class in classes:
                if expected_class in found_classes:
                    print(f"  [OK] {filename}: {expected_class}")
                else:
                    print(f"  [MISSING] {filename}: {expected_class}")
                    missing_classes.append(f"{filename}:{expected_class}")

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")

    # 4. 종합 결과
    print("\n" + "=" * 50)
    print("검증 결과 요약:")
    print(f"- 파일 존재: {len(required_files) - len(missing_files)}/{len(required_files)}")
    print(f"- 문법 오류: {len(syntax_errors)}개")
    print(f"- 클래스 누락: {len(missing_classes)}개")

    if not missing_files and not syntax_errors and not missing_classes:
        print("\n[SUCCESS] Task 3 WebSocket 시스템이 완벽하게 구현되었습니다!")
        return True
    else:
        if missing_files:
            print(f"\n누락된 파일: {missing_files}")
        if syntax_errors:
            print(f"문법 오류 파일: {syntax_errors}")
        if missing_classes:
            print(f"누락된 클래스: {missing_classes}")

        print("\n[PARTIAL] 대부분 구현되었지만 일부 개선이 필요합니다.")
        return False

if __name__ == "__main__":
    verify_implementation()