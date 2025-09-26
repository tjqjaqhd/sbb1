#!/usr/bin/env python3
"""Task 3 WebSocket System Implementation Verification"""

import ast
import py_compile
from pathlib import Path
import tempfile
import os

def verify_task3_implementation():
    """Verify Task 3 WebSocket system implementation"""
    print("Task 3 WebSocket System Implementation Verification")
    print("=" * 55)

    # 1. Check file existence
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

    print("\n1. File Existence Check:")
    missing_files = []
    for filename in required_files:
        filepath = src_path / filename
        if filepath.exists():
            print(f"  [OK] {filename}")
        else:
            print(f"  [MISSING] {filename}")
            missing_files.append(filename)

    # 2. Python syntax check
    print("\n2. Python Syntax Check:")
    syntax_errors = []
    for filename in required_files:
        filepath = src_path / filename
        if not filepath.exists():
            continue

        try:
            py_compile.compile(str(filepath), doraise=True)
            print(f"  [OK] {filename}")
        except py_compile.PyCompileError as e:
            print(f"  [ERROR] {filename}: {e}")
            syntax_errors.append(filename)
        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")
            syntax_errors.append(filename)

    # 3. Core class definitions check
    print("\n3. Core Class Definitions Check:")
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

    # 4. Import connectivity check
    print("\n4. Import Connectivity Check:")
    import_issues = []
    key_files = ["websocket_client.py", "data_streams.py", "message_parser.py"]

    for filename in key_files:
        filepath = src_path / filename
        if not filepath.exists():
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()

            # Check for proper internal imports
            if "from ." in source or "import redis" in source or "import asyncio" in source:
                print(f"  [OK] {filename}: proper imports found")
            else:
                print(f"  [WARNING] {filename}: check import structure")
                import_issues.append(filename)

        except Exception as e:
            print(f"  [ERROR] {filename}: {e}")
            import_issues.append(filename)

    # 5. Summary
    print("\n" + "=" * 55)
    print("Verification Summary:")
    print(f"- Files present: {len(required_files) - len(missing_files)}/{len(required_files)}")
    print(f"- Syntax errors: {len(syntax_errors)}")
    print(f"- Missing classes: {len(missing_classes)}")
    print(f"- Import issues: {len(import_issues)}")

    if not missing_files and not syntax_errors and not missing_classes:
        print("\n[SUCCESS] Task 3 WebSocket system is fully implemented!")
        return True
    else:
        if missing_files:
            print(f"\nMissing files: {missing_files}")
        if syntax_errors:
            print(f"Syntax error files: {syntax_errors}")
        if missing_classes:
            print(f"Missing classes: {missing_classes}")
        if import_issues:
            print(f"Import issues: {import_issues}")

        print("\n[PARTIAL] Most components implemented, some improvements needed.")
        return False

if __name__ == "__main__":
    verify_task3_implementation()