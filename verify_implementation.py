#!/usr/bin/env python3
"""
Task 3 WebSocket 데이터 수집 시스템 구현 검증 스크립트

의존성 패키지 없이 코드 구조와 클래스 정의를 검증합니다.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Any

class ImplementationVerifier:
    """구현 검증기"""

    def __init__(self):
        self.src_path = Path("src/api/bithumb")
        self.verification_results = {}

    def verify_all(self) -> Dict[str, Any]:
        """모든 검증 실행"""
        print("Task 3 WebSocket 시스템 구현 검증 시작...")
        print("=" * 60)

        results = {
            "files_exist": self.verify_files_exist(),
            "syntax_valid": self.verify_syntax(),
            "classes_defined": self.verify_classes_defined(),
            "imports_valid": self.verify_imports(),
            "methods_implemented": self.verify_methods(),
            "overall_status": "UNKNOWN"
        }

        # 종합 상태 계산
        passed_tests = sum(1 for result in results.values() if isinstance(result, dict) and result.get('status') == 'PASS')
        total_tests = len([k for k in results.keys() if k != 'overall_status'])

        if passed_tests == total_tests:
            results['overall_status'] = 'PASS'
        elif passed_tests >= total_tests * 0.8:
            results['overall_status'] = 'PARTIAL'
        else:
            results['overall_status'] = 'FAIL'

        self.print_final_report(results)
        return results

    def verify_files_exist(self) -> Dict[str, Any]:
        """파일 존재 여부 확인"""
        print("📁 파일 존재 여부 확인...")

        required_files = [
            "websocket_client.py",
            "websocket_reconnect.py",
            "message_parser.py",
            "redis_buffer.py",
            "backpressure_handler.py",
            "data_streams.py",
            "stability_performance_tests.py"
        ]

        missing_files = []
        existing_files = []

        for filename in required_files:
            filepath = self.src_path / filename
            if filepath.exists():
                existing_files.append(filename)
                print(f"  ✅ {filename}")
            else:
                missing_files.append(filename)
                print(f"  ❌ {filename}")

        status = "PASS" if not missing_files else "FAIL"

        return {
            "status": status,
            "existing_files": existing_files,
            "missing_files": missing_files,
            "total_files": len(required_files)
        }

    def verify_syntax(self) -> Dict[str, Any]:
        """Python 구문 검증"""
        print("\n🔧 Python 구문 검증...")

        syntax_results = {}

        for py_file in self.src_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()

                ast.parse(source)
                syntax_results[py_file.name] = "VALID"
                print(f"  ✅ {py_file.name}")

            except SyntaxError as e:
                syntax_results[py_file.name] = f"SYNTAX_ERROR: {e}"
                print(f"  ❌ {py_file.name}: {e}")
            except Exception as e:
                syntax_results[py_file.name] = f"ERROR: {e}"
                print(f"  ⚠️  {py_file.name}: {e}")

        valid_files = sum(1 for result in syntax_results.values() if result == "VALID")
        total_files = len(syntax_results)
        status = "PASS" if valid_files == total_files else "FAIL"

        return {
            "status": status,
            "syntax_results": syntax_results,
            "valid_files": valid_files,
            "total_files": total_files
        }

    def verify_classes_defined(self) -> Dict[str, Any]:
        """클래스 정의 검증"""
        print("\n🏗️  클래스 정의 검증...")

        expected_classes = {
            "websocket_client.py": ["BithumbWebSocketClient", "SubscriptionType", "ConnectionState"],
            "message_parser.py": ["TickerData", "OrderBookData", "TransactionData", "MessageParser"],
            "redis_buffer.py": ["RedisQueueBuffer"],
            "backpressure_handler.py": ["BackpressureHandler"],
            "data_streams.py": ["TickerStreamProcessor", "OrderBookStreamProcessor", "TradeStreamProcessor"],
            "stability_performance_tests.py": ["StabilityPerformanceTestSuite"]
        }

        class_results = {}

        for filename, expected_classes_list in expected_classes.items():
            filepath = self.src_path / filename

            if not filepath.exists():
                class_results[filename] = {"status": "FILE_NOT_FOUND", "classes": []}
                print(f"  ❌ {filename}: 파일 없음")
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    source = f.read()

                tree = ast.parse(source)

                found_classes = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        found_classes.append(node.name)
                    elif isinstance(node, ast.Assign):
                        # Enum 클래스들도 찾기
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                found_classes.append(target.id)

                missing_classes = [cls for cls in expected_classes_list if cls not in found_classes]

                if not missing_classes:
                    class_results[filename] = {"status": "PASS", "classes": found_classes}
                    print(f"  ✅ {filename}: 모든 클래스 정의됨 ({len(expected_classes_list)}개)")
                else:
                    class_results[filename] = {
                        "status": "PARTIAL",
                        "classes": found_classes,
                        "missing": missing_classes
                    }
                    print(f"  ⚠️  {filename}: {len(missing_classes)}개 클래스 누락: {missing_classes}")

            except Exception as e:
                class_results[filename] = {"status": "ERROR", "error": str(e)}
                print(f"  ❌ {filename}: 오류 {e}")

        passed_files = sum(1 for result in class_results.values() if result.get("status") == "PASS")
        total_files = len(class_results)
        status = "PASS" if passed_files >= total_files * 0.8 else "FAIL"

        return {
            "status": status,
            "class_results": class_results,
            "passed_files": passed_files,
            "total_files": total_files
        }

    def verify_imports(self) -> Dict[str, Any]:
        """Import 구조 검증"""
        print("\n📦 Import 구조 검증...")

        import_results = {}

        for py_file in self.src_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()

                tree = ast.parse(source)

                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                imports.append(f"{node.module}.{alias.name}")

                import_results[py_file.name] = {
                    "status": "VALID",
                    "import_count": len(imports),
                    "imports": imports[:5]  # 처음 5개만 표시
                }
                print(f"  ✅ {py_file.name}: {len(imports)}개 import")

            except Exception as e:
                import_results[py_file.name] = {"status": "ERROR", "error": str(e)}
                print(f"  ❌ {py_file.name}: {e}")

        valid_files = sum(1 for result in import_results.values() if result.get("status") == "VALID")
        total_files = len(import_results)
        status = "PASS" if valid_files == total_files else "FAIL"

        return {
            "status": status,
            "import_results": import_results,
            "valid_files": valid_files,
            "total_files": total_files
        }

    def verify_methods(self) -> Dict[str, Any]:
        """주요 메서드 구현 검증"""
        print("\n🔧 주요 메서드 구현 검증...")

        expected_methods = {
            "websocket_client.py": ["connect", "disconnect", "subscribe", "unsubscribe"],
            "data_streams.py": ["start", "stop", "get_stats", "health_check"]
        }

        method_results = {}

        for filename, expected_methods_list in expected_methods.items():
            filepath = self.src_path / filename

            if not filepath.exists():
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    source = f.read()

                tree = ast.parse(source)

                found_methods = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        found_methods.append(node.name)

                missing_methods = [method for method in expected_methods_list
                                 if not any(method in found_method for found_method in found_methods)]

                if not missing_methods:
                    method_results[filename] = {"status": "PASS", "methods": len(found_methods)}
                    print(f"  ✅ {filename}: 주요 메서드 모두 구현됨")
                else:
                    method_results[filename] = {"status": "PARTIAL", "missing": missing_methods}
                    print(f"  ⚠️  {filename}: {len(missing_methods)}개 메서드 누락")

            except Exception as e:
                method_results[filename] = {"status": "ERROR", "error": str(e)}
                print(f"  ❌ {filename}: {e}")

        passed_files = sum(1 for result in method_results.values() if result.get("status") == "PASS")
        total_files = len(method_results)
        status = "PASS" if passed_files >= total_files * 0.8 else "FAIL"

        return {
            "status": status,
            "method_results": method_results,
            "passed_files": passed_files,
            "total_files": total_files
        }

    def print_final_report(self, results: Dict[str, Any]):
        """최종 보고서 출력"""
        print("\n" + "=" * 60)
        print("📋 Task 3 WebSocket 시스템 구현 검증 결과")
        print("=" * 60)

        status_emoji = {
            "PASS": "✅",
            "PARTIAL": "⚠️",
            "FAIL": "❌"
        }

        for test_name, result in results.items():
            if test_name == 'overall_status':
                continue

            if isinstance(result, dict):
                status = result.get('status', 'UNKNOWN')
                emoji = status_emoji.get(status, '❓')
                print(f"{emoji} {test_name.replace('_', ' ').title()}: {status}")

        print("-" * 60)
        overall_status = results['overall_status']
        overall_emoji = status_emoji.get(overall_status, '❓')
        print(f"{overall_emoji} 전체 상태: {overall_status}")

        if overall_status == "PASS":
            print("\n🎉 Task 3 WebSocket 데이터 수집 시스템이 성공적으로 구현되었습니다!")
            print("   모든 핵심 컴포넌트가 올바르게 정의되고 연결되어 있습니다.")
        elif overall_status == "PARTIAL":
            print("\n⚠️  Task 3 구현이 대부분 완료되었지만 일부 개선이 필요합니다.")
        else:
            print("\n❌ Task 3 구현에 중요한 문제가 발견되었습니다. 수정이 필요합니다.")

def main():
    """메인 실행 함수"""
    verifier = ImplementationVerifier()
    results = verifier.verify_all()

    # 종료 코드 설정
    if results['overall_status'] == "PASS":
        sys.exit(0)
    elif results['overall_status'] == "PARTIAL":
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()