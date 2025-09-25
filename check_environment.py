#!/usr/bin/env python3
"""
빗썸 자동매매 시스템 환경 검증 스크립트

프로젝트 환경이 올바르게 설정되었는지 확인합니다.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Python 버전 확인"""
    print("🐍 Python 버전 확인...")
    if sys.version_info >= (3, 11):
        print(f"   ✅ Python {sys.version.split()[0]} (요구사항 충족)")
        return True
    else:
        print(f"   ❌ Python {sys.version.split()[0]} (Python 3.11+ 필요)")
        return False


def check_packages():
    """필수 패키지 import 확인"""
    print("📦 패키지 import 확인...")

    required_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("aiohttp", "aiohttp"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("redis", "Redis"),
        ("decouple", "python-decouple"),
        ("sqlalchemy", "SQLAlchemy"),
        ("asyncpg", "asyncpg"),
    ]

    failed_packages = []

    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            failed_packages.append(name)

    if failed_packages:
        print(f"\n   ⚠️  누락된 패키지들: {', '.join(failed_packages)}")
        print(f"   💡 해결방법: pip install -r requirements.txt")
        return False

    return True


def check_project_structure():
    """프로젝트 구조 확인"""
    print("📁 프로젝트 구조 확인...")

    project_root = Path(__file__).parent

    required_items = [
        ("src/", "소스 코드 디렉토리"),
        ("tests/", "테스트 디렉토리"),
        ("config/", "설정 디렉토리"),
        ("logs/", "로그 디렉토리"),
        ("data/", "데이터 디렉토리"),
        ("venv/", "가상환경 디렉토리"),
        ("main.py", "메인 실행 파일"),
        ("requirements.txt", "의존성 패키지 목록"),
        ("pyproject.toml", "프로젝트 설정 파일"),
        (".env.example", "환경변수 예시 파일"),
        ("config/config.py", "설정 모듈"),
    ]

    missing_items = []

    for item, description in required_items:
        item_path = project_root / item
        if item_path.exists():
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description}")
            missing_items.append(item)

    if missing_items:
        print(f"\n   ⚠️  누락된 항목들: {', '.join(missing_items)}")
        return False

    return True


def check_config():
    """설정 모듈 확인"""
    print("⚙️  설정 모듈 확인...")

    try:
        from config.config import settings, validate_settings

        print(f"   ✅ 앱 이름: {settings.APP_NAME}")
        print(f"   ✅ 앱 버전: {settings.APP_VERSION}")
        print(f"   ✅ 디버그 모드: {settings.DEBUG}")
        print(f"   ✅ API 설정됨: {settings.is_api_configured()}")
        print(f"   ✅ 알림 설정됨: {settings.is_notification_configured()}")

        # 설정 검증
        warnings = validate_settings()
        if warnings:
            print("\n   ⚠️  설정 경고:")
            for warning in warnings:
                print(f"      - {warning}")

        return True

    except Exception as e:
        print(f"   ❌ 설정 모듈 로드 실패: {e}")
        return False


def check_main_module():
    """메인 모듈 확인"""
    print("🚀 메인 모듈 확인...")

    try:
        import main
        if hasattr(main, 'main'):
            print("   ✅ 메인 함수 존재")
            return True
        else:
            print("   ❌ 메인 함수 없음")
            return False
    except Exception as e:
        print(f"   ❌ 메인 모듈 import 실패: {e}")
        return False


def run_basic_test():
    """기본 테스트 실행"""
    print("🧪 기본 테스트 실행...")

    try:
        # pytest가 설치되어 있는지 확인
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("   ✅ pytest 설치됨")

            # 환경 테스트 실행
            test_result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_environment.py", "-v"],
                capture_output=True,
                text=True
            )

            if test_result.returncode == 0:
                print("   ✅ 환경 테스트 통과")
                return True
            else:
                print("   ❌ 환경 테스트 실패")
                print(f"   오류: {test_result.stderr}")
                return False
        else:
            print("   ❌ pytest 미설치")
            return False

    except Exception as e:
        print(f"   ❌ 테스트 실행 실패: {e}")
        return False


def main():
    """메인 함수"""
    print("=" * 60)
    print("🔍 빗썸 자동매매 시스템 환경 검증")
    print("=" * 60)

    checks = [
        check_python_version,
        check_project_structure,
        check_packages,
        check_config,
        check_main_module,
        run_basic_test,
    ]

    passed = 0
    total = len(checks)

    for check in checks:
        print()
        if check():
            passed += 1
        print("-" * 40)

    print()
    print("=" * 60)
    print("📊 검증 결과")
    print("=" * 60)

    if passed == total:
        print(f"🎉 모든 검증 통과! ({passed}/{total})")
        print("✨ 빗썸 자동매매 시스템 환경이 올바르게 설정되었습니다!")
        print()
        print("다음 단계:")
        print("1. .env 파일을 생성하고 API 키를 설정하세요")
        print("2. python main.py 명령어로 애플리케이션을 실행하세요")
    else:
        print(f"⚠️  일부 검증 실패 ({passed}/{total})")
        print("❌ 위의 오류들을 수정한 후 다시 실행해주세요")
        sys.exit(1)


if __name__ == "__main__":
    main()