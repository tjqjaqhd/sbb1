"""
환경 설정 및 기본 import 테스트

프로젝트 환경이 올바르게 설정되었는지 확인하는 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEnvironment:
    """환경 설정 테스트 클래스"""

    def test_python_version(self):
        """Python 버전이 3.11 이상인지 확인"""
        assert sys.version_info >= (3, 11), f"Python 3.11+ 필요, 현재: {sys.version}"

    def test_basic_imports(self):
        """기본 패키지들이 정상적으로 import 되는지 확인"""
        try:
            import fastapi
            import uvicorn
            import aiohttp
            import pandas
            import numpy
            import redis
            from decouple import config
        except ImportError as e:
            pytest.fail(f"필수 패키지 import 실패: {e}")

    def test_project_structure(self):
        """프로젝트 디렉토리 구조가 올바른지 확인"""
        project_root = Path(__file__).parent.parent

        required_dirs = ["src", "tests", "config", "logs", "data", "venv"]
        required_files = ["main.py", "requirements.txt", "pyproject.toml", ".env.example"]

        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"디렉토리가 없습니다: {dir_path}"

        for file_name in required_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"파일이 없습니다: {file_path}"

    def test_config_module(self):
        """설정 모듈이 정상적으로 로드되는지 확인"""
        try:
            from config.config import settings, get_settings

            # 기본 설정 확인
            assert settings.APP_NAME is not None
            assert settings.APP_VERSION is not None

            # settings 인스턴스 확인
            config_instance = get_settings()
            assert config_instance is not None

        except Exception as e:
            pytest.fail(f"설정 모듈 로드 실패: {e}")

    def test_main_module(self):
        """메인 모듈이 정상적으로 import 되는지 확인"""
        try:
            import main
            assert hasattr(main, 'main'), "main 함수가 없습니다"
        except Exception as e:
            pytest.fail(f"메인 모듈 import 실패: {e}")


if __name__ == "__main__":
    # 테스트 실행 시 결과 출력
    pytest.main([__file__, "-v"])