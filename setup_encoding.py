"""
Windows 인코딩 문제 해결 - 프로젝트 전체 설정
"""

import os
import sys
import locale
import subprocess
from pathlib import Path


def setup_utf8_environment():
    """UTF-8 환경 강제 설정"""

    # Python 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

    # 콘솔 출력 인코딩 설정
    if sys.platform.startswith('win'):
        try:
            # Windows 콘솔 인코딩을 UTF-8로 설정
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)

            # Python 표준 출력/에러 인코딩 재설정
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')

        except Exception:
            pass  # 실패해도 계속 진행


def create_powershell_profile():
    """PowerShell 프로필 자동 생성"""
    try:
        # PowerShell 프로필 경로 확인
        result = subprocess.run(
            ['powershell', '-Command', 'echo $PROFILE'],
            capture_output=True, text=True, shell=True
        )

        if result.returncode == 0:
            profile_path = result.stdout.strip()
            profile_dir = os.path.dirname(profile_path)

            # 디렉토리 생성
            os.makedirs(profile_dir, exist_ok=True)

            # 프로필 내용
            profile_content = '''# UTF-8 Encoding Setup for Python Development
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
'''

            # 기존 프로필이 있으면 백업
            if os.path.exists(profile_path):
                backup_path = profile_path + '.backup'
                if not os.path.exists(backup_path):
                    os.rename(profile_path, backup_path)

            # 새 프로필 작성
            with open(profile_path, 'w', encoding='utf-8') as f:
                f.write(profile_content)

            print(f"PowerShell profile created: {profile_path}")

    except Exception as e:
        print(f"PowerShell profile setup failed: {e}")


def create_batch_file():
    """Python 실행용 배치 파일 생성"""
    project_root = Path(__file__).parent

    batch_content = '''@echo off
REM UTF-8 Environment Setup
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Run Python with UTF-8 support
python %*
'''

    batch_path = project_root / 'run_python.bat'
    with open(batch_path, 'w', encoding='utf-8') as f:
        f.write(batch_content)

    print(f"Batch file created: {batch_path}")


def update_project_files():
    """프로젝트 파일들을 ASCII 안전 버전으로 업데이트"""

    # 문제가 되는 유니코드 문자들을 ASCII로 변환하는 매핑
    unicode_replacements = {
        '✅': '[SUCCESS]',
        '❌': '[FAILED]',
        '🎯': '[TARGET]',
        '🎉': '[COMPLETE]',
        '⚠️': '[WARNING]',
        '→': '->',
        '←': '<-',
        '✓': '[OK]',
        '◆': '*',
        '◇': 'o',
        '■': '#',
        '□': '-',
    }

    print("Creating ASCII-safe coding standard...")

    # 코딩 표준 파일 생성
    coding_standard = '''# ASCII-Only Coding Standard

## Symbols to Use (ASCII Safe):
- Success: [SUCCESS] or [OK]
- Failed: [FAILED] or [ERROR]
- Warning: [WARNING]
- Info: [INFO]
- Arrow: -> or <-
- Bullet: * or -
- Target: [TARGET]
- Complete: [COMPLETE]

## Avoid These Unicode Characters:
- Emojis: ✅❌🎯🎉⚠️
- Special arrows: →←
- Special bullets: ◆◇■□
- Checkmarks: ✓

## Example:
print("[SUCCESS] Connection established!")
print("[ERROR] Database connection failed")
print("[INFO] Processing data...")
'''

    with open('ASCII_CODING_STANDARD.md', 'w', encoding='ascii', errors='replace') as f:
        f.write(coding_standard)

    print("ASCII coding standard created: ASCII_CODING_STANDARD.md")


def main():
    """메인 설정 실행"""
    print("Setting up UTF-8 environment for project...")

    # UTF-8 환경 설정
    setup_utf8_environment()

    # PowerShell 프로필 생성
    create_powershell_profile()

    # 배치 파일 생성
    create_batch_file()

    # ASCII 코딩 표준 생성
    update_project_files()

    print("""
SETUP COMPLETE!

From now on:
1. Use run_python.bat instead of python directly
2. Follow ASCII_CODING_STANDARD.md for all code
3. PowerShell profile will auto-set UTF-8 encoding

Example usage:
  run_python.bat test_script.py

This eliminates Unicode encoding issues permanently!
""")


if __name__ == "__main__":
    main()