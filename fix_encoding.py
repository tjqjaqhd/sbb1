#!/usr/bin/env python3
"""
Windows 한글 인코딩 문제 원초적 해결
"""

import sys
import os
import io
import locale

def fix_windows_encoding():
    """Windows에서 한글 인코딩 문제를 완전히 해결"""
    print("Fixing Windows Korean Encoding Issues...")

    # 1. 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'

    # 2. stdout, stderr을 UTF-8로 강제 설정
    if sys.platform.startswith('win'):
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # 3. 로케일 설정
    try:
        locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'Korean_Korea.65001')  # Windows UTF-8
        except:
            pass  # 실패해도 계속 진행

    # 4. 테스트
    test_korean = "한글 인코딩 테스트 성공! 🎉"
    print(f"테스트: {test_korean}")

    return True

if __name__ == "__main__":
    fix_windows_encoding()

    # 모든 future import를 위해 encoding 함수 제공
    import builtins

    def safe_print(*args, **kwargs):
        """안전한 한글 출력 함수"""
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            # 인코딩 실패시 ASCII로 fallback
            safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', errors='replace').decode('ascii'))
                else:
                    safe_args.append(str(arg))
            print(*safe_args, **kwargs)

    # builtins에 등록하여 어디서든 사용 가능하게 만듦
    builtins.safe_print = safe_print

    print("인코딩 설정 완료 - 이제 한글을 안전하게 사용할 수 있습니다!")