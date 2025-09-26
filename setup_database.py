"""
PostgreSQL 데이터베이스 및 사용자 자동 생성 스크립트
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("[INFO] Installing psycopg2...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def create_database_and_user():
    """데이터베이스와 사용자 생성"""
    print("=" * 60)
    print("PostgreSQL Database and User Setup")
    print("=" * 60)

    # PostgreSQL 기본 설정
    postgres_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'postgres',  # 기본 관리자 계정
        'password': '*tj1748426',  # 설치 시 설정한 비밀번호
        'database': 'postgres'  # 기본 데이터베이스
    }

    target_db_name = 'bithumb_trading'
    target_username = 'tjqjaqhd'
    target_password = '*tj1748426'

    try:
        print("[STEP 1] Connecting to PostgreSQL as admin...")

        # 관리자로 연결
        conn = psycopg2.connect(**postgres_config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        print("[SUCCESS] Connected to PostgreSQL!")

        # PostgreSQL 버전 확인
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"[INFO] PostgreSQL Version: {version[:50]}...")

        print(f"\n[STEP 2] Creating database '{target_db_name}'...")

        # 데이터베이스 존재 확인
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db_name,))
        if cursor.fetchone():
            print(f"[INFO] Database '{target_db_name}' already exists.")
        else:
            # 데이터베이스 생성
            cursor.execute(f'CREATE DATABASE "{target_db_name}";')
            print(f"[SUCCESS] Database '{target_db_name}' created!")

        print(f"\n[STEP 3] Creating user '{target_username}'...")

        # 사용자 존재 확인
        cursor.execute("SELECT 1 FROM pg_user WHERE usename = %s", (target_username,))
        if cursor.fetchone():
            print(f"[INFO] User '{target_username}' already exists.")
            # 비밀번호 업데이트
            cursor.execute(f'ALTER USER "{target_username}" WITH ENCRYPTED PASSWORD %s;', (target_password,))
            print(f"[INFO] Password updated for user '{target_username}'.")
        else:
            # 사용자 생성
            cursor.execute(f'CREATE USER "{target_username}" WITH ENCRYPTED PASSWORD %s;', (target_password,))
            print(f"[SUCCESS] User '{target_username}' created!")

        print(f"\n[STEP 4] Granting privileges...")

        # 권한 부여
        cursor.execute(f'GRANT ALL PRIVILEGES ON DATABASE "{target_db_name}" TO "{target_username}";')
        cursor.execute(f'ALTER USER "{target_username}" CREATEDB;')
        print(f"[SUCCESS] All privileges granted to '{target_username}'!")

        # 연결 종료
        cursor.close()
        conn.close()

        print(f"\n[STEP 5] Testing new user connection...")

        # 새 사용자로 연결 테스트
        test_config = {
            'host': 'localhost',
            'port': 5432,
            'user': target_username,
            'password': target_password,
            'database': target_db_name
        }

        test_conn = psycopg2.connect(**test_config)
        test_cursor = test_conn.cursor()

        # 간단한 테스트 쿼리
        test_cursor.execute("SELECT 1;")
        result = test_cursor.fetchone()

        if result[0] == 1:
            print("[SUCCESS] New user connection test passed!")

        test_cursor.close()
        test_conn.close()

        print("\n" + "=" * 60)
        print("[SUCCESS] Database setup completed successfully!")
        print("=" * 60)
        print(f"Database: {target_db_name}")
        print(f"Username: {target_username}")
        print(f"Password: {target_password}")
        print(f"Host: localhost")
        print(f"Port: 5432")
        print("=" * 60)

        return True

    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL Error: {e}")
        print(f"[ERROR] Error Code: {e.pgcode}")

        # 일반적인 에러 해결책 제공
        if "authentication failed" in str(e):
            print("\n[SOLUTION] Authentication failed!")
            print("  1. Check if PostgreSQL admin password is '*tj1748426'")
            print("  2. Try connecting with pgAdmin to verify credentials")
        elif "connection refused" in str(e):
            print("\n[SOLUTION] Connection refused!")
            print("  1. Check if PostgreSQL service is running")
            print("  2. Check if port 5432 is accessible")

        return False

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


if __name__ == "__main__":
    result = create_database_and_user()
    sys.exit(0 if result else 1)