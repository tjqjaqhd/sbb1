"""
PostgreSQL 사용자 권한 추가 설정
"""

import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def fix_database_permissions():
    """데이터베이스 사용자 권한 수정"""
    print("=" * 60)
    print("PostgreSQL User Permissions Fix")
    print("=" * 60)

    # PostgreSQL 기본 설정
    postgres_config = {
        'host': 'localhost',
        'port': 5432,
        'user': 'postgres',  # 기본 관리자 계정
        'password': '*tj1748426',
        'database': 'bithumb_trading'  # 타겟 데이터베이스
    }

    target_username = 'tjqjaqhd'

    try:
        print("[STEP 1] Connecting to target database as admin...")

        # 타겟 데이터베이스에 관리자로 연결
        conn = psycopg2.connect(**postgres_config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        print("[SUCCESS] Connected to bithumb_trading database!")

        print(f"\n[STEP 2] Granting schema permissions to '{target_username}'...")

        # public 스키마에 대한 모든 권한 부여
        cursor.execute(f'GRANT ALL ON SCHEMA public TO "{target_username}";')
        print("[SUCCESS] Schema permissions granted!")

        # public 스키마에서 테이블 생성 권한 부여
        cursor.execute(f'GRANT CREATE ON SCHEMA public TO "{target_username}";')
        print("[SUCCESS] CREATE permission granted!")

        # 기존 테이블들에 대한 권한 (만약 있다면)
        cursor.execute(f'GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "{target_username}";')
        print("[SUCCESS] All table privileges granted!")

        # 시퀀스에 대한 권한
        cursor.execute(f'GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "{target_username}";')
        print("[SUCCESS] All sequence privileges granted!")

        # 기본 권한 설정 (향후 생성되는 객체들에 대해)
        cursor.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO "{target_username}";')
        cursor.execute(f'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO "{target_username}";')
        print("[SUCCESS] Default privileges set!")

        print(f"\n[STEP 3] Testing permissions...")

        # 권한 테스트를 위해 사용자 계정으로 연결
        test_config = {
            'host': 'localhost',
            'port': 5432,
            'user': target_username,
            'password': '*tj1748426',
            'database': 'bithumb_trading'
        }

        test_conn = psycopg2.connect(**test_config)
        test_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        test_cursor = test_conn.cursor()

        # 간단한 테이블 생성 테스트
        test_cursor.execute("""
            CREATE TABLE IF NOT EXISTS permission_test (
                id SERIAL PRIMARY KEY,
                test_value TEXT
            );
        """)

        # 데이터 삽입 테스트
        test_cursor.execute("INSERT INTO permission_test (test_value) VALUES ('Permission test successful!');")

        # 데이터 조회 테스트
        test_cursor.execute("SELECT test_value FROM permission_test LIMIT 1;")
        result = test_cursor.fetchone()

        if result:
            print(f"[SUCCESS] Permission test passed: {result[0]}")

        # 테스트 테이블 삭제
        test_cursor.execute("DROP TABLE permission_test;")

        test_cursor.close()
        test_conn.close()

        cursor.close()
        conn.close()

        print("\n" + "=" * 60)
        print("[SUCCESS] All permissions fixed successfully!")
        print("User 'tjqjaqhd' now has full access to create tables and objects.")
        print("=" * 60)

        return True

    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL Error: {e}")
        return False

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


if __name__ == "__main__":
    result = fix_database_permissions()
    sys.exit(0 if result else 1)