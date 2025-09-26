"""
PostgreSQL 설치 후 전체 테스트 스크립트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.config import init_database, close_database
from src.database.models import Base
from config.config import get_settings


async def complete_database_test():
    """완전한 데이터베이스 테스트"""
    print("=" * 60)
    print("Complete Database Test - After PostgreSQL Installation")
    print("=" * 60)

    try:
        # 1. 설정 확인
        settings = get_settings()
        print(f"[INFO] Database URL: {settings.DATABASE_URL}")
        print(f"[INFO] DB User: {settings.DB_USER}")
        print(f"[INFO] DB Name: {settings.DB_NAME}")

        # 2. 데이터베이스 연결
        print("\n[STEP 1] Connecting to database...")
        db_config = await init_database()
        print("[SUCCESS] Database connection established!")

        # 3. 헬스 체크
        print("\n[STEP 2] Running health check...")
        is_healthy = await db_config.health_check()

        if not is_healthy:
            print("[ERROR] Health check failed!")
            return False

        print("[SUCCESS] Database health check passed!")

        # 4. 테이블 생성
        print("\n[STEP 3] Creating database tables...")

        # 동기 엔진으로 테이블 생성
        from sqlalchemy import create_engine
        sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
        sync_engine = create_engine(sync_url)

        Base.metadata.create_all(sync_engine)
        print("[SUCCESS] All tables created successfully!")

        async with db_config.get_session() as session:

            # 5. PostgreSQL 버전 확인
            from sqlalchemy import text
            result = await session.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"[INFO] PostgreSQL Version: {version[:50]}...")

            # 6. 테이블 목록 확인
            result = await session.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            tables = [row[0] for row in result.fetchall()]

            print(f"\n[INFO] Created tables ({len(tables)}):")
            for table in tables:
                print(f"  - {table}")

            # 7. 간단한 데이터 삽입 테스트
            print("\n[STEP 4] Testing data insertion...")
            from src.database.models.user import UserSetting

            # 테스트 설정 데이터 삽입
            test_setting = UserSetting(
                category="system",
                key="test_connection",
                value="Database connection successful!",
                description="Connection test setting"
            )

            session.add(test_setting)
            await session.commit()
            print("[SUCCESS] Test data inserted successfully!")

            # 8. 데이터 조회 테스트
            print("\n[STEP 5] Testing data retrieval...")
            result = await session.execute(
                text("SELECT key, value FROM user_settings WHERE key = 'test_connection'")
            )
            row = result.fetchone()

            if row:
                print(f"[SUCCESS] Retrieved data: {row[0]} = {row[1]}")
            else:
                print("[WARNING] No data found!")

        # 9. 연결 종료
        await close_database()
        print("\n[STEP 6] Database connection closed successfully.")

        print("\n" + "=" * 60)
        print("[SUCCESS] All database tests completed successfully!")
        print("Task 4 - Database Schema Design and Implementation: COMPLETED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[ERROR] Database test failed: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")

        # 상세 에러 분석
        error_str = str(e).lower()
        if "connection refused" in error_str:
            print("\n[SOLUTION] PostgreSQL server is not running!")
            print("  - Check if PostgreSQL service is started")
            print("  - Run: services.msc -> find PostgreSQL service -> Start")
        elif "authentication failed" in error_str:
            print("\n[SOLUTION] Authentication failed!")
            print("  - Check username and password in .env file")
            print("  - Verify PostgreSQL user 'tjqjaqhd' exists")
        elif "database" in error_str and "does not exist" in error_str:
            print("\n[SOLUTION] Database does not exist!")
            print("  - Create database 'bithumb_trading'")
            print("  - Run: CREATE DATABASE bithumb_trading;")

        return False


if __name__ == "__main__":
    result = asyncio.run(complete_database_test())
    sys.exit(0 if result else 1)