"""
데이터베이스 연결 테스트 스크립트
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


async def test_db_connection():
    """Database connection test"""
    print("[INFO] Starting database connection test...")

    try:
        # Check settings
        settings = get_settings()
        print(f"[INFO] Database URL: {settings.DATABASE_URL}")

        # Attempt database initialization
        print("[INFO] Attempting database connection...")
        db_config = await init_database()
        print("[SUCCESS] Database connection successful!")

        # Run health check
        print("[INFO] Running database health check...")
        is_healthy = await db_config.health_check()

        if is_healthy:
            print("[SUCCESS] Database health check successful!")

            # Test table creation
            print("[INFO] Testing table creation...")
            async with db_config.get_session() as session:
                # Create basic tables
                await session.run_sync(Base.metadata.create_all, bind=db_config.engine.sync_engine)
                print("[SUCCESS] Table creation successful!")

                # Simple query test
                from sqlalchemy import text
                result = await session.execute(text("SELECT version()"))
                version = result.scalar()
                print(f"[INFO] PostgreSQL version: {version}")

        else:
            print("[ERROR] Database health check failed!")
            return False

        # Close connection
        await close_database()
        print("[INFO] Database connection closed successfully.")
        return True

    except Exception as e:
        print(f"[ERROR] Database connection test failed: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")

        # Suggest possible solutions
        if "connection refused" in str(e).lower():
            print("\n[SOLUTION] Solutions:")
            print("  1. Check if PostgreSQL server is running")
            print("  2. Check if port 5432 is open")
            print("  3. Check firewall settings")
        elif "authentication failed" in str(e).lower():
            print("\n[SOLUTION] Solutions:")
            print("  1. Check database username and password")
            print("  2. Check DATABASE_URL in .env file")
        elif "database" in str(e).lower() and "does not exist" in str(e).lower():
            print("\n[SOLUTION] Solutions:")
            print("  1. Create 'bithumb_trading' database")
            print("  2. Run: CREATE DATABASE bithumb_trading;")

        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Bithumb Trading System - Database Connection Test")
    print("=" * 60)

    result = asyncio.run(test_db_connection())

    if result:
        print("\n[SUCCESS] All database tests completed successfully!")
        sys.exit(0)
    else:
        print("\n[WARNING] Database connection has issues.")
        print("Please refer to the solutions above to check your configuration.")
        sys.exit(1)