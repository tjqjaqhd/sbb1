"""
데이터베이스 설정 및 연결 관리

PostgreSQL 데이터베이스 연결, 세션 관리, 연결 풀 등을 관리합니다.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from config.config import get_settings

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """데이터베이스 설정 및 연결 관리"""

    def __init__(self):
        """데이터베이스 설정 초기화"""
        self.settings = get_settings()
        self._engine: Optional[AsyncEngine] = None
        self._async_session_factory: Optional[async_sessionmaker] = None
        self._initialized = False

    @property
    def database_url(self) -> str:
        """PostgreSQL 연결 URL 생성"""
        return (
            f"postgresql+asyncpg://"
            f"{self.settings.DB_USER}:{self.settings.DB_PASSWORD}@"
            f"{self.settings.DB_HOST}:{self.settings.DB_PORT}/"
            f"{self.settings.DB_NAME}"
        )

    @property
    def sync_database_url(self) -> str:
        """동기 PostgreSQL 연결 URL (마이그레이션용)"""
        return (
            f"postgresql://"
            f"{self.settings.DB_USER}:{self.settings.DB_PASSWORD}@"
            f"{self.settings.DB_HOST}:{self.settings.DB_PORT}/"
            f"{self.settings.DB_NAME}"
        )

    async def initialize(self) -> None:
        """데이터베이스 연결 및 세션 팩토리 초기화"""
        if self._initialized:
            return

        logger.info("데이터베이스 연결을 초기화합니다...")

        # 비동기 엔진 생성
        self._engine = create_async_engine(
            self.database_url,
            # 연결 풀 설정
            poolclass=QueuePool,
            pool_size=20,                    # 기본 연결 수
            max_overflow=30,                 # 추가 연결 수
            pool_timeout=30,                 # 연결 대기 타임아웃
            pool_recycle=3600,               # 연결 재활용 시간 (1시간)
            pool_pre_ping=True,              # 연결 유효성 검사

            # 엔진 옵션
            echo=self.settings.DEBUG,        # SQL 로깅 (개발 환경에서만)
            echo_pool=False,                 # 연결 풀 로깅
            future=True,                     # SQLAlchemy 2.0 모드

            # asyncpg 설정
            connect_args={
                "server_settings": {
                    "application_name": "bithumb_trading_bot",
                    "jit": "off",            # JIT 비활성화 (안정성)
                }
            }
        )

        # 세션 팩토리 생성
        self._async_session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            autoflush=False,                 # 자동 flush 비활성화
            autocommit=False,                # 자동 commit 비활성화
            expire_on_commit=False,          # commit 후 객체 만료 방지
        )

        self._initialized = True
        logger.info("데이터베이스 연결이 초기화되었습니다.")

    async def close(self) -> None:
        """데이터베이스 연결 종료"""
        if self._engine:
            logger.info("데이터베이스 연결을 종료합니다...")
            await self._engine.dispose()
            self._engine = None
            self._async_session_factory = None
            self._initialized = False
            logger.info("데이터베이스 연결이 종료되었습니다.")

    @property
    def engine(self) -> AsyncEngine:
        """비동기 엔진 반환"""
        if not self._initialized or not self._engine:
            raise RuntimeError("데이터베이스가 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker:
        """세션 팩토리 반환"""
        if not self._initialized or not self._async_session_factory:
            raise RuntimeError("데이터베이스가 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")
        return self._async_session_factory

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """비동기 세션 컨텍스트 매니저"""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def health_check(self) -> bool:
        """데이터베이스 연결 상태 확인"""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"데이터베이스 헬스 체크 실패: {e}")
            return False


# 전역 데이터베이스 설정 인스턴스
_db_config: Optional[DatabaseConfig] = None


def get_database_config() -> DatabaseConfig:
    """전역 데이터베이스 설정 인스턴스 반환"""
    global _db_config
    if _db_config is None:
        _db_config = DatabaseConfig()
    return _db_config


async def init_database() -> DatabaseConfig:
    """데이터베이스 초기화"""
    db_config = get_database_config()
    await db_config.initialize()
    return db_config


async def close_database() -> None:
    """데이터베이스 연결 종료"""
    global _db_config
    if _db_config:
        await _db_config.close()
        _db_config = None


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """데이터베이스 세션 컨텍스트 매니저 (편의 함수)"""
    db_config = get_database_config()
    async with db_config.get_session() as session:
        yield session


# 동기 엔진 생성 함수 (Alembic 마이그레이션용)
def create_sync_engine():
    """동기 엔진 생성 (마이그레이션 등에서 사용)"""
    settings = get_settings()
    sync_url = (
        f"postgresql://"
        f"{settings.DB_USER}:{settings.DB_PASSWORD}@"
        f"{settings.DB_HOST}:{settings.DB_PORT}/"
        f"{settings.DB_NAME}"
    )

    return create_engine(
        sync_url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True,
        echo=settings.DEBUG,
    )