"""
빗썸 자동매매 시스템 설정 파일

환경 변수를 통해 설정을 로드하고 애플리케이션 전체에서 사용할 수 있는
중앙 집중식 설정 관리 모듈
"""

import os
from typing import List, Optional
from decouple import config, Csv
from pathlib import Path

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """애플리케이션 설정 클래스"""

    # =============================================================================
    # 기본 애플리케이션 설정
    # =============================================================================
    APP_NAME: str = config("APP_NAME", default="빗썸 자동매매 시스템")
    APP_VERSION: str = config("APP_VERSION", default="0.1.0")
    DEBUG: bool = config("DEBUG", default=False, cast=bool)
    LOG_LEVEL: str = config("LOG_LEVEL", default="INFO")

    # =============================================================================
    # 빗썸 API 설정
    # =============================================================================
    BITHUMB_API_KEY: Optional[str] = config("BITHUMB_API_KEY", default=None)
    BITHUMB_SECRET_KEY: Optional[str] = config("BITHUMB_SECRET_KEY", default=None)
    BITHUMB_API_URL: str = config("BITHUMB_API_URL", default="https://api.bithumb.com")
    BITHUMB_WEBSOCKET_URL: str = config(
        "BITHUMB_WEBSOCKET_URL",
        default="wss://pubwss.bithumb.com/pub/ws"
    )

    # API 호출 제한 (초당 요청 수)
    API_RATE_LIMIT: int = config("API_RATE_LIMIT", default=10, cast=int)

    # =============================================================================
    # 데이터베이스 설정
    # =============================================================================
    DATABASE_URL: str = config(
        "DATABASE_URL",
        default="postgresql+asyncpg://user:password@localhost:5432/bithumb_trading"
    )
    DATABASE_POOL_SIZE: int = config("DATABASE_POOL_SIZE", default=20, cast=int)
    DATABASE_MAX_OVERFLOW: int = config("DATABASE_MAX_OVERFLOW", default=30, cast=int)

    # =============================================================================
    # Redis 설정
    # =============================================================================
    REDIS_URL: str = config("REDIS_URL", default="redis://localhost:6379/0")
    REDIS_PASSWORD: Optional[str] = config("REDIS_PASSWORD", default=None)
    REDIS_DB: int = config("REDIS_DB", default=0, cast=int)

    # =============================================================================
    # JWT 보안 설정
    # =============================================================================
    SECRET_KEY: str = config(
        "SECRET_KEY",
        default="your_super_secret_jwt_key_here_change_in_production"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = config(
        "ACCESS_TOKEN_EXPIRE_MINUTES", default=30, cast=int
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = config(
        "REFRESH_TOKEN_EXPIRE_DAYS", default=7, cast=int
    )

    # =============================================================================
    # 매매 전략 설정
    # =============================================================================
    # 최대 포지션 크기 (KRW)
    MAX_POSITION_SIZE: int = config("MAX_POSITION_SIZE", default=100000, cast=int)

    # 위험 관리
    RISK_PERCENTAGE: float = config("RISK_PERCENTAGE", default=2.0, cast=float)
    STOP_LOSS_PERCENTAGE: float = config("STOP_LOSS_PERCENTAGE", default=5.0, cast=float)
    TAKE_PROFIT_PERCENTAGE: float = config("TAKE_PROFIT_PERCENTAGE", default=10.0, cast=float)

    # 선정할 종목 수
    TARGET_COINS_COUNT: int = config("TARGET_COINS_COUNT", default=5, cast=int)

    # 기술적 지표 설정
    SMA_SHORT_PERIOD: int = config("SMA_SHORT_PERIOD", default=20, cast=int)
    SMA_LONG_PERIOD: int = config("SMA_LONG_PERIOD", default=50, cast=int)
    RSI_PERIOD: int = config("RSI_PERIOD", default=14, cast=int)
    RSI_OVERBOUGHT: int = config("RSI_OVERBOUGHT", default=70, cast=int)
    RSI_OVERSOLD: int = config("RSI_OVERSOLD", default=30, cast=int)

    # =============================================================================
    # 알림 설정
    # =============================================================================
    NOTIFICATION_EMAIL: Optional[str] = config("NOTIFICATION_EMAIL", default=None)
    NOTIFICATION_TELEGRAM_BOT_TOKEN: Optional[str] = config(
        "NOTIFICATION_TELEGRAM_BOT_TOKEN", default=None
    )
    NOTIFICATION_TELEGRAM_CHAT_ID: Optional[str] = config(
        "NOTIFICATION_TELEGRAM_CHAT_ID", default=None
    )
    NOTIFICATION_DISCORD_WEBHOOK: Optional[str] = config(
        "NOTIFICATION_DISCORD_WEBHOOK", default=None
    )

    # =============================================================================
    # 모니터링 및 로깅
    # =============================================================================
    SENTRY_DSN: Optional[str] = config("SENTRY_DSN", default=None)
    PROMETHEUS_PORT: int = config("PROMETHEUS_PORT", default=8000, cast=int)

    # 로그 파일 설정
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: str = config("LOG_FILE", default="trading.log")
    LOG_MAX_SIZE: str = config("LOG_MAX_SIZE", default="100MB")
    LOG_BACKUP_COUNT: int = config("LOG_BACKUP_COUNT", default=10, cast=int)

    # =============================================================================
    # FastAPI 서버 설정
    # =============================================================================
    HOST: str = config("HOST", default="0.0.0.0")
    PORT: int = config("PORT", default=8080, cast=int)
    WORKERS: int = config("WORKERS", default=1, cast=int)

    # CORS 설정
    ALLOWED_ORIGINS: List[str] = config(
        "ALLOWED_ORIGINS",
        default="http://localhost:3000,http://localhost:8080",
        cast=Csv()
    )

    # =============================================================================
    # 개발 환경 설정
    # =============================================================================
    # 개발 모드에서 사용할 가짜 데이터
    USE_MOCK_DATA: bool = config("USE_MOCK_DATA", default=False, cast=bool)

    # 백테스팅 데이터 기간
    BACKTEST_DAYS: int = config("BACKTEST_DAYS", default=365, cast=int)

    @classmethod
    def is_api_configured(cls) -> bool:
        """빗썸 API 키가 설정되어 있는지 확인"""
        return cls.BITHUMB_API_KEY is not None and cls.BITHUMB_SECRET_KEY is not None

    @classmethod
    def is_notification_configured(cls) -> bool:
        """알림 설정이 되어 있는지 확인"""
        return any([
            cls.NOTIFICATION_EMAIL,
            cls.NOTIFICATION_TELEGRAM_BOT_TOKEN,
            cls.NOTIFICATION_DISCORD_WEBHOOK
        ])

    @classmethod
    def ensure_directories(cls) -> None:
        """필요한 디렉토리들을 생성"""
        cls.LOG_DIR.mkdir(exist_ok=True)
        (BASE_DIR / "data").mkdir(exist_ok=True)


# 전역 설정 인스턴스
settings = Settings()

# 애플리케이션 시작 시 필요한 디렉토리 생성
settings.ensure_directories()


def get_settings() -> Settings:
    """설정 인스턴스 반환"""
    return settings


# 설정 검증
def validate_settings() -> List[str]:
    """설정 유효성 검증 및 경고 메시지 반환"""
    warnings = []

    if not settings.is_api_configured():
        warnings.append("빗썸 API 키가 설정되지 않았습니다. 실제 거래는 불가능합니다.")

    if not settings.is_notification_configured():
        warnings.append("알림 설정이 되어 있지 않습니다. 중요한 알림을 받을 수 없습니다.")

    if settings.DEBUG and settings.SECRET_KEY == "your_super_secret_jwt_key_here_change_in_production":
        warnings.append("기본 SECRET_KEY를 사용 중입니다. 보안을 위해 변경해주세요.")

    if settings.DATABASE_URL.startswith("postgresql+asyncpg://user:password@"):
        warnings.append("기본 데이터베이스 설정을 사용 중입니다. 실제 데이터베이스 정보로 변경해주세요.")

    return warnings


if __name__ == "__main__":
    # 설정 확인 스크립트
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"🔧 Debug 모드: {settings.DEBUG}")
    print(f"📡 API 설정됨: {settings.is_api_configured()}")
    print(f"🔔 알림 설정됨: {settings.is_notification_configured()}")

    warnings = validate_settings()
    if warnings:
        print("\n⚠️  설정 경고:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\n✅ 모든 설정이 올바르게 구성되었습니다!")