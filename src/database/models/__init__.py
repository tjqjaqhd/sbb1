"""
데이터베이스 모델

SQLAlchemy 2.0 기반의 모든 모델 클래스를 정의합니다.
"""

# 기본 클래스
from .base import Base, BaseModel, TimestampMixin

# 시장 데이터 모델
from .market import MarketData, Ticker, OrderBook, Transaction

# 거래 관련 모델
from .trading import Order, Balance, TradingSignal, TradeHistory

# 사용자 관련 모델
from .user import Strategy, UserSetting

# 스케줄러 관련 모델
from .scheduler import ExecutionLog, SchedulerConfig, ExecutionStatus, ScheduleType

# 모든 모델 클래스 리스트
__all__ = [
    # Base classes
    "Base",
    "BaseModel",
    "TimestampMixin",

    # Market models
    "MarketData",
    "Ticker",
    "OrderBook",
    "Transaction",

    # Trading models
    "Order",
    "Balance",
    "TradingSignal",
    "TradeHistory",

    # User models
    "Strategy",
    "UserSetting",

    # Scheduler models
    "ExecutionLog",
    "SchedulerConfig",
    "ExecutionStatus",
    "ScheduleType",
]

# 테이블 생성 순서 정의 (Foreign Key 의존성 순서)
TABLE_CREATION_ORDER = [
    # 독립적인 테이블들 (Foreign Key 없음)
    "MarketData",
    "Ticker",
    "OrderBook",
    "Transaction",
    "Balance",
    "UserSetting",

    # Strategy (독립적)
    "Strategy",

    # Scheduler 테이블들 (독립적)
    "SchedulerConfig",
    "ExecutionLog",

    # 의존성이 있는 테이블들
    "TradingSignal",  # Strategy에 의존하지 않지만 TradeHistory에서 참조됨
    "Order",          # Strategy를 참조
    "TradeHistory",   # Order, TradingSignal, Strategy를 참조
]