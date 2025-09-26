"""
시장 데이터 관련 모델

시계열 데이터, 실시간 시세, 호가, 거래 체결 정보를 저장하는 모델들을 정의합니다.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    ARRAY, BigInteger, String, Integer, Index, UniqueConstraint,
    text, DECIMAL, DateTime
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


class MarketData(Base):
    """
    시장 데이터 - OHLCV 캔들 데이터

    시계열 데이터로 월별 파티셔닝 적용
    """
    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # 기본 정보
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 1d
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )

    # OHLCV 데이터
    open_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    high_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    low_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    close_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)

    # 부가 정보
    quote_volume: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))
    trade_count: Mapped[Optional[int]] = mapped_column(Integer)

    # 시간 정보
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()"),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()"),
        onupdate=text("NOW()"),
        nullable=False
    )

    # 인덱스 및 제약 조건
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp', name='uq_market_data_symbol_timeframe_timestamp'),
        Index('idx_market_data_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_market_data_timeframe_timestamp', 'timeframe', 'timestamp'),
        # 파티셔닝은 일단 제거 (나중에 수동으로 설정 가능)
        # {
        #     'postgresql_partition_by': 'RANGE (timestamp)',
        # }
    )


class Ticker(Base):
    """
    실시간 시세 데이터

    WebSocket으로 수신한 실시간 시세 정보
    """
    __tablename__ = "tickers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # 가격 정보
    opening_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))
    closing_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))
    min_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))
    max_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))

    # 거래량 정보
    volume_24h: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))
    volume_7d: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))

    # 호가 정보
    best_bid: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))
    best_ask: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))

    # 변동률 정보
    price_change_24h: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))
    price_change_rate_24h: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 4))

    # 시간 정보
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()"),
        nullable=False
    )

    # 인덱스 및 제약 조건
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', name='uq_ticker_symbol_timestamp'),
        Index('idx_ticker_symbol_timestamp', 'symbol', 'timestamp'),
    )


class OrderBook(Base):
    """
    호가 데이터

    실시간 호가창 정보 (매수/매도 호가)
    """
    __tablename__ = "orderbooks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # 매수 호가 (상위 10개)
    bid_prices: Mapped[Optional[List[Decimal]]] = mapped_column(ARRAY(DECIMAL(20, 8)))
    bid_quantities: Mapped[Optional[List[Decimal]]] = mapped_column(ARRAY(DECIMAL(20, 8)))

    # 매도 호가 (상위 10개)
    ask_prices: Mapped[Optional[List[Decimal]]] = mapped_column(ARRAY(DECIMAL(20, 8)))
    ask_quantities: Mapped[Optional[List[Decimal]]] = mapped_column(ARRAY(DECIMAL(20, 8)))

    # 총 호가량
    total_bid_quantity: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))
    total_ask_quantity: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))

    # 시간 정보
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()"),
        nullable=False
    )

    # 인덱스 및 제약 조건
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', name='uq_orderbook_symbol_timestamp'),
        Index('idx_orderbook_symbol_timestamp', 'symbol', 'timestamp'),
    )


class Transaction(Base):
    """
    거래 체결 데이터

    실시간 거래 체결 정보
    """
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # 체결 정보
    price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    total: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)

    # 거래 타입
    transaction_type: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY, SELL

    # 빗썸 원본 데이터
    transaction_id: Mapped[Optional[str]] = mapped_column(String(50))

    # 시간 정보
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()"),
        nullable=False
    )

    # 인덱스
    __table_args__ = (
        Index('idx_transactions_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_transactions_timestamp', 'timestamp'),
        Index('idx_transactions_transaction_id', 'transaction_id'),
    )