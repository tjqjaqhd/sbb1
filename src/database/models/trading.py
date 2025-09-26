"""
거래 관련 모델

주문, 잔고, 매매 신호, 거래 내역 등 거래 관련 데이터를 저장하는 모델들을 정의합니다.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger, String, Integer, Index, UniqueConstraint,
    text, DECIMAL, DateTime, Boolean, ForeignKey
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Order(Base):
    """
    주문 정보

    사용자가 실행한 주문의 상세 정보를 저장
    """
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # 주문 식별
    order_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)  # 빗썸 주문 ID
    client_order_id: Mapped[Optional[str]] = mapped_column(String(100))  # 클라이언트 주문 ID

    # 주문 기본 정보
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY, SELL
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)  # LIMIT, MARKET

    # 가격/수량 정보
    price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))  # 주문 가격 (시장가는 NULL)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)  # 주문 수량
    filled_quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)  # 체결된 수량
    remaining_quantity: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))  # 미체결 수량

    # 상태 정보
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # PENDING, FILLED, PARTIAL, CANCELLED

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
    filled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # 추가 정보
    commission: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))  # 수수료
    strategy_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("strategies.id")
    )  # 연관된 전략 ID

    # 관계
    strategy: Mapped[Optional["Strategy"]] = relationship("Strategy", back_populates="orders")
    trade_histories: Mapped[list["TradeHistory"]] = relationship("TradeHistory", back_populates="order")

    # 인덱스
    __table_args__ = (
        Index('idx_orders_symbol_status', 'symbol', 'status'),
        Index('idx_orders_created_at', 'created_at'),
        Index('idx_orders_strategy_id', 'strategy_id'),
    )


class Balance(Base):
    """
    잔고 정보

    사용자 계정의 자산별 잔고 정보
    """
    __tablename__ = "balances"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # 자산 정보
    currency: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)  # KRW, BTC, ETH 등
    available: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)  # 사용 가능한 잔고
    locked: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), default=0)  # 주문 중인 잔고

    # 총 잔고는 계산 컬럼으로 처리 (available + locked)

    # 시간 정보
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()"),
        onupdate=text("NOW()"),
        nullable=False
    )

    @property
    def total(self) -> Decimal:
        """총 잔고 (사용가능 + 락된 잔고)"""
        return self.available + self.locked


class TradingSignal(Base):
    """
    매매 신호

    기술적 분석을 통해 생성된 매수/매도 신호 정보
    """
    __tablename__ = "trading_signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # 신호 기본 정보
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    signal_type: Mapped[str] = mapped_column(String(20), nullable=False)  # BUY, SELL, HOLD
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)  # 전략명

    # 신호 상세 정보
    confidence_score: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(5, 4))  # 신뢰도 점수 (0-1)
    price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)  # 신호 발생 시점 가격
    target_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))  # 목표 가격
    stop_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))  # 손절 가격

    # 기술적 지표 값들
    rsi_value: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(5, 2))
    macd_value: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 6))
    bollinger_position: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(5, 4))
    volume_ratio: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 4))

    # 상태 및 시간
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE")  # ACTIVE, EXECUTED, EXPIRED
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()"),
        nullable=False
    )
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # 관계
    trade_histories: Mapped[list["TradeHistory"]] = relationship("TradeHistory", back_populates="signal")

    # 인덱스
    __table_args__ = (
        Index('idx_signals_symbol_status', 'symbol', 'status'),
        Index('idx_signals_created_at', 'created_at'),
        Index('idx_signals_strategy_name', 'strategy_name'),
    )


class TradeHistory(Base):
    """
    거래 내역

    실제 거래 실행 결과를 저장하는 로그 테이블
    """
    __tablename__ = "trade_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # 거래 기본 정보
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY, SELL

    # 가격/수량 정보
    price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    total: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    commission: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)

    # 연관 정보
    order_id: Mapped[str] = mapped_column(String(50), nullable=False)
    signal_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("trading_signals.id")
    )  # 매매 신호 ID
    strategy_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("strategies.id")
    )  # 전략 ID

    # 손익 정보
    pnl: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8))  # 실현 손익
    pnl_pct: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 6))  # 손익률

    # 시간 정보
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("NOW()"),
        nullable=False
    )

    # 관계
    order: Mapped["Order"] = relationship("Order", back_populates="trade_histories")
    signal: Mapped[Optional["TradingSignal"]] = relationship("TradingSignal", back_populates="trade_histories")
    strategy: Mapped[Optional["Strategy"]] = relationship("Strategy", back_populates="trade_histories")

    # 인덱스
    __table_args__ = (
        Index('idx_trade_history_symbol_executed', 'symbol', 'executed_at'),
        Index('idx_trade_history_strategy', 'strategy_id'),
        Index('idx_trade_history_signal', 'signal_id'),
        Index('idx_trade_history_order_id', 'order_id'),
    )