"""
사용자 관련 모델

거래 전략, 사용자 설정 등 사용자 개인화 관련 데이터를 저장하는 모델들을 정의합니다.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Integer, String, Index, UniqueConstraint,
    text, DECIMAL, DateTime, Boolean, TEXT, JSON
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Strategy(Base):
    """
    거래 전략

    사용자가 설정한 거래 전략의 설정과 파라미터를 저장
    """
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 전략 기본 정보
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(TEXT)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # 전략 설정
    symbols: Mapped[Optional[List[str]]] = mapped_column(JSON)  # 대상 종목들
    timeframes: Mapped[Optional[List[str]]] = mapped_column(JSON)  # 사용할 시간 간격들

    # 리스크 관리
    max_position_size: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(5, 4))  # 최대 포지션 크기 (%)
    stop_loss_pct: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(5, 4))  # 손절 비율
    take_profit_pct: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(5, 4))  # 익절 비율

    # 기술적 지표 파라미터 (JSON)
    technical_params: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

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
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # 관계
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="strategy")
    trade_histories: Mapped[list["TradeHistory"]] = relationship("TradeHistory", back_populates="strategy")

    # 인덱스
    __table_args__ = (
        Index('idx_strategies_is_active', 'is_active'),
        Index('idx_strategies_last_run_at', 'last_run_at'),
    )


class UserSetting(Base):
    """
    사용자 설정

    시스템 전반의 사용자 설정을 키-값 형태로 저장
    """
    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 설정 식별
    category: Mapped[str] = mapped_column(String(50), nullable=False)  # API, TRADING, NOTIFICATION 등
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[Optional[str]] = mapped_column(TEXT)

    # 설정 메타데이터
    description: Mapped[Optional[str]] = mapped_column(TEXT)
    is_encrypted: Mapped[bool] = mapped_column(Boolean, default=False)  # 암호화 필요 여부

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

    # 제약 조건
    __table_args__ = (
        UniqueConstraint('category', 'key', name='uq_user_settings_category_key'),
        Index('idx_user_settings_category', 'category'),
    )