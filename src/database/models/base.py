"""
SQLAlchemy 기본 모델 클래스

모든 모델이 상속받을 기본 클래스와 공통 설정을 정의합니다.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from decimal import Decimal

from sqlalchemy import DateTime, func
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy 기본 모델 클래스"""

    # 타입 어노테이션 맵핑
    type_annotation_map = {
        datetime: DateTime(timezone=True),
        Decimal: "DECIMAL(20, 8)",
    }


class TimestampMixin:
    """생성일시, 수정일시를 자동으로 관리하는 Mixin"""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class BaseModel(Base, TimestampMixin):
    """공통 필드를 가진 기본 모델"""

    __abstract__ = True

    @declared_attr
    def __tablename__(cls) -> str:
        """클래스명을 기반으로 테이블명 자동 생성"""
        return cls.__name__.lower()

    def to_dict(self) -> Dict[str, Any]:
        """모델을 딕셔너리로 변환"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, Decimal):
                value = float(value)
            result[column.name] = value
        return result

    def __repr__(self) -> str:
        """모델의 문자열 표현"""
        class_name = self.__class__.__name__
        attrs = []

        # Primary key만 표시
        for column in self.__table__.primary_key.columns:
            value = getattr(self, column.name, None)
            if value is not None:
                attrs.append(f"{column.name}={value}")

        return f"<{class_name}({', '.join(attrs)})>"