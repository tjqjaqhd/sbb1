"""
스케줄러 관련 데이터베이스 모델

스케줄링 시스템의 실행 이력과 설정을 저장하는 모델들을 정의합니다.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List
from enum import Enum

from sqlalchemy import String, Text, Boolean, Integer, JSON, DateTime, Enum as SQLEnum, DECIMAL
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel


class ExecutionStatus(str, Enum):
    """실행 상태"""
    PENDING = "pending"      # 실행 대기
    RUNNING = "running"      # 실행 중
    SUCCESS = "success"      # 성공
    FAILED = "failed"        # 실패
    CANCELLED = "cancelled"  # 취소됨


class ScheduleType(str, Enum):
    """스케줄 타입"""
    DAILY = "daily"          # 일일 실행
    WEEKLY = "weekly"        # 주간 실행
    MONTHLY = "monthly"      # 월간 실행
    MANUAL = "manual"        # 수동 실행


class ExecutionLog(BaseModel):
    """스케줄러 실행 이력"""

    __tablename__ = 'execution_log'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 실행 정보
    execution_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    scheduled_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completion_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # 실행 타입 및 상태
    schedule_type: Mapped[ScheduleType] = mapped_column(SQLEnum(ScheduleType), nullable=False)
    status: Mapped[ExecutionStatus] = mapped_column(SQLEnum(ExecutionStatus), nullable=False)

    # 결과 정보
    selected_assets: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    total_assets_analyzed: Mapped[Optional[int]] = mapped_column(Integer)
    execution_duration_seconds: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 3))

    # 오류 및 로그 정보
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_traceback: Mapped[Optional[str]] = mapped_column(Text)
    log_messages: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # 재시도 정보
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)

    # 스케줄러 설정 참조
    config_snapshot: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    def __repr__(self) -> str:
        return (f"<ExecutionLog(id={self.id}, "
                f"execution_time={self.execution_time}, "
                f"status={self.status})>")

    @property
    def is_successful(self) -> bool:
        """실행이 성공했는지 확인"""
        return self.status == ExecutionStatus.SUCCESS

    @property
    def can_retry(self) -> bool:
        """재시도 가능 여부 확인"""
        return (self.status == ExecutionStatus.FAILED and
                self.retry_count < self.max_retries)

    @property
    def duration_minutes(self) -> Optional[float]:
        """실행 시간을 분 단위로 반환"""
        if self.execution_duration_seconds:
            return float(self.execution_duration_seconds) / 60.0
        return None


class SchedulerConfig(BaseModel):
    """스케줄러 설정"""

    __tablename__ = 'scheduler_config'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # 스케줄 설정
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(String(500))
    schedule_type: Mapped[ScheduleType] = mapped_column(SQLEnum(ScheduleType), nullable=False)

    # 시간 설정
    target_time: Mapped[str] = mapped_column(String(10), nullable=False, default="09:00")  # HH:MM 형식
    timezone: Mapped[str] = mapped_column(String(50), nullable=False, default="Asia/Seoul")

    # 휴장일 설정
    check_holidays: Mapped[bool] = mapped_column(Boolean, default=True)
    skip_weekends: Mapped[bool] = mapped_column(Boolean, default=True)
    custom_skip_dates: Mapped[Optional[List[str]]] = mapped_column(JSON)  # ["YYYY-MM-DD"] 형식

    # 재시도 설정
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    retry_interval_minutes: Mapped[int] = mapped_column(Integer, default=30)

    # 알림 설정
    enable_notifications: Mapped[bool] = mapped_column(Boolean, default=True)
    notification_on_success: Mapped[bool] = mapped_column(Boolean, default=True)
    notification_on_failure: Mapped[bool] = mapped_column(Boolean, default=True)
    notification_recipients: Mapped[Optional[List[str]]] = mapped_column(JSON)

    # 상태 및 제어
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_paused: Mapped[bool] = mapped_column(Boolean, default=False)

    # 추가 설정
    config_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    def __repr__(self) -> str:
        return (f"<SchedulerConfig(id={self.id}, "
                f"name='{self.name}', "
                f"is_active={self.is_active})>")

    @property
    def is_enabled(self) -> bool:
        """스케줄러가 활성화되어 있고 일시정지되지 않았는지 확인"""
        return self.is_active and not self.is_paused

    def to_cron_expression(self) -> str:
        """cron 표현식으로 변환"""
        hour, minute = self.target_time.split(":")

        if self.schedule_type == ScheduleType.DAILY:
            if self.skip_weekends:
                return f"{minute} {hour} * * 1-5"  # 월-금
            else:
                return f"{minute} {hour} * * *"    # 매일
        elif self.schedule_type == ScheduleType.WEEKLY:
            return f"{minute} {hour} * * 1"        # 매주 월요일
        elif self.schedule_type == ScheduleType.MONTHLY:
            return f"{minute} {hour} 1 * *"        # 매월 1일
        else:
            return f"{minute} {hour} * * *"        # 기본값: 매일