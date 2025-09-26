"""
스케줄링 및 자동 재선정 시스템

APScheduler를 활용한 종목 자동 재선정 스케줄링 시스템을 구현합니다.
매일 한국시간 09:00에 자동으로 종목을 재선정하고 결과를 저장합니다.
"""

import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import json
from functools import wraps

import holidays
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func

from src.database.config import DatabaseConfig
from src.database.models.scheduler import (
    ExecutionLog, SchedulerConfig, ExecutionStatus, ScheduleType
)
from src.services.asset_selector import AssetSelector, get_asset_selector

logger = logging.getLogger(__name__)


def error_handler(func):
    """스케줄러 메서드 오류 처리 데코레이터"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper


class SchedulerService:
    """스케줄링 및 자동 재선정 서비스"""

    def __init__(self, db_config: DatabaseConfig):
        """스케줄러 서비스 초기화"""
        self.db_config = db_config
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.asset_selector: Optional[AssetSelector] = None
        self.is_running = False

        # 한국 시간대 설정
        self.korea_tz = pytz.timezone('Asia/Seoul')

        # 한국 공휴일 처리
        self.korea_holidays = holidays.SouthKorea(years=range(2024, 2030))

        # 알림 콜백 함수들
        self._notification_callbacks: List[Callable] = []

    async def initialize(self):
        """스케줄러 초기화"""
        try:
            logger.info("스케줄러 서비스 초기화 중...")

            # AssetSelector 초기화
            self.asset_selector = await get_asset_selector(self.db_config)

            # 스케줄러 설정
            jobstores = {
                'default': SQLAlchemyJobStore(url=self.db_config.sync_database_url)
            }

            executors = {
                'default': AsyncIOExecutor()
            }

            job_defaults = {
                'coalesce': False,       # 놓친 작업들을 한번에 실행하지 않음
                'max_instances': 1,      # 동시에 하나의 인스턴스만 실행
                'misfire_grace_time': 300  # 5분 내에서는 놓친 작업도 실행
            }

            # AsyncIOScheduler 생성
            self.scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=self.korea_tz
            )

            # 이벤트 리스너 등록
            self.scheduler.add_listener(
                self._job_executed_listener,
                EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED
            )

            logger.info("스케줄러 서비스 초기화 완료")

        except Exception as e:
            logger.error(f"스케줄러 초기화 실패: {str(e)}")
            raise

    async def start(self):
        """스케줄러 시작"""
        if self.scheduler is None:
            await self.initialize()

        if not self.is_running:
            logger.info("스케줄러 시작 중...")
            self.scheduler.start()
            self.is_running = True

            # 기본 스케줄 설정
            await self._load_and_apply_schedules()

            logger.info("스케줄러가 시작되었습니다")

    async def stop(self):
        """스케줄러 중지"""
        if self.scheduler and self.is_running:
            logger.info("스케줄러 중지 중...")
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("스케줄러가 중지되었습니다")

    @error_handler
    async def _load_and_apply_schedules(self):
        """데이터베이스에서 스케줄 설정을 로드하고 적용"""
        # 데이터베이스가 초기화되지 않았다면 초기화
        if not self.db_config._initialized:
            await self.db_config.initialize()

        async with self.db_config.get_session() as session:
            # 활성화된 스케줄 설정들을 조회
            result = await session.execute(
                select(SchedulerConfig).where(
                    SchedulerConfig.is_active == True,
                    SchedulerConfig.is_paused == False
                )
            )
            configs = result.scalars().all()

            for config in configs:
                await self._add_schedule_job(config)

    async def _add_schedule_job(self, config: SchedulerConfig):
        """스케줄 작업 추가"""
        job_id = f"asset_selection_{config.id}"

        try:
            # 기존 작업이 있다면 제거
            if self.scheduler.get_job(job_id):
                self.scheduler.remove_job(job_id)

            # cron 표현식 생성
            cron_expr = config.to_cron_expression()
            hour, minute = config.target_time.split(":")

            if config.schedule_type == ScheduleType.DAILY:
                if config.skip_weekends:
                    # 월-금 실행
                    self.scheduler.add_job(
                        func=self._execute_asset_selection,
                        trigger='cron',
                        hour=int(hour),
                        minute=int(minute),
                        day_of_week='mon-fri',
                        id=job_id,
                        args=[config.id],
                        replace_existing=True
                    )
                else:
                    # 매일 실행
                    self.scheduler.add_job(
                        func=self._execute_asset_selection,
                        trigger='cron',
                        hour=int(hour),
                        minute=int(minute),
                        id=job_id,
                        args=[config.id],
                        replace_existing=True
                    )
            elif config.schedule_type == ScheduleType.WEEKLY:
                # 매주 월요일 실행
                self.scheduler.add_job(
                    func=self._execute_asset_selection,
                    trigger='cron',
                    hour=int(hour),
                    minute=int(minute),
                    day_of_week='mon',
                    id=job_id,
                    args=[config.id],
                    replace_existing=True
                )
            elif config.schedule_type == ScheduleType.MONTHLY:
                # 매월 1일 실행
                self.scheduler.add_job(
                    func=self._execute_asset_selection,
                    trigger='cron',
                    hour=int(hour),
                    minute=int(minute),
                    day=1,
                    id=job_id,
                    args=[config.id],
                    replace_existing=True
                )

            logger.info(f"스케줄 작업 추가됨: {job_id} ({config.name})")

        except Exception as e:
            logger.error(f"스케줄 작업 추가 실패: {job_id} - {str(e)}")

    @error_handler
    async def _execute_asset_selection(self, config_id: int):
        """자동 종목 재선정 실행"""
        execution_log = None
        start_time = datetime.now(self.korea_tz)

        try:
            logger.info(f"자동 종목 재선정 시작 (config_id: {config_id})")

            # 스케줄 설정 조회
            async with self.db_config.get_session() as session:
                config_result = await session.execute(
                    select(SchedulerConfig).where(SchedulerConfig.id == config_id)
                )
                config = config_result.scalar_one_or_none()

                if not config or not config.is_enabled:
                    logger.warning(f"비활성화된 스케줄 설정: {config_id}")
                    return

                # 휴장일 체크
                if await self._is_market_closed(config):
                    logger.info("시장 휴장일로 인한 실행 건너뜀")
                    return

                # 실행 로그 생성
                execution_log = ExecutionLog(
                    execution_time=start_time,
                    scheduled_time=start_time,
                    schedule_type=config.schedule_type,
                    status=ExecutionStatus.RUNNING,
                    config_snapshot=config.to_dict()
                )
                session.add(execution_log)
                await session.commit()

                # AssetSelector를 통한 종목 재선정
                result = await self.asset_selector.run_asset_selection()

                # 실행 완료 시간 계산
                completion_time = datetime.now(self.korea_tz)
                duration = (completion_time - start_time).total_seconds()

                # 선정된 종목 정보 구성
                selected_assets = {
                    "top_assets": [
                        {
                            "symbol": asset.symbol,
                            "score": asset.score,
                            "reliability": asset.reliability,
                            "grade": asset.grade
                        }
                        for asset in result.top_assets
                    ],
                    "portfolio_changes": [
                        {
                            "action": change.action,
                            "symbol": change.symbol,
                            "old_symbol": change.old_symbol,
                            "score_improvement": change.score_improvement,
                            "reason": change.reason
                        }
                        for change in result.changes
                    ],
                    "summary": {
                        "total_analyzed": result.total_analyzed,
                        "reliable_count": result.reliable_count,
                        "avg_score": result.avg_score,
                        "avg_reliability": result.avg_reliability
                    }
                }

                # 실행 로그 업데이트
                execution_log.status = ExecutionStatus.SUCCESS
                execution_log.completion_time = completion_time
                execution_log.execution_duration_seconds = duration
                execution_log.selected_assets = selected_assets
                execution_log.total_assets_analyzed = result.total_analyzed
                execution_log.log_messages = [
                    f"선정 완료: {len(result.top_assets)}개 종목",
                    f"포트폴리오 변경: {len(result.changes)}건",
                    f"평균 점수: {result.avg_score:.2f}",
                    f"평균 신뢰도: {result.avg_reliability:.2f}"
                ]

                await session.commit()

                logger.info(f"자동 종목 재선정 성공 완료 (실행시간: {duration:.2f}초)")

                # 성공 알림 발송
                if config.notification_on_success:
                    await self._send_notification(
                        "SUCCESS",
                        f"종목 재선정 성공",
                        f"총 {len(result.top_assets)}개 종목 선정 완료",
                        selected_assets
                    )

        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()

            logger.error(f"자동 종목 재선정 실패: {error_msg}")
            logger.error(error_traceback)

            # 실행 로그 오류 업데이트
            if execution_log:
                async with self.db_config.get_session() as session:
                    # 기존 세션에서 분리된 객체를 다시 연결
                    await session.merge(execution_log)

                    execution_log.status = ExecutionStatus.FAILED
                    execution_log.completion_time = datetime.now(self.korea_tz)
                    execution_log.error_message = error_msg
                    execution_log.error_traceback = error_traceback

                    await session.commit()

            # 오류 알림 발송
            async with self.db_config.get_session() as session:
                config_result = await session.execute(
                    select(SchedulerConfig).where(SchedulerConfig.id == config_id)
                )
                config = config_result.scalar_one_or_none()

                if config and config.notification_on_failure:
                    await self._send_notification(
                        "ERROR",
                        f"종목 재선정 실패",
                        error_msg,
                        {"error_traceback": error_traceback}
                    )

            # 재시도 로직
            if execution_log and execution_log.can_retry:
                await self._schedule_retry(config_id, execution_log.retry_count + 1)

    async def _is_market_closed(self, config: SchedulerConfig) -> bool:
        """시장 휴장일 여부 확인"""
        now = datetime.now(self.korea_tz).date()

        # 주말 체크
        if config.skip_weekends and now.weekday() >= 5:  # 토요일(5), 일요일(6)
            return True

        # 한국 공휴일 체크
        if config.check_holidays and now in self.korea_holidays:
            return True

        # 커스텀 휴장일 체크
        if config.custom_skip_dates:
            date_str = now.strftime('%Y-%m-%d')
            if date_str in config.custom_skip_dates:
                return True

        return False

    async def _schedule_retry(self, config_id: int, retry_count: int):
        """재시도 스케줄링"""
        try:
            # 재시도 시간 계산 (30분 후)
            retry_time = datetime.now(self.korea_tz) + timedelta(minutes=30)

            job_id = f"retry_asset_selection_{config_id}_{retry_count}"

            self.scheduler.add_job(
                func=self._execute_asset_selection,
                trigger='date',
                run_date=retry_time,
                id=job_id,
                args=[config_id],
                replace_existing=True
            )

            logger.info(f"재시도 스케줄링됨: {job_id} at {retry_time}")

        except Exception as e:
            logger.error(f"재시도 스케줄링 실패: {str(e)}")

    async def _job_executed_listener(self, event):
        """작업 실행 이벤트 리스너"""
        if event.exception:
            logger.error(f"작업 실행 중 오류 발생: {event.job_id} - {event.exception}")
        else:
            logger.info(f"작업 실행 완료: {event.job_id}")

    async def _send_notification(self, level: str, title: str, message: str, data: Optional[Dict] = None):
        """알림 발송"""
        try:
            notification_data = {
                "level": level,
                "title": title,
                "message": message,
                "timestamp": datetime.now(self.korea_tz).isoformat(),
                "data": data or {}
            }

            # 등록된 알림 콜백 함수들 실행
            for callback in self._notification_callbacks:
                try:
                    await callback(notification_data)
                except Exception as e:
                    logger.error(f"알림 콜백 실행 오류: {str(e)}")

            logger.info(f"알림 발송: {level} - {title}")

        except Exception as e:
            logger.error(f"알림 발송 실패: {str(e)}")

    def add_notification_callback(self, callback: Callable):
        """알림 콜백 함수 추가"""
        self._notification_callbacks.append(callback)

    def remove_notification_callback(self, callback: Callable):
        """알림 콜백 함수 제거"""
        if callback in self._notification_callbacks:
            self._notification_callbacks.remove(callback)

    @error_handler
    async def execute_manual_selection(self) -> Dict[str, Any]:
        """수동 종목 재선정 실행"""
        start_time = datetime.now(self.korea_tz)
        execution_log = None

        try:
            logger.info("수동 종목 재선정 시작")

            # 데이터베이스가 초기화되지 않았다면 초기화
            if not self.db_config._initialized:
                await self.db_config.initialize()

            async with self.db_config.get_session() as session:
                # 수동 실행 로그 생성
                execution_log = ExecutionLog(
                    execution_time=start_time,
                    schedule_type=ScheduleType.MANUAL,
                    status=ExecutionStatus.RUNNING
                )
                session.add(execution_log)
                await session.commit()

                # AssetSelector를 통한 종목 재선정
                result = await self.asset_selector.run_asset_selection()

                # 실행 완료 처리
                completion_time = datetime.now(self.korea_tz)
                duration = (completion_time - start_time).total_seconds()

                selected_assets = {
                    "top_assets": [
                        {
                            "symbol": asset.symbol,
                            "score": asset.score,
                            "reliability": asset.reliability,
                            "grade": asset.grade
                        }
                        for asset in result.top_assets
                    ],
                    "portfolio_changes": [
                        {
                            "action": change.action,
                            "symbol": change.symbol,
                            "old_symbol": change.old_symbol,
                            "score_improvement": change.score_improvement,
                            "reason": change.reason
                        }
                        for change in result.changes
                    ],
                    "summary": {
                        "total_analyzed": result.total_analyzed,
                        "reliable_count": result.reliable_count,
                        "avg_score": result.avg_score,
                        "avg_reliability": result.avg_reliability
                    }
                }

                # 실행 로그 업데이트
                execution_log.status = ExecutionStatus.SUCCESS
                execution_log.completion_time = completion_time
                execution_log.execution_duration_seconds = duration
                execution_log.selected_assets = selected_assets
                execution_log.total_assets_analyzed = result.total_analyzed

                await session.commit()

                logger.info(f"수동 종목 재선정 완료 (실행시간: {duration:.2f}초)")

                return {
                    "success": True,
                    "execution_time": duration,
                    "selected_assets": selected_assets,
                    "log_id": execution_log.id
                }

        except Exception as e:
            error_msg = str(e)
            error_traceback = traceback.format_exc()

            logger.error(f"수동 종목 재선정 실패: {error_msg}")

            # 오류 로그 업데이트
            if execution_log:
                async with self.db_config.get_session() as session:
                    await session.merge(execution_log)

                    execution_log.status = ExecutionStatus.FAILED
                    execution_log.completion_time = datetime.now(self.korea_tz)
                    execution_log.error_message = error_msg
                    execution_log.error_traceback = error_traceback

                    await session.commit()

            return {
                "success": False,
                "error": error_msg,
                "log_id": execution_log.id if execution_log else None
            }

    async def get_execution_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """실행 이력 조회"""
        if not self.db_config._initialized:
            await self.db_config.initialize()

        async with self.db_config.get_session() as session:
            result = await session.execute(
                select(ExecutionLog)
                .order_by(desc(ExecutionLog.execution_time))
                .limit(limit)
            )
            logs = result.scalars().all()

            return [log.to_dict() for log in logs]

    async def get_execution_statistics(self, days: int = 30) -> Dict[str, Any]:
        """실행 통계 조회"""
        if not self.db_config._initialized:
            await self.db_config.initialize()

        since_date = datetime.now(self.korea_tz) - timedelta(days=days)

        async with self.db_config.get_session() as session:
            # 전체 실행 수
            total_result = await session.execute(
                select(func.count(ExecutionLog.id))
                .where(ExecutionLog.execution_time >= since_date)
            )
            total_executions = total_result.scalar()

            # 성공 실행 수
            success_result = await session.execute(
                select(func.count(ExecutionLog.id))
                .where(
                    ExecutionLog.execution_time >= since_date,
                    ExecutionLog.status == ExecutionStatus.SUCCESS
                )
            )
            successful_executions = success_result.scalar()

            # 평균 실행 시간
            avg_duration_result = await session.execute(
                select(func.avg(ExecutionLog.execution_duration_seconds))
                .where(
                    ExecutionLog.execution_time >= since_date,
                    ExecutionLog.status == ExecutionStatus.SUCCESS
                )
            )
            avg_duration = avg_duration_result.scalar()

            success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0

            return {
                "period_days": days,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": total_executions - successful_executions,
                "success_rate": round(success_rate, 2),
                "avg_execution_time_seconds": float(avg_duration or 0),
                "avg_execution_time_minutes": round(float(avg_duration or 0) / 60, 2)
            }

    async def create_schedule_config(self, config_data: Dict[str, Any]) -> SchedulerConfig:
        """스케줄 설정 생성"""
        if not self.db_config._initialized:
            await self.db_config.initialize()

        async with self.db_config.get_session() as session:
            config = SchedulerConfig(**config_data)
            session.add(config)
            await session.commit()

            # 스케줄러가 실행 중이라면 작업 추가
            if self.is_running:
                await self._add_schedule_job(config)

            return config

    async def update_schedule_config(self, config_id: int, updates: Dict[str, Any]) -> Optional[SchedulerConfig]:
        """스케줄 설정 업데이트"""
        async with self.db_config.get_session() as session:
            result = await session.execute(
                select(SchedulerConfig).where(SchedulerConfig.id == config_id)
            )
            config = result.scalar_one_or_none()

            if not config:
                return None

            # 설정 업데이트
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            await session.commit()

            # 스케줄러가 실행 중이라면 작업 재등록
            if self.is_running:
                job_id = f"asset_selection_{config.id}"
                if self.scheduler.get_job(job_id):
                    self.scheduler.remove_job(job_id)

                if config.is_enabled:
                    await self._add_schedule_job(config)

            return config

    async def delete_schedule_config(self, config_id: int) -> bool:
        """스케줄 설정 삭제"""
        async with self.db_config.get_session() as session:
            result = await session.execute(
                select(SchedulerConfig).where(SchedulerConfig.id == config_id)
            )
            config = result.scalar_one_or_none()

            if not config:
                return False

            # 실행 중인 작업 제거
            if self.is_running:
                job_id = f"asset_selection_{config.id}"
                if self.scheduler.get_job(job_id):
                    self.scheduler.remove_job(job_id)

            # 설정 삭제
            await session.delete(config)
            await session.commit()

            return True

    def get_scheduler_status(self) -> Dict[str, Any]:
        """스케줄러 상태 조회"""
        if not self.scheduler:
            return {"running": False, "jobs": []}

        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name or job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })

        return {
            "running": self.is_running,
            "jobs": jobs,
            "job_count": len(jobs)
        }


# 전역 인스턴스 관리
_scheduler_service: Optional[SchedulerService] = None


async def get_scheduler_service(db_config: Optional[DatabaseConfig] = None) -> SchedulerService:
    """SchedulerService 싱글톤 인스턴스 반환"""
    global _scheduler_service

    if db_config is None:
        db_config = DatabaseConfig()

    if _scheduler_service is None:
        _scheduler_service = SchedulerService(db_config)
        await _scheduler_service.initialize()

    return _scheduler_service


async def cleanup_scheduler_service():
    """SchedulerService 정리"""
    global _scheduler_service

    if _scheduler_service:
        await _scheduler_service.stop()
        _scheduler_service = None