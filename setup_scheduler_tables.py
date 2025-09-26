#!/usr/bin/env python3
"""
스케줄러 테이블 생성 스크립트

스케줄링 시스템에 필요한 데이터베이스 테이블들을 생성합니다.
"""

import asyncio
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.config import DatabaseConfig
from src.database.models import Base, ExecutionLog, SchedulerConfig
from src.database.models.scheduler import ExecutionStatus, ScheduleType
from sqlalchemy import text, select

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scheduler_setup.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


async def create_scheduler_tables():
    """스케줄러 테이블 생성 및 초기 데이터 삽입"""
    try:
        # 데이터베이스 설정
        db_config = DatabaseConfig()
        await db_config.initialize()

        logger.info("데이터베이스 연결 테스트 중...")

        # 연결 테스트
        async with db_config.get_session() as session:
            await session.execute(text("SELECT 1"))
            logger.info("데이터베이스 연결 성공")

        # 테이블 생성
        logger.info("스케줄러 테이블 생성 중...")

        async with db_config.engine.begin() as conn:
            # 스케줄러 관련 테이블만 생성
            await conn.run_sync(Base.metadata.create_all)

        logger.info("스케줄러 테이블 생성 완료")

        # 기본 스케줄 설정 삽입
        logger.info("기본 스케줄 설정 생성 중...")

        async with db_config.get_session() as session:
            # 기본 일일 자동 재선정 스케줄 설정
            default_config = SchedulerConfig(
                name="daily_asset_selection",
                description="매일 오전 9시 자동 종목 재선정",
                schedule_type=ScheduleType.DAILY,
                target_time="09:00",
                timezone="Asia/Seoul",
                check_holidays=True,
                skip_weekends=True,
                max_retries=3,
                retry_interval_minutes=30,
                enable_notifications=True,
                notification_on_success=True,
                notification_on_failure=True,
                is_active=True,
                is_paused=False,
                config_data={
                    "description": "한국 주식시장 개장 전 종목 재선정",
                    "created_by": "system",
                    "version": "1.0"
                }
            )

            session.add(default_config)
            await session.commit()

            logger.info(f"기본 스케줄 설정 생성됨: ID {default_config.id}")

        # 테이블 확인
        logger.info("생성된 테이블 확인 중...")

        async with db_config.get_session() as session:
            # ExecutionLog 테이블 확인
            result = await session.execute(text("SELECT COUNT(*) as count FROM execution_log"))
            execution_log_count = result.fetchone()[0]
            logger.info(f"execution_log 테이블: {execution_log_count}건")

            # SchedulerConfig 테이블 확인
            result = await session.execute(text("SELECT COUNT(*) as count FROM scheduler_config"))
            config_count = result.fetchone()[0]
            logger.info(f"scheduler_config 테이블: {config_count}건")

        logger.info("스케줄러 테이블 설정 완료!")
        return True

    except Exception as e:
        logger.error(f"스케줄러 테이블 생성 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_scheduler_models():
    """스케줄러 모델 테스트"""
    try:
        logger.info("스케줄러 모델 테스트 시작...")

        db_config = DatabaseConfig()
        await db_config.initialize()

        async with db_config.get_session() as session:
            # 테스트 실행 로그 생성
            from datetime import datetime
            import pytz

            korea_tz = pytz.timezone('Asia/Seoul')
            now = datetime.now(korea_tz)

            test_log = ExecutionLog(
                execution_time=now,
                scheduled_time=now,
                schedule_type=ScheduleType.MANUAL,
                status=ExecutionStatus.SUCCESS,
                selected_assets={
                    "test": "data",
                    "top_assets": ["BTC", "ETH", "XRP"]
                },
                total_assets_analyzed=50,
                execution_duration_seconds=15.5,
                log_messages=["테스트 실행", "모든 기능 정상"]
            )

            session.add(test_log)
            await session.commit()

            logger.info(f"테스트 실행 로그 생성됨: ID {test_log.id}")

            # 생성된 데이터 조회 테스트

            result = await session.execute(
                select(ExecutionLog).where(ExecutionLog.id == test_log.id)
            )
            retrieved_log = result.scalar_one_or_none()

            if retrieved_log:
                logger.info("데이터 조회 성공:")
                logger.info(f"  - ID: {retrieved_log.id}")
                logger.info(f"  - 실행시간: {retrieved_log.execution_time}")
                logger.info(f"  - 상태: {retrieved_log.status}")
                logger.info(f"  - 선정 종목: {retrieved_log.selected_assets}")
                logger.info(f"  - 실행 시간(분): {retrieved_log.duration_minutes}")
                logger.info(f"  - 성공 여부: {retrieved_log.is_successful}")

            # 스케줄 설정 조회 테스트
            config_result = await session.execute(select(SchedulerConfig))
            configs = config_result.scalars().all()

            logger.info(f"스케줄 설정 조회: {len(configs)}개")
            for config in configs:
                logger.info(f"  - {config.name}: {config.to_cron_expression()}")
                logger.info(f"    활성화: {config.is_enabled}")
                logger.info(f"    알림: {config.enable_notifications}")

        logger.info("스케줄러 모델 테스트 완료!")
        return True

    except Exception as e:
        logger.error(f"스케줄러 모델 테스트 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def main():
    """메인 실행 함수"""
    logger.info("="*50)
    logger.info("스케줄러 시스템 데이터베이스 설정 시작")
    logger.info("="*50)

    # 1. 테이블 생성
    success = await create_scheduler_tables()
    if not success:
        logger.error("테이블 생성 실패!")
        return

    # 2. 모델 테스트
    success = await test_scheduler_models()
    if not success:
        logger.error("모델 테스트 실패!")
        return

    logger.info("="*50)
    logger.info("스케줄러 시스템 데이터베이스 설정 완료")
    logger.info("="*50)

    # 다음 단계 안내
    print("\n" + "="*50)
    print("스케줄러 시스템 설정이 완료되었습니다!")
    print("="*50)
    print("\n다음 단계:")
    print("1. 스케줄러 서비스 테스트:")
    print("   python test_scheduler_service.py")
    print("\n2. 스케줄러 시작:")
    print("   from src.services.scheduler_service import get_scheduler_service")
    print("   scheduler = await get_scheduler_service(db_config)")
    print("   await scheduler.start()")
    print("\n3. 수동 종목 재선정 실행:")
    print("   result = await scheduler.execute_manual_selection()")
    print("\n생성된 테이블:")
    print("- execution_log: 스케줄러 실행 이력")
    print("- scheduler_config: 스케줄 설정")


if __name__ == "__main__":
    asyncio.run(main())