#!/usr/bin/env python3
"""
스케줄러 서비스 테스트 스크립트

스케줄링 시스템의 모든 기능을 테스트합니다.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.config import DatabaseConfig
from src.services.scheduler_service import get_scheduler_service, cleanup_scheduler_service
from src.database.models.scheduler import ScheduleType, ExecutionStatus

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scheduler_test.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)


class TestNotificationHandler:
    """테스트용 알림 핸들러"""

    def __init__(self):
        self.notifications = []

    async def handle_notification(self, notification_data):
        """알림 처리"""
        self.notifications.append(notification_data)
        logger.info(f"알림 수신: {notification_data['level']} - {notification_data['title']}")
        logger.info(f"메시지: {notification_data['message']}")

    def get_notifications(self):
        """수신한 알림 반환"""
        return self.notifications.copy()

    def clear_notifications(self):
        """알림 목록 초기화"""
        self.notifications.clear()


async def test_scheduler_initialization():
    """스케줄러 초기화 테스트"""
    logger.info("\n" + "="*50)
    logger.info("1. 스케줄러 초기화 테스트")
    logger.info("="*50)

    try:
        # 데이터베이스 설정
        db_config = DatabaseConfig()

        # 스케줄러 서비스 가져오기
        scheduler = await get_scheduler_service(db_config)
        logger.info("스케줄러 서비스 초기화 성공")

        # 상태 확인
        status = scheduler.get_scheduler_status()
        logger.info(f"스케줄러 상태: {status}")

        return True

    except Exception as e:
        logger.error(f"스케줄러 초기화 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_schedule_config_management():
    """스케줄 설정 관리 테스트"""
    logger.info("\n" + "="*50)
    logger.info("2. 스케줄 설정 관리 테스트")
    logger.info("="*50)

    try:
        db_config = DatabaseConfig()
        scheduler = await get_scheduler_service(db_config)

        # 테스트용 스케줄 설정 생성
        test_config_data = {
            "name": "test_schedule",
            "description": "테스트용 스케줄",
            "schedule_type": ScheduleType.DAILY,
            "target_time": "10:30",
            "timezone": "Asia/Seoul",
            "check_holidays": True,
            "skip_weekends": True,
            "max_retries": 2,
            "retry_interval_minutes": 15,
            "enable_notifications": True,
            "is_active": True,
            "is_paused": False
        }

        # 1. 스케줄 설정 생성
        logger.info("스케줄 설정 생성 중...")
        config = await scheduler.create_schedule_config(test_config_data)
        logger.info(f"스케줄 설정 생성됨: ID {config.id}, 이름 '{config.name}'")
        logger.info(f"Cron 표현식: {config.to_cron_expression()}")

        # 2. 스케줄 설정 수정
        logger.info("스케줄 설정 수정 중...")
        updated_config = await scheduler.update_schedule_config(
            config.id,
            {
                "target_time": "11:00",
                "description": "수정된 테스트 스케줄"
            }
        )
        logger.info(f"스케줄 설정 수정됨: 시간 {updated_config.target_time}")

        # 3. 스케줄러 상태 확인
        status = scheduler.get_scheduler_status()
        logger.info(f"등록된 작업 수: {status['job_count']}")

        # 4. 스케줄 설정 삭제
        logger.info("테스트 스케줄 설정 삭제 중...")
        deleted = await scheduler.delete_schedule_config(config.id)
        logger.info(f"스케줄 설정 삭제됨: {deleted}")

        return True

    except Exception as e:
        logger.error(f"스케줄 설정 관리 테스트 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_manual_execution():
    """수동 실행 테스트"""
    logger.info("\n" + "="*50)
    logger.info("3. 수동 종목 재선정 테스트")
    logger.info("="*50)

    try:
        db_config = DatabaseConfig()
        scheduler = await get_scheduler_service(db_config)

        # 알림 핸들러 등록
        notification_handler = TestNotificationHandler()
        scheduler.add_notification_callback(notification_handler.handle_notification)

        logger.info("수동 종목 재선정 실행 중...")
        start_time = datetime.now()

        # 수동 실행
        result = await scheduler.execute_manual_selection()

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"실행 시간: {execution_time:.2f}초")

        if result['success']:
            logger.info("수동 종목 재선정 성공!")
            logger.info(f"로그 ID: {result['log_id']}")

            # 선정된 종목 정보 출력
            selected_assets = result['selected_assets']
            logger.info(f"분석된 총 종목 수: {selected_assets['summary']['total_analyzed']}")
            logger.info(f"신뢰 가능한 종목 수: {selected_assets['summary']['reliable_count']}")
            logger.info(f"평균 점수: {selected_assets['summary']['avg_score']:.2f}")
            logger.info(f"평균 신뢰도: {selected_assets['summary']['avg_reliability']:.2f}")

            logger.info("선정된 상위 종목:")
            for i, asset in enumerate(selected_assets['top_assets'], 1):
                logger.info(f"  {i}. {asset['symbol']}: 점수 {asset['score']:.2f}, "
                           f"신뢰도 {asset['reliability']:.2f}, 등급 {asset['grade']}")

            if selected_assets['portfolio_changes']:
                logger.info("포트폴리오 변경사항:")
                for change in selected_assets['portfolio_changes']:
                    logger.info(f"  {change['action']}: {change['symbol']} "
                               f"(사유: {change['reason']})")
        else:
            logger.error(f"수동 종목 재선정 실패: {result['error']}")

        # 수신한 알림 확인
        notifications = notification_handler.get_notifications()
        logger.info(f"수신한 알림 수: {len(notifications)}")

        return result['success']

    except Exception as e:
        logger.error(f"수동 실행 테스트 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_execution_history():
    """실행 이력 조회 테스트"""
    logger.info("\n" + "="*50)
    logger.info("4. 실행 이력 조회 테스트")
    logger.info("="*50)

    try:
        db_config = DatabaseConfig()
        scheduler = await get_scheduler_service(db_config)

        # 실행 이력 조회
        logger.info("최근 실행 이력 조회 중...")
        history = await scheduler.get_execution_history(limit=10)
        logger.info(f"조회된 이력 수: {len(history)}")

        for i, log in enumerate(history[:3], 1):  # 최근 3개만 출력
            logger.info(f"{i}. 실행시간: {log['execution_time']}")
            logger.info(f"   상태: {log['status']}, 타입: {log['schedule_type']}")
            if log['execution_duration_seconds']:
                logger.info(f"   소요시간: {log['execution_duration_seconds']:.2f}초")
            if log['total_assets_analyzed']:
                logger.info(f"   분석 종목: {log['total_assets_analyzed']}개")

        # 실행 통계 조회
        logger.info("\n실행 통계 조회 중...")
        stats = await scheduler.get_execution_statistics(days=7)  # 최근 7일
        logger.info(f"최근 7일 통계:")
        logger.info(f"  - 총 실행 수: {stats['total_executions']}")
        logger.info(f"  - 성공 실행 수: {stats['successful_executions']}")
        logger.info(f"  - 실패 실행 수: {stats['failed_executions']}")
        logger.info(f"  - 성공률: {stats['success_rate']:.1f}%")
        logger.info(f"  - 평균 실행시간: {stats['avg_execution_time_minutes']:.2f}분")

        return True

    except Exception as e:
        logger.error(f"실행 이력 조회 테스트 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_scheduler_lifecycle():
    """스케줄러 생명주기 테스트"""
    logger.info("\n" + "="*50)
    logger.info("5. 스케줄러 생명주기 테스트")
    logger.info("="*50)

    try:
        db_config = DatabaseConfig()
        scheduler = await get_scheduler_service(db_config)

        # 시작 상태 확인
        status = scheduler.get_scheduler_status()
        logger.info(f"초기 상태 - 실행중: {status['running']}, 작업수: {status['job_count']}")

        # 스케줄러 시작
        logger.info("스케줄러 시작 중...")
        await scheduler.start()

        status = scheduler.get_scheduler_status()
        logger.info(f"시작 후 상태 - 실행중: {status['running']}, 작업수: {status['job_count']}")

        # 등록된 작업들 확인
        if status['jobs']:
            logger.info("등록된 작업들:")
            for job in status['jobs']:
                logger.info(f"  - {job['id']}: 다음 실행 {job['next_run']}")

        # 잠시 대기
        await asyncio.sleep(2)

        # 스케줄러 중지
        logger.info("스케줄러 중지 중...")
        await scheduler.stop()

        status = scheduler.get_scheduler_status()
        logger.info(f"중지 후 상태 - 실행중: {status['running']}, 작업수: {status['job_count']}")

        return True

    except Exception as e:
        logger.error(f"스케줄러 생명주기 테스트 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_holiday_checking():
    """휴장일 체크 테스트"""
    logger.info("\n" + "="*50)
    logger.info("6. 휴장일 체크 테스트")
    logger.info("="*50)

    try:
        db_config = DatabaseConfig()
        scheduler = await get_scheduler_service(db_config)

        # 한국 공휴일 확인
        import holidays
        korea_holidays = holidays.SouthKorea(years=[2024, 2025])

        logger.info("2024-2025년 한국 공휴일:")
        for date, name in sorted(korea_holidays.items())[:10]:  # 처음 10개만
            logger.info(f"  {date}: {name}")

        # 현재 날짜의 휴장일 여부 확인
        from datetime import date
        today = date.today()
        is_holiday = today in korea_holidays
        is_weekend = today.weekday() >= 5

        logger.info(f"\n오늘 ({today}):")
        logger.info(f"  - 공휴일: {is_holiday}")
        logger.info(f"  - 주말: {is_weekend}")
        logger.info(f"  - 휴장일: {is_holiday or is_weekend}")

        return True

    except Exception as e:
        logger.error(f"휴장일 체크 테스트 실패: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def run_comprehensive_test():
    """종합 테스트 실행"""
    logger.info("스케줄러 서비스 종합 테스트 시작")
    logger.info("="*60)

    test_results = []

    # 1. 초기화 테스트
    result = await test_scheduler_initialization()
    test_results.append(("초기화", result))

    # 2. 설정 관리 테스트
    result = await test_schedule_config_management()
    test_results.append(("설정 관리", result))

    # 3. 수동 실행 테스트
    result = await test_manual_execution()
    test_results.append(("수동 실행", result))

    # 4. 이력 조회 테스트
    result = await test_execution_history()
    test_results.append(("이력 조회", result))

    # 5. 생명주기 테스트
    result = await test_scheduler_lifecycle()
    test_results.append(("생명주기", result))

    # 6. 휴장일 체크 테스트
    result = await test_holiday_checking()
    test_results.append(("휴장일 체크", result))

    # 정리
    await cleanup_scheduler_service()

    # 결과 요약
    logger.info("\n" + "="*60)
    logger.info("테스트 결과 요약")
    logger.info("="*60)

    passed = 0
    failed = 0

    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:15s}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    logger.info("-" * 60)
    logger.info(f"총 테스트: {len(test_results)}")
    logger.info(f"성공: {passed}")
    logger.info(f"실패: {failed}")
    logger.info(f"성공률: {passed/len(test_results)*100:.1f}%")

    if failed == 0:
        logger.info("\n모든 테스트가 성공했습니다!")
        logger.info("스케줄러 시스템이 정상적으로 작동합니다.")
    else:
        logger.warning(f"\n{failed}개의 테스트가 실패했습니다.")
        logger.warning("로그를 확인하여 문제를 해결해주세요.")

    return failed == 0


async def main():
    """메인 함수"""
    try:
        success = await run_comprehensive_test()

        print("\n" + "="*50)
        if success:
            print("스케줄러 서비스 테스트 완료!")
        else:
            print("스케줄러 서비스 테스트 중 오류 발생!")
        print("="*50)

        print("\n다음 단계:")
        print("1. 스케줄러 시작:")
        print("   from src.services.scheduler_service import get_scheduler_service")
        print("   scheduler = await get_scheduler_service(db_config)")
        print("   await scheduler.start()")
        print("\n2. 관리 기능:")
        print("   - 수동 실행: await scheduler.execute_manual_selection()")
        print("   - 이력 조회: await scheduler.get_execution_history()")
        print("   - 통계 조회: await scheduler.get_execution_statistics()")
        print("   - 상태 확인: scheduler.get_scheduler_status()")

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("사용자에 의해 테스트가 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"예상치 못한 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)