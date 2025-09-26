#!/usr/bin/env python3
"""
스케줄러 관리 도구

스케줄러 시스템을 관리하기 위한 CLI 도구입니다.
시작, 중지, 상태 확인, 수동 실행 등의 기능을 제공합니다.
"""

import asyncio
import argparse
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
from src.database.models.scheduler import ScheduleType

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class SchedulerManager:
    """스케줄러 관리 클래스"""

    def __init__(self):
        self.db_config = DatabaseConfig()
        self.scheduler = None

    async def initialize(self):
        """스케줄러 초기화"""
        self.scheduler = await get_scheduler_service(self.db_config)

    async def start_scheduler(self):
        """스케줄러 시작"""
        print("스케줄러를 시작하는 중...")
        await self.scheduler.start()
        print("스케줄러가 시작되었습니다.")

        status = self.scheduler.get_scheduler_status()
        print(f"등록된 작업 수: {status['job_count']}")

        if status['jobs']:
            print("\n등록된 작업들:")
            for job in status['jobs']:
                next_run = job['next_run'] or "예정 없음"
                print(f"  - {job['name']}: {next_run}")

    async def stop_scheduler(self):
        """스케줄러 중지"""
        print("스케줄러를 중지하는 중...")
        await self.scheduler.stop()
        print("스케줄러가 중지되었습니다.")

    async def show_status(self):
        """스케줄러 상태 조회"""
        status = self.scheduler.get_scheduler_status()

        print("="*50)
        print("스케줄러 상태")
        print("="*50)
        print(f"실행 상태: {'실행 중' if status['running'] else '중지됨'}")
        print(f"등록된 작업 수: {status['job_count']}")

        if status['jobs']:
            print("\n등록된 작업:")
            for job in status['jobs']:
                print(f"  ID: {job['id']}")
                print(f"  이름: {job['name']}")
                print(f"  다음 실행: {job['next_run'] or '예정 없음'}")
                print(f"  트리거: {job['trigger']}")
                print("-" * 30)

    async def execute_manual(self):
        """수동 종목 재선정 실행"""
        print("수동 종목 재선정을 실행합니다...")
        print("이 작업은 시간이 걸릴 수 있습니다. 잠시 기다려주세요.")

        start_time = datetime.now()
        result = await self.scheduler.execute_manual_selection()
        execution_time = (datetime.now() - start_time).total_seconds()

        print(f"\n실행 시간: {execution_time:.2f}초")

        if result['success']:
            print("종목 재선정 성공!")

            selected_assets = result['selected_assets']
            summary = selected_assets['summary']

            print(f"\n분석 결과:")
            print(f"  - 총 분석 종목: {summary['total_analyzed']}개")
            print(f"  - 신뢰 가능한 종목: {summary['reliable_count']}개")
            print(f"  - 평균 점수: {summary['avg_score']:.2f}")
            print(f"  - 평균 신뢰도: {summary['avg_reliability']:.2f}")

            print(f"\n선정된 상위 종목:")
            for i, asset in enumerate(selected_assets['top_assets'], 1):
                print(f"  {i}. {asset['symbol']}: "
                      f"점수 {asset['score']:.2f}, "
                      f"신뢰도 {asset['reliability']:.2f}, "
                      f"등급 {asset['grade']}")

            if selected_assets['portfolio_changes']:
                print(f"\n포트폴리오 변경사항:")
                for change in selected_assets['portfolio_changes']:
                    print(f"  - {change['action']}: {change['symbol']}")
                    if change['reason']:
                        print(f"    사유: {change['reason']}")

        else:
            print(f"종목 재선정 실패: {result['error']}")

    async def show_history(self, limit=10):
        """실행 이력 조회"""
        print("="*50)
        print(f"최근 {limit}개 실행 이력")
        print("="*50)

        history = await self.scheduler.get_execution_history(limit=limit)

        if not history:
            print("실행 이력이 없습니다.")
            return

        for i, log in enumerate(history, 1):
            status_emoji = "✅" if log['status'] == 'success' else "❌" if log['status'] == 'failed' else "⏳"
            print(f"{i}. {status_emoji} {log['execution_time']}")
            print(f"   상태: {log['status']}, 타입: {log['schedule_type']}")

            if log['execution_duration_seconds']:
                duration = round(log['execution_duration_seconds'], 2)
                print(f"   소요시간: {duration}초")

            if log['total_assets_analyzed']:
                print(f"   분석 종목: {log['total_assets_analyzed']}개")

            if log['error_message']:
                print(f"   오류: {log['error_message'][:100]}...")

            print("-" * 30)

    async def show_statistics(self, days=30):
        """실행 통계 조회"""
        print("="*50)
        print(f"최근 {days}일 실행 통계")
        print("="*50)

        stats = await self.scheduler.get_execution_statistics(days=days)

        print(f"총 실행 수: {stats['total_executions']}")
        print(f"성공 실행: {stats['successful_executions']}")
        print(f"실패 실행: {stats['failed_executions']}")
        print(f"성공률: {stats['success_rate']:.1f}%")
        print(f"평균 실행시간: {stats['avg_execution_time_minutes']:.2f}분")

    async def create_config(self):
        """새로운 스케줄 설정 생성"""
        print("새로운 스케줄 설정을 생성합니다.")
        print()

        # 사용자 입력
        name = input("스케줄 이름: ").strip()
        if not name:
            print("이름은 필수입니다.")
            return

        description = input("설명 (선택): ").strip()

        print("\n스케줄 타입:")
        print("1. 매일 (daily)")
        print("2. 매주 (weekly)")
        print("3. 매월 (monthly)")
        schedule_type_input = input("선택 (1-3): ").strip()

        schedule_type_map = {
            "1": ScheduleType.DAILY,
            "2": ScheduleType.WEEKLY,
            "3": ScheduleType.MONTHLY
        }

        schedule_type = schedule_type_map.get(schedule_type_input, ScheduleType.DAILY)

        target_time = input("실행 시간 (HH:MM, 기본값 09:00): ").strip() or "09:00"

        # 유효성 검사
        try:
            hour, minute = target_time.split(":")
            hour, minute = int(hour), int(minute)
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("잘못된 시간 형식")
        except:
            print("시간 형식이 올바르지 않습니다. HH:MM 형식으로 입력해주세요.")
            return

        skip_weekends = input("주말 건너뛰기 (Y/n): ").strip().lower() != 'n'
        check_holidays = input("공휴일 체크 (Y/n): ").strip().lower() != 'n'

        # 설정 데이터 구성
        config_data = {
            "name": name,
            "description": description,
            "schedule_type": schedule_type,
            "target_time": target_time,
            "timezone": "Asia/Seoul",
            "check_holidays": check_holidays,
            "skip_weekends": skip_weekends,
            "max_retries": 3,
            "retry_interval_minutes": 30,
            "enable_notifications": True,
            "notification_on_success": True,
            "notification_on_failure": True,
            "is_active": True,
            "is_paused": False
        }

        try:
            config = await self.scheduler.create_schedule_config(config_data)
            print(f"\n스케줄 설정이 생성되었습니다!")
            print(f"ID: {config.id}")
            print(f"이름: {config.name}")
            print(f"Cron 표현식: {config.to_cron_expression()}")
            print(f"활성화: {config.is_enabled}")

        except Exception as e:
            print(f"스케줄 설정 생성 실패: {str(e)}")

    async def run_daemon(self):
        """데몬 모드로 스케줄러 실행"""
        print("스케줄러를 데몬 모드로 시작합니다.")
        print("중지하려면 Ctrl+C를 누르세요.")

        try:
            await self.scheduler.start()

            # 상태 출력
            status = self.scheduler.get_scheduler_status()
            print(f"등록된 작업 수: {status['job_count']}")

            if status['jobs']:
                print("등록된 작업들:")
                for job in status['jobs']:
                    next_run = job['next_run'] or "예정 없음"
                    print(f"  - {job['name']}: {next_run}")

            print("\n스케줄러가 실행 중입니다...")

            # 무한 대기
            while True:
                await asyncio.sleep(60)  # 1분마다 깨어남
                # 주기적으로 상태 확인 (옵션)

        except KeyboardInterrupt:
            print("\n스케줄러를 중지합니다...")
            await self.scheduler.stop()
            print("스케줄러가 중지되었습니다.")

    async def cleanup(self):
        """정리"""
        if self.scheduler:
            await cleanup_scheduler_service()


async def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="스케줄러 관리 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python scheduler_manager.py start          # 스케줄러 시작 (일회성)
  python scheduler_manager.py daemon         # 데몬 모드로 실행
  python scheduler_manager.py status         # 상태 조회
  python scheduler_manager.py manual         # 수동 실행
  python scheduler_manager.py history        # 실행 이력 조회
  python scheduler_manager.py stats          # 통계 조회
  python scheduler_manager.py create-config  # 설정 생성
        """
    )

    parser.add_argument(
        'command',
        choices=['start', 'stop', 'status', 'manual', 'history', 'stats', 'create-config', 'daemon'],
        help='실행할 명령'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='이력 조회 시 최대 개수 (기본값: 10)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='통계 조회 기간 (일 단위, 기본값: 30)'
    )

    args = parser.parse_args()

    manager = SchedulerManager()

    try:
        await manager.initialize()

        if args.command == 'start':
            await manager.start_scheduler()

        elif args.command == 'stop':
            await manager.stop_scheduler()

        elif args.command == 'status':
            await manager.show_status()

        elif args.command == 'manual':
            await manager.execute_manual()

        elif args.command == 'history':
            await manager.show_history(limit=args.limit)

        elif args.command == 'stats':
            await manager.show_statistics(days=args.days)

        elif args.command == 'create-config':
            await manager.create_config()

        elif args.command == 'daemon':
            await manager.run_daemon()

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    finally:
        await manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())