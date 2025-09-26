#!/usr/bin/env python3
"""
AssetSelector 통합 테스트 스크립트

종목 선정 알고리즘의 전체적인 동작을 검증합니다.
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database.config import DatabaseConfig
from src.services.asset_selector import AssetSelector, get_asset_selector

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('asset_selector_test.log')
    ]
)

logger = logging.getLogger(__name__)


async def test_basic_functionality():
    """기본 기능 테스트"""
    try:
        logger.info("=== AssetSelector 기본 기능 테스트 시작 ===")

        # 데이터베이스 설정
        db_config = DatabaseConfig()

        async with AssetSelector(db_config) as selector:
            # 1. 상태 확인
            logger.info("1. 상태 확인 테스트")
            health = await selector.health_check()
            logger.info(f"상태: {health}")

            if not health.get('all_services_healthy', False):
                logger.warning("일부 서비스가 불안정합니다")

            # 2. 거래 가능 종목 조회 테스트
            logger.info("2. 거래 가능 종목 조회 테스트")
            symbols = await selector.get_available_symbols()
            logger.info(f"거래 가능 종목 수: {len(symbols)}")

            if symbols:
                logger.info(f"예시 종목들: {symbols[:10]}")
            else:
                logger.error("거래 가능 종목이 없습니다")
                return False

            # 3. 소수 종목 평가 테스트 (속도 고려)
            logger.info("3. 종목 평가 테스트 (상위 10개만)")
            test_symbols = symbols[:10]  # 테스트용으로 10개만

            asset_scores = await selector.evaluate_symbols(test_symbols)
            logger.info(f"평가 완료된 종목 수: {len(asset_scores)}")

            for asset in asset_scores[:5]:  # 상위 5개 출력
                logger.info(
                    f"  {asset.symbol}: 점수={asset.score:.1f}, "
                    f"신뢰도={asset.reliability:.2f}, 등급={asset.grade}"
                )

            # 4. 필터링 및 순위 매기기 테스트
            logger.info("4. 필터링 및 순위 매기기 테스트")
            ranked_assets = selector.filter_and_rank_assets(asset_scores)
            logger.info(f"필터링 후 종목 수: {len(ranked_assets)}")

            # 5. 상위 종목 선별 테스트
            logger.info("5. 상위 종목 선별 테스트")
            top_assets = selector.select_top_assets(ranked_assets, 5)
            logger.info(f"선별된 종목 수: {len(top_assets)}")

            for i, asset in enumerate(top_assets, 1):
                logger.info(f"  {i}위: {asset.symbol} (점수: {asset.score:.1f})")

            # 6. 통계 정보 확인
            stats = selector.get_stats()
            logger.info(f"통계 정보: {stats}")

            return True

    except Exception as e:
        logger.error(f"기본 기능 테스트 중 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return False


async def test_full_selection_process():
    """전체 선정 프로세스 테스트"""
    try:
        logger.info("=== 전체 종목 선정 프로세스 테스트 시작 ===")

        db_config = DatabaseConfig()

        async with AssetSelector(db_config) as selector:
            # 전체 종목 선정 프로세스 실행
            logger.info("전체 종목 선정 프로세스 실행 (최대 20개 종목으로 제한)")

            # 성능을 위해 소수 종목만 평가
            available_symbols = await selector.get_available_symbols()
            test_symbols = available_symbols[:20] if len(available_symbols) > 20 else available_symbols

            # 선정 프로세스 실행
            start_time = datetime.now()

            # 직접적인 프로세스 실행 대신 단계별로 실행
            asset_scores = await selector.evaluate_symbols(test_symbols)

            if not asset_scores:
                logger.error("종목 평가 결과가 없습니다")
                return False

            ranked_assets = selector.filter_and_rank_assets(asset_scores)
            selected_assets = selector.select_top_assets(ranked_assets, 5)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info(f"선정 완료 - 소요시간: {duration:.2f}초")
            logger.info(f"평가된 종목: {len(asset_scores)}")
            logger.info(f"필터링 후: {len(ranked_assets)}")
            logger.info(f"최종 선별: {len(selected_assets)}")

            # 선별된 종목 상세 출력
            logger.info("최종 선별된 종목들:")
            for i, asset in enumerate(selected_assets, 1):
                logger.info(
                    f"  {i}. {asset.symbol}"
                    f" - 점수: {asset.score:.1f}"
                    f" - 신뢰도: {asset.reliability:.2f}"
                    f" - 등급: {asset.grade}"
                    f" - 추천: {asset.recommendation.get('action', 'N/A')}"
                )

            # 포트폴리오 변경 테스트
            logger.info("포트폴리오 변경 분석 테스트")
            current_portfolio = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']  # 가상의 기존 포트폴리오

            changes = selector.analyze_portfolio_changes(selected_assets, current_portfolio)
            logger.info(f"포트폴리오 변경 사항: {len(changes)}건")

            for change in changes:
                logger.info(f"  {change.action}: {change.symbol} - {change.reason}")

            return True

    except Exception as e:
        logger.error(f"전체 선정 프로세스 테스트 중 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return False


async def test_edge_cases():
    """엣지 케이스 테스트"""
    try:
        logger.info("=== 엣지 케이스 테스트 시작 ===")

        db_config = DatabaseConfig()

        async with AssetSelector(db_config) as selector:
            # 1. 빈 리스트 처리 테스트
            logger.info("1. 빈 종목 리스트 처리 테스트")
            empty_scores = await selector.evaluate_symbols([])
            logger.info(f"빈 리스트 결과: {len(empty_scores)}")

            # 2. 존재하지 않는 종목 테스트
            logger.info("2. 존재하지 않는 종목 테스트")
            fake_symbols = ['FAKE1', 'FAKE2', 'NONEXISTENT']
            fake_scores = await selector.evaluate_symbols(fake_symbols)
            logger.info(f"가짜 종목 평가 결과: {len(fake_scores)}")

            # 3. 매우 적은 수의 종목으로 선별 테스트
            logger.info("3. 소수 종목으로 선별 테스트")
            real_symbols = await selector.get_available_symbols()
            if real_symbols:
                small_symbols = real_symbols[:2]  # 2개만
                small_scores = await selector.evaluate_symbols(small_symbols)
                ranked = selector.filter_and_rank_assets(small_scores)
                selected = selector.select_top_assets(ranked, 5)  # 5개 요청하지만 2개만 있음

                logger.info(f"2개 종목으로 5개 선별 요청 결과: {len(selected)}개")

            return True

    except Exception as e:
        logger.error(f"엣지 케이스 테스트 중 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return False


async def main():
    """메인 테스트 함수"""
    try:
        logger.info("AssetSelector 통합 테스트 시작")
        logger.info("=" * 60)

        # 테스트 실행
        test_results = []

        # 1. 기본 기능 테스트
        result1 = await test_basic_functionality()
        test_results.append(("기본 기능", result1))

        print("\n" + "=" * 60 + "\n")

        # 2. 전체 프로세스 테스트
        result2 = await test_full_selection_process()
        test_results.append(("전체 프로세스", result2))

        print("\n" + "=" * 60 + "\n")

        # 3. 엣지 케이스 테스트
        result3 = await test_edge_cases()
        test_results.append(("엣지 케이스", result3))

        # 결과 요약
        logger.info("=" * 60)
        logger.info("테스트 결과 요약")
        logger.info("=" * 60)

        all_passed = True
        for test_name, result in test_results:
            status = "통과" if result else "실패"
            logger.info(f"{test_name}: {status}")
            if not result:
                all_passed = False

        logger.info("=" * 60)
        final_status = "모든 테스트 통과" if all_passed else "일부 테스트 실패"
        logger.info(f"최종 결과: {final_status}")

        return all_passed

    except Exception as e:
        logger.error(f"메인 테스트 실행 중 오류: {str(e)}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # 비동기 실행
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("테스트가 사용자에 의해 중단되었습니다")
        sys.exit(1)
    except Exception as e:
        logger.error(f"테스트 실행 중 치명적 오류: {str(e)}")
        sys.exit(1)