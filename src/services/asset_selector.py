"""
종목 선정 알고리즘

가중 점수 기반 종목 순위 결정 및 상위 5개 종목 선별 로직을 구현합니다.
ScoringSystem을 활용하여 각 종목을 평가하고 최적의 포트폴리오를 구성합니다.
"""

import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from src.database.config import DatabaseConfig
from src.services.scoring_system import ScoringSystem, get_scoring_system
from src.api.bithumb.client import BithumbHTTPClient, get_http_client

logger = logging.getLogger(__name__)


@dataclass
class AssetScore:
    """종목 점수 정보"""
    symbol: str
    score: float
    reliability: float
    recommendation: Dict[str, Any]
    analysis_timestamp: datetime
    quality_assessment: Dict[str, Any]

    @property
    def is_reliable(self) -> bool:
        """신뢰할 수 있는 점수인지 확인"""
        return self.reliability >= 0.6

    @property
    def grade(self) -> str:
        """점수 등급 반환"""
        return self.recommendation.get('grade', '알 수 없음')


@dataclass
class PortfolioChange:
    """포트폴리오 변경 정보"""
    action: str  # 'add', 'remove', 'replace'
    symbol: str
    old_symbol: Optional[str] = None  # 교체 시 이전 종목
    score_improvement: float = 0.0
    reason: str = ""


@dataclass
class SelectionResult:
    """종목 선정 결과"""
    timestamp: datetime
    selected_assets: List[AssetScore]
    excluded_assets: List[AssetScore]
    portfolio_changes: List[PortfolioChange]
    selection_criteria: Dict[str, Any]
    total_evaluated: int
    success_rate: float


class AssetSelector:
    """
    종목 선정 알고리즘

    빗썸 거래 가능 종목을 대상으로 ScoringSystem을 활용한 종합 평가를 수행하여
    상위 5개 종목을 선별하고 포트폴리오 최적화를 지원합니다.
    """

    # 기본 설정
    DEFAULT_TOP_COUNT = 5  # 선별할 상위 종목 수
    MIN_SCORE_THRESHOLD = 45.0  # 최소 점수 기준
    MIN_RELIABILITY_THRESHOLD = 0.6  # 최소 신뢰도 기준

    # 교체 기준
    SCORE_DIFFERENCE_THRESHOLD = 10.0  # 점수 차이 최소 기준
    RELIABILITY_IMPROVEMENT_THRESHOLD = 0.2  # 신뢰도 개선 최소 기준

    # 제외 기준
    EXCLUDED_SYMBOLS = {'KRW', 'BTC_KRW', 'ETH_KRW'}  # 항상 제외할 심볼 (테스트용)
    MIN_DAILY_VOLUME_KRW = 1_000_000_000  # 최소 일일 거래량 (10억원)

    def __init__(self, db_config: DatabaseConfig):
        """
        종목 선정기 초기화

        Args:
            db_config: 데이터베이스 설정 객체
        """
        self.db_config = db_config
        self._scoring_system: Optional[ScoringSystem] = None
        self._http_client: Optional[BithumbHTTPClient] = None

        # 통계 정보
        self._stats = {
            'total_selections': 0,
            'successful_selections': 0,
            'last_selection': None,
            'avg_evaluation_time': 0.0,
            'errors': 0
        }

        # 이전 선정 결과 캐시 (교체 판단용)
        self._last_selection: Optional[SelectionResult] = None

    async def _ensure_services(self):
        """필요한 서비스들 초기화 확인"""
        try:
            if self._scoring_system is None:
                self._scoring_system = await get_scoring_system(self.db_config)

            if self._http_client is None:
                self._http_client = await get_http_client()

        except Exception as e:
            logger.error(f"서비스 초기화 중 오류: {str(e)}")
            raise

    async def get_available_symbols(self) -> List[str]:
        """
        빗썸에서 거래 가능한 모든 종목 목록 조회

        Returns:
            거래 가능한 심볼 리스트
        """
        try:
            await self._ensure_services()

            # 빗썸 ticker API로 전체 종목 조회
            response = await self._http_client.get("/public/ticker/ALL_KRW")

            if not response or 'data' not in response:
                logger.error("빗썸 ticker API 응답이 비어있습니다")
                return []

            # 응답 데이터에서 심볼 추출
            ticker_data = response['data']
            symbols = []

            for symbol, data in ticker_data.items():
                if symbol == 'date':  # 날짜 정보 제외
                    continue

                # 기본적인 필터링
                if self._is_symbol_tradeable(symbol, data):
                    symbols.append(symbol)

            logger.info(f"빗썸에서 {len(symbols)}개 거래 가능 종목 발견")
            return symbols

        except Exception as e:
            logger.error(f"거래 가능 종목 조회 중 오류: {str(e)}")
            return []

    def _is_symbol_tradeable(self, symbol: str, ticker_data: Dict[str, Any]) -> bool:
        """
        종목이 거래 가능한지 확인

        Args:
            symbol: 종목 심볼
            ticker_data: 해당 종목의 ticker 데이터

        Returns:
            거래 가능 여부
        """
        try:
            # 제외 목록 확인
            if symbol in self.EXCLUDED_SYMBOLS:
                return False

            # 거래량 확인
            volume_24h = ticker_data.get('acc_trade_value_24H', '0')
            try:
                volume_krw = float(volume_24h)
                if volume_krw < self.MIN_DAILY_VOLUME_KRW:
                    return False
            except (ValueError, TypeError):
                return False

            # 가격 정보 확인
            closing_price = ticker_data.get('closing_price', '0')
            try:
                price = float(closing_price)
                if price <= 0:
                    return False
            except (ValueError, TypeError):
                return False

            # 거래 상태 확인 (가능하다면)
            # 빗썸 API에서 거래 정지 정보가 있다면 여기서 확인

            return True

        except Exception as e:
            logger.warning(f"{symbol} 거래 가능성 확인 중 오류: {str(e)}")
            return False

    async def evaluate_symbols(self, symbols: List[str]) -> List[AssetScore]:
        """
        종목들에 대한 종합 평가 수행

        Args:
            symbols: 평가할 종목 리스트

        Returns:
            평가된 종목 점수 리스트
        """
        try:
            await self._ensure_services()

            logger.info(f"{len(symbols)}개 종목에 대한 종합 평가 시작")

            # 병렬로 모든 종목 평가 (동시성 제한)
            semaphore = asyncio.Semaphore(10)  # 최대 10개 동시 실행
            tasks = []

            async def evaluate_single_symbol(symbol: str) -> Optional[AssetScore]:
                """단일 종목 평가"""
                async with semaphore:
                    try:
                        # ScoringSystem을 통한 종합 점수 계산
                        result = await self._scoring_system.calculate_comprehensive_score(
                            symbol=symbol,
                            enable_dynamic_weights=True,
                            filter_outliers=True
                        )

                        if not result:
                            logger.warning(f"{symbol} 점수 계산 실패")
                            return None

                        return AssetScore(
                            symbol=symbol,
                            score=result['comprehensive_score'],
                            reliability=result['quality_assessment']['overall_quality'],
                            recommendation=result['recommendation'],
                            analysis_timestamp=result['timestamp'],
                            quality_assessment=result['quality_assessment']
                        )

                    except Exception as e:
                        logger.error(f"{symbol} 평가 중 오류: {str(e)}")
                        return None

            # 모든 종목에 대한 평가 태스크 생성
            for symbol in symbols:
                tasks.append(evaluate_single_symbol(symbol))

            # 병렬 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과 정리
            asset_scores = []
            for result in results:
                if isinstance(result, AssetScore):
                    asset_scores.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"종목 평가 중 예외: {str(result)}")

            logger.info(
                f"종목 평가 완료 - 성공: {len(asset_scores)}, "
                f"실패: {len(symbols) - len(asset_scores)}"
            )

            return asset_scores

        except Exception as e:
            logger.error(f"종목 평가 중 오류: {str(e)}")
            return []

    def filter_and_rank_assets(self, asset_scores: List[AssetScore]) -> List[AssetScore]:
        """
        종목들을 필터링하고 순위 매기기

        Args:
            asset_scores: 평가된 종목 점수 리스트

        Returns:
            필터링되고 순위가 매겨진 종목 리스트
        """
        try:
            # 1단계: 기본 필터링
            filtered_assets = []

            for asset in asset_scores:
                # 최소 점수 기준 확인
                if asset.score < self.MIN_SCORE_THRESHOLD:
                    continue

                # 최소 신뢰도 기준 확인
                if asset.reliability < self.MIN_RELIABILITY_THRESHOLD:
                    continue

                # 추가 품질 확인
                if not asset.is_reliable:
                    continue

                filtered_assets.append(asset)

            # 2단계: 점수 기준 정렬 (내림차순)
            ranked_assets = sorted(
                filtered_assets,
                key=lambda x: (x.score, x.reliability),
                reverse=True
            )

            logger.info(
                f"필터링 완료 - 전체: {len(asset_scores)}, "
                f"필터링 후: {len(ranked_assets)}"
            )

            return ranked_assets

        except Exception as e:
            logger.error(f"종목 필터링 및 순위 매기기 중 오류: {str(e)}")
            return []

    def select_top_assets(
        self,
        ranked_assets: List[AssetScore],
        count: int = None
    ) -> List[AssetScore]:
        """
        상위 N개 종목 선별

        Args:
            ranked_assets: 순위가 매겨진 종목 리스트
            count: 선별할 종목 수 (None이면 기본값 사용)

        Returns:
            선별된 상위 종목 리스트
        """
        try:
            if count is None:
                count = self.DEFAULT_TOP_COUNT

            # 상위 N개 선별
            selected = ranked_assets[:count]

            logger.info(f"상위 {len(selected)}개 종목 선별 완료")

            # 선별된 종목 정보 로깅
            for i, asset in enumerate(selected, 1):
                logger.info(
                    f"{i}순위: {asset.symbol} - 점수: {asset.score:.2f}, "
                    f"신뢰도: {asset.reliability:.2f}, 등급: {asset.grade}"
                )

            return selected

        except Exception as e:
            logger.error(f"상위 종목 선별 중 오류: {str(e)}")
            return []

    def analyze_portfolio_changes(
        self,
        new_selection: List[AssetScore],
        current_portfolio: Optional[List[str]] = None
    ) -> List[PortfolioChange]:
        """
        포트폴리오 변경 사항 분석

        Args:
            new_selection: 새로 선별된 종목들
            current_portfolio: 현재 포트폴리오 종목들

        Returns:
            포트폴리오 변경 사항 리스트
        """
        try:
            changes = []

            # 현재 포트폴리오가 없으면 모두 신규 추가
            if not current_portfolio:
                for asset in new_selection:
                    changes.append(PortfolioChange(
                        action='add',
                        symbol=asset.symbol,
                        score_improvement=asset.score,
                        reason=f"신규 선별 (점수: {asset.score:.1f})"
                    ))
                return changes

            # 기존 종목과 신규 종목 비교
            new_symbols = {asset.symbol for asset in new_selection}
            current_symbols = set(current_portfolio)

            # 추가된 종목
            added_symbols = new_symbols - current_symbols
            for symbol in added_symbols:
                asset = next((a for a in new_selection if a.symbol == symbol), None)
                if asset:
                    changes.append(PortfolioChange(
                        action='add',
                        symbol=symbol,
                        score_improvement=asset.score,
                        reason=f"신규 추가 (점수: {asset.score:.1f})"
                    ))

            # 제거된 종목
            removed_symbols = current_symbols - new_symbols
            for symbol in removed_symbols:
                changes.append(PortfolioChange(
                    action='remove',
                    symbol=symbol,
                    reason="선정 기준 미달"
                ))

            # 교체 상황 분석
            if self._last_selection and self._last_selection.selected_assets:
                old_assets = {a.symbol: a for a in self._last_selection.selected_assets}
                new_assets = {a.symbol: a for a in new_selection}

                for new_symbol, new_asset in new_assets.items():
                    if new_symbol in old_assets:
                        old_asset = old_assets[new_symbol]
                        score_diff = new_asset.score - old_asset.score

                        if abs(score_diff) > self.SCORE_DIFFERENCE_THRESHOLD:
                            changes.append(PortfolioChange(
                                action='replace',
                                symbol=new_symbol,
                                old_symbol=new_symbol,
                                score_improvement=score_diff,
                                reason=f"점수 변화: {score_diff:+.1f}"
                            ))

            return changes

        except Exception as e:
            logger.error(f"포트폴리오 변경 분석 중 오류: {str(e)}")
            return []

    async def run_asset_selection(
        self,
        target_count: int = None,
        current_portfolio: Optional[List[str]] = None,
        custom_criteria: Optional[Dict[str, Any]] = None
    ) -> Optional[SelectionResult]:
        """
        전체 종목 선정 프로세스 실행

        Args:
            target_count: 선별할 종목 수
            current_portfolio: 현재 포트폴리오
            custom_criteria: 사용자 정의 선정 기준

        Returns:
            종목 선정 결과
        """
        try:
            start_time = datetime.now(timezone.utc)
            logger.info("종목 선정 프로세스 시작")

            self._stats['total_selections'] += 1

            if target_count is None:
                target_count = self.DEFAULT_TOP_COUNT

            # 1단계: 거래 가능 종목 목록 수집
            available_symbols = await self.get_available_symbols()
            if not available_symbols:
                logger.error("거래 가능 종목을 찾을 수 없습니다")
                return None

            # 2단계: 모든 종목 평가
            asset_scores = await self.evaluate_symbols(available_symbols)
            if not asset_scores:
                logger.error("종목 평가 결과가 없습니다")
                return None

            # 3단계: 필터링 및 순위 매기기
            ranked_assets = self.filter_and_rank_assets(asset_scores)
            if len(ranked_assets) < target_count:
                logger.warning(
                    f"필터링된 종목 수({len(ranked_assets)})가 "
                    f"목표 수({target_count})보다 적습니다"
                )

            # 4단계: 상위 종목 선별
            selected_assets = self.select_top_assets(ranked_assets, target_count)
            if not selected_assets:
                logger.error("선별된 종목이 없습니다")
                return None

            # 5단계: 제외된 종목들
            excluded_assets = [asset for asset in asset_scores if asset not in selected_assets]

            # 6단계: 포트폴리오 변경 분석
            portfolio_changes = self.analyze_portfolio_changes(selected_assets, current_portfolio)

            # 7단계: 결과 생성
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            selection_result = SelectionResult(
                timestamp=end_time,
                selected_assets=selected_assets,
                excluded_assets=excluded_assets,
                portfolio_changes=portfolio_changes,
                selection_criteria={
                    'min_score_threshold': self.MIN_SCORE_THRESHOLD,
                    'min_reliability_threshold': self.MIN_RELIABILITY_THRESHOLD,
                    'target_count': target_count,
                    'custom_criteria': custom_criteria or {}
                },
                total_evaluated=len(asset_scores),
                success_rate=len(asset_scores) / len(available_symbols) if available_symbols else 0.0
            )

            # 통계 업데이트
            self._stats['successful_selections'] += 1
            self._stats['last_selection'] = end_time
            self._stats['avg_evaluation_time'] = (
                (self._stats['avg_evaluation_time'] + duration) / 2
                if self._stats['avg_evaluation_time'] > 0 else duration
            )

            # 결과 캐시
            self._last_selection = selection_result

            logger.info(
                f"종목 선정 완료 - 소요시간: {duration:.2f}초, "
                f"선별: {len(selected_assets)}개, "
                f"변경: {len(portfolio_changes)}건"
            )

            return selection_result

        except Exception as e:
            logger.error(f"종목 선정 프로세스 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def get_selection_summary(self, result: SelectionResult) -> Dict[str, Any]:
        """
        선정 결과 요약 정보 생성

        Args:
            result: 종목 선정 결과

        Returns:
            요약 정보 딕셔너리
        """
        try:
            # 선별된 종목 통계
            selected_scores = [asset.score for asset in result.selected_assets]
            selected_reliabilities = [asset.reliability for asset in result.selected_assets]

            # 변경 사항 통계
            change_counts = {}
            for change in result.portfolio_changes:
                change_counts[change.action] = change_counts.get(change.action, 0) + 1

            summary = {
                'selection_timestamp': result.timestamp.isoformat(),
                'total_evaluated': result.total_evaluated,
                'success_rate': f"{result.success_rate:.1%}",
                'selected_count': len(result.selected_assets),
                'score_statistics': {
                    'average': sum(selected_scores) / len(selected_scores) if selected_scores else 0,
                    'highest': max(selected_scores) if selected_scores else 0,
                    'lowest': min(selected_scores) if selected_scores else 0
                },
                'reliability_statistics': {
                    'average': sum(selected_reliabilities) / len(selected_reliabilities) if selected_reliabilities else 0,
                    'highest': max(selected_reliabilities) if selected_reliabilities else 0,
                    'lowest': min(selected_reliabilities) if selected_reliabilities else 0
                },
                'portfolio_changes': change_counts,
                'selected_assets': [
                    {
                        'symbol': asset.symbol,
                        'score': asset.score,
                        'reliability': asset.reliability,
                        'grade': asset.grade
                    }
                    for asset in result.selected_assets
                ],
                'selection_criteria': result.selection_criteria
            }

            return summary

        except Exception as e:
            logger.error(f"선정 결과 요약 생성 중 오류: {str(e)}")
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계 정보 반환"""
        return self._stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """
        시스템 상태 확인

        Returns:
            상태 정보 딕셔너리
        """
        try:
            await self._ensure_services()

            # 의존 서비스들 상태 확인
            scoring_health = await self._scoring_system.health_check()
            http_health = await self._http_client.health_check()
            db_health = await self.db_config.health_check()

            return {
                'service_name': 'AssetSelector',
                'database_connected': db_health,
                'scoring_system_healthy': scoring_health.get('all_services_healthy', False),
                'http_client_healthy': http_health,
                'all_services_healthy': (
                    db_health and
                    scoring_health.get('all_services_healthy', False) and
                    http_health
                ),
                'stats': self.get_stats()
            }

        except Exception as e:
            logger.error(f"상태 확인 중 오류: {str(e)}")
            return {
                'service_name': 'AssetSelector',
                'database_connected': False,
                'scoring_system_healthy': False,
                'http_client_healthy': False,
                'all_services_healthy': False,
                'stats': self.get_stats(),
                'error': str(e)
            }

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self._ensure_services()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        # 의존 서비스들 정리
        if self._scoring_system:
            try:
                await self._scoring_system.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.error(f"ScoringSystem 정리 중 오류: {str(e)}")

        if self._http_client:
            try:
                await self._http_client.close()
            except Exception as e:
                logger.error(f"HTTP 클라이언트 정리 중 오류: {str(e)}")


# 전역 인스턴스를 위한 팩토리 함수
_global_asset_selector: Optional[AssetSelector] = None


async def get_asset_selector(db_config: Optional[DatabaseConfig] = None) -> AssetSelector:
    """
    전역 AssetSelector 인스턴스 반환

    Args:
        db_config: 데이터베이스 설정 객체 (선택적, None이면 기본값 생성)

    Returns:
        AssetSelector 인스턴스
    """
    global _global_asset_selector

    if db_config is None:
        db_config = DatabaseConfig()

    if _global_asset_selector is None:
        _global_asset_selector = AssetSelector(db_config)

    return _global_asset_selector


async def close_asset_selector():
    """전역 AssetSelector 인스턴스 정리"""
    global _global_asset_selector

    if _global_asset_selector is not None:
        await _global_asset_selector.__aexit__(None, None, None)
        _global_asset_selector = None