"""
거래량 분석 서비스

빗썸 API에서 24시간 거래량 데이터를 수집하고 분석하는 서비스입니다.
거래량 급등 감지, 평균 대비 비교 분석, 시간대별 패턴 분석 등을 제공합니다.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal, InvalidOperation

from src.api.bithumb.client import get_http_client, BithumbHTTPClient
from src.database.config import DatabaseConfig
from src.database.models.market import Ticker, Transaction
from sqlalchemy import select, func, and_
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class VolumeAnalyzer:
    """
    거래량 분석 서비스

    빗썸 API와 데이터베이스를 활용하여 거래량 데이터를 분석하고
    거래량 기반 투자 신호를 생성하는 서비스
    """

    def __init__(self, db_config: DatabaseConfig):
        """
        거래량 분석기 초기화

        Args:
            db_config: 데이터베이스 설정 객체
        """
        self.db_config = db_config
        self._http_client: Optional[BithumbHTTPClient] = None
        self._stats = {
            'analysis_count': 0,
            'api_calls': 0,
            'db_queries': 0,
            'errors': 0,
            'last_analysis': None
        }

    async def _ensure_http_client(self):
        """HTTP 클라이언트 확인 및 초기화"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = await get_http_client()

    async def get_24h_volume_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        24시간 거래량 데이터 수집

        Args:
            symbol: 분석할 심볼 (예: "BTC_KRW")

        Returns:
            거래량 데이터 딕셔너리 또는 None
        """
        try:
            await self._ensure_http_client()

            # 빗썸 공개 API에서 24시간 거래량 데이터 수집
            endpoint = f"/public/ticker/{symbol}"
            response = await self._http_client.get(endpoint)

            self._stats['api_calls'] += 1

            if response.get('status') == '0000' and 'data' in response:
                ticker_data = response['data']

                # 거래량 데이터 추출 및 정제
                volume_data = {
                    'symbol': symbol,
                    'volume_24h': self._safe_decimal(ticker_data.get('units_traded_24H')),
                    'volume_value_24h': self._safe_decimal(ticker_data.get('acc_trade_value_24H')),
                    'closing_price': self._safe_decimal(ticker_data.get('closing_price')),
                    'prev_closing_price': self._safe_decimal(ticker_data.get('prev_closing_price')),
                    'fluctate_24h': self._safe_decimal(ticker_data.get('fluctate_24H')),
                    'fluctate_rate_24h': self._safe_decimal(ticker_data.get('fluctate_rate_24H')),
                    'timestamp': datetime.now(timezone.utc)
                }

                logger.debug(f"{symbol} 24시간 거래량 데이터 수집 완료: {volume_data['volume_24h']}")
                return volume_data
            else:
                logger.warning(f"{symbol} 거래량 데이터 수집 실패: {response}")
                return None

        except Exception as e:
            logger.error(f"{symbol} 24시간 거래량 데이터 수집 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def get_historical_volume_avg(
        self,
        symbol: str,
        days: int = 7
    ) -> Optional[Decimal]:
        """
        과거 N일간 평균 거래량 계산

        Args:
            symbol: 분석할 심볼
            days: 분석 기간 (일)

        Returns:
            평균 거래량 또는 None
        """
        try:
            # 데이터베이스 초기화 확인
            if not await self.db_config.health_check():
                logger.warning(f"{symbol} 과거 거래량 계산: 데이터베이스 연결 비활성")
                return None

            async with self.db_config.get_session() as session:
                # 과거 N일간 데이터 조회
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=days)

                # Ticker 테이블에서 과거 거래량 데이터 조회
                query = select(Ticker.volume_24h).where(
                    and_(
                        Ticker.symbol == symbol,
                        Ticker.created_at >= start_time,
                        Ticker.created_at <= end_time,
                        Ticker.volume_24h.is_not(None),
                        Ticker.volume_24h > 0
                    )
                ).order_by(Ticker.created_at.desc())

                result = await session.execute(query)
                volumes = [row[0] for row in result.fetchall()]

                self._stats['db_queries'] += 1

                if volumes:
                    avg_volume = sum(volumes) / len(volumes)
                    logger.debug(f"{symbol} {days}일 평균 거래량: {avg_volume} (샘플: {len(volumes)}개)")
                    return avg_volume
                else:
                    logger.warning(f"{symbol} {days}일간 거래량 데이터가 부족합니다")
                    return None

        except SQLAlchemyError as e:
            logger.error(f"{symbol} 과거 거래량 조회 중 DB 오류: {str(e)}")
            self._stats['errors'] += 1
            return None
        except Exception as e:
            logger.error(f"{symbol} 과거 거래량 계산 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def calculate_volume_surge_score(
        self,
        symbol: str,
        current_volume: Decimal,
        lookback_days: int = 7
    ) -> Optional[float]:
        """
        거래량 급등 점수 계산

        표준편차를 활용하여 현재 거래량이 과거 평균 대비 얼마나 높은지 계산

        Args:
            symbol: 분석할 심볼
            current_volume: 현재 거래량
            lookback_days: 비교 기간 (일)

        Returns:
            급등 점수 (0.0 ~ 10.0) 또는 None
        """
        try:
            # 데이터베이스 초기화 확인
            if not await self.db_config.health_check():
                logger.warning(f"{symbol} 거래량 급등 점수 계산: 데이터베이스 연결 비활성")
                return None

            async with self.db_config.get_session() as session:
                # 과거 거래량 데이터 수집
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=lookback_days)

                query = select(Ticker.volume_24h).where(
                    and_(
                        Ticker.symbol == symbol,
                        Ticker.created_at >= start_time,
                        Ticker.created_at <= end_time,
                        Ticker.volume_24h.is_not(None),
                        Ticker.volume_24h > 0
                    )
                ).order_by(Ticker.created_at.desc()).limit(50)  # 최대 50개 샘플

                result = await session.execute(query)
                historical_volumes = [float(row[0]) for row in result.fetchall()]

                self._stats['db_queries'] += 1

                if len(historical_volumes) < 3:
                    logger.warning(f"{symbol} 거래량 급등 분석을 위한 데이터가 부족합니다")
                    return None

                # 통계 계산
                mean_volume = statistics.mean(historical_volumes)
                stdev_volume = statistics.stdev(historical_volumes) if len(historical_volumes) > 1 else 0

                current_volume_float = float(current_volume)

                # Z-score 계산 (표준편차 기반)
                if stdev_volume > 0:
                    z_score = (current_volume_float - mean_volume) / stdev_volume
                else:
                    z_score = 0

                # 급등 점수 계산 (0 ~ 10 스케일)
                # Z-score 2.0 이상을 최고점(10)으로 설정
                surge_score = max(0.0, min(10.0, (z_score / 2.0) * 10.0))

                logger.debug(
                    f"{symbol} 거래량 급등 분석 - "
                    f"현재: {current_volume_float:.2f}, "
                    f"평균: {mean_volume:.2f}, "
                    f"표준편차: {stdev_volume:.2f}, "
                    f"Z-score: {z_score:.2f}, "
                    f"급등점수: {surge_score:.2f}"
                )

                return surge_score

        except Exception as e:
            logger.error(f"{symbol} 거래량 급등 점수 계산 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def analyze_hourly_volume_pattern(
        self,
        symbol: str,
        hours_back: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        시간대별 거래량 패턴 분석

        Args:
            symbol: 분석할 심볼
            hours_back: 분석할 시간 범위 (시간)

        Returns:
            시간대별 패턴 분석 결과
        """
        try:
            # 데이터베이스 초기화 확인
            if not await self.db_config.health_check():
                logger.warning(f"{symbol} 시간대별 패턴 분석: 데이터베이스 연결 비활성")
                return None

            async with self.db_config.get_session() as session:
                # 최근 N시간 거래 데이터 조회
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=hours_back)

                # Transaction 테이블에서 시간별 거래량 집계
                query = select(
                    func.date_trunc('hour', Transaction.timestamp).label('hour'),
                    func.sum(Transaction.quantity).label('volume'),
                    func.count(Transaction.id).label('trade_count')
                ).where(
                    and_(
                        Transaction.symbol == symbol,
                        Transaction.timestamp >= start_time,
                        Transaction.timestamp <= end_time
                    )
                ).group_by(
                    func.date_trunc('hour', Transaction.timestamp)
                ).order_by('hour')

                result = await session.execute(query)
                hourly_data = result.fetchall()

                self._stats['db_queries'] += 1

                if not hourly_data:
                    logger.warning(f"{symbol} 시간별 거래량 데이터가 없습니다")
                    return None

                # 시간별 데이터 정리
                volumes = [float(row.volume) for row in hourly_data]
                trade_counts = [row.trade_count for row in hourly_data]

                if not volumes:
                    return None

                # 통계 계산
                total_volume = sum(volumes)
                avg_volume = statistics.mean(volumes)
                max_volume = max(volumes)
                min_volume = min(volumes)
                volume_variance = statistics.variance(volumes) if len(volumes) > 1 else 0

                # 정규화 점수 계산 (0 ~ 100)
                if max_volume > min_volume:
                    current_volume = volumes[-1] if volumes else avg_volume
                    normalized_score = ((current_volume - min_volume) / (max_volume - min_volume)) * 100
                else:
                    normalized_score = 50.0  # 변화가 없는 경우 중간값

                # 시간대별 분포 계산
                peak_hours = []
                for i, (hour, volume, count) in enumerate(zip([row.hour for row in hourly_data], volumes, trade_counts)):
                    if volume >= avg_volume * 1.5:  # 평균 대비 150% 이상
                        peak_hours.append({
                            'hour': hour.hour if hasattr(hour, 'hour') else hour,
                            'volume': volume,
                            'trade_count': count
                        })

                pattern_analysis = {
                    'symbol': symbol,
                    'analysis_period_hours': hours_back,
                    'total_volume': total_volume,
                    'avg_hourly_volume': avg_volume,
                    'max_hourly_volume': max_volume,
                    'min_hourly_volume': min_volume,
                    'volume_variance': volume_variance,
                    'normalized_score': normalized_score,
                    'peak_hours': peak_hours,
                    'total_trades': sum(trade_counts),
                    'avg_hourly_trades': statistics.mean(trade_counts),
                    'timestamp': datetime.now(timezone.utc)
                }

                logger.debug(f"{symbol} 시간대별 거래량 패턴 분석 완료: 정규화 점수 {normalized_score:.2f}")
                return pattern_analysis

        except Exception as e:
            logger.error(f"{symbol} 시간대별 거래량 패턴 분석 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def comprehensive_volume_analysis(
        self,
        symbol: str,
        include_patterns: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        종합적인 거래량 분석

        24시간 거래량, 급등 점수, 패턴 분석을 모두 포함하는 종합 분석

        Args:
            symbol: 분석할 심볼
            include_patterns: 패턴 분석 포함 여부

        Returns:
            종합 거래량 분석 결과
        """
        try:
            logger.info(f"{symbol} 종합 거래량 분석 시작")

            # 1. 24시간 거래량 데이터 수집
            current_data = await self.get_24h_volume_data(symbol)
            if not current_data:
                logger.error(f"{symbol} 현재 거래량 데이터 수집 실패")
                return None

            # 2. 과거 평균 거래량 계산
            avg_volume_7d = await self.get_historical_volume_avg(symbol, 7)
            avg_volume_30d = await self.get_historical_volume_avg(symbol, 30)

            # 3. 거래량 급등 점수 계산
            surge_score = None
            if current_data.get('volume_24h'):
                surge_score = await self.calculate_volume_surge_score(
                    symbol,
                    current_data['volume_24h']
                )

            # 4. 시간대별 패턴 분석 (선택적)
            pattern_analysis = None
            if include_patterns:
                pattern_analysis = await self.analyze_hourly_volume_pattern(symbol)

            # 5. 종합 점수 계산
            comprehensive_score = self._calculate_comprehensive_score(
                current_data, avg_volume_7d, avg_volume_30d, surge_score
            )

            # 결과 정리
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'current_volume_24h': current_data.get('volume_24h'),
                'current_volume_value_24h': current_data.get('volume_value_24h'),
                'closing_price': current_data.get('closing_price'),
                'price_change_24h': current_data.get('fluctate_24h'),
                'price_change_rate_24h': current_data.get('fluctate_rate_24h'),
                'avg_volume_7d': avg_volume_7d,
                'avg_volume_30d': avg_volume_30d,
                'surge_score': surge_score,
                'comprehensive_score': comprehensive_score,
                'pattern_analysis': pattern_analysis,
                'volume_ratios': {
                    'vs_7d_avg': self._calculate_ratio(current_data.get('volume_24h'), avg_volume_7d),
                    'vs_30d_avg': self._calculate_ratio(current_data.get('volume_24h'), avg_volume_30d)
                }
            }

            self._stats['analysis_count'] += 1
            self._stats['last_analysis'] = datetime.now(timezone.utc)

            surge_score_str = f"{surge_score:.2f}" if surge_score is not None else "N/A"
            logger.info(
                f"{symbol} 종합 거래량 분석 완료 - "
                f"현재거래량: {current_data.get('volume_24h')}, "
                f"급등점수: {surge_score_str}, "
                f"종합점수: {comprehensive_score:.2f}"
            )

            return analysis_result

        except Exception as e:
            logger.error(f"{symbol} 종합 거래량 분석 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    def _calculate_comprehensive_score(
        self,
        current_data: Dict[str, Any],
        avg_7d: Optional[Decimal],
        avg_30d: Optional[Decimal],
        surge_score: Optional[float]
    ) -> float:
        """
        종합 점수 계산 (0 ~ 100)

        여러 지표를 종합하여 최종 점수를 계산
        """
        try:
            score = 0.0
            weight_sum = 0.0

            current_volume = current_data.get('volume_24h')

            # 1. 거래량 급등 점수 (가중치: 40%)
            if surge_score is not None:
                score += surge_score * 4.0  # 10점 만점을 40점으로 변환
                weight_sum += 40.0

            # 2. 7일 평균 대비 비율 (가중치: 30%)
            if current_volume and avg_7d:
                ratio_7d = float(current_volume / avg_7d)
                # 2배 이상을 만점으로 설정
                ratio_score = min(30.0, (ratio_7d - 1.0) * 30.0)
                score += max(0.0, ratio_score)
                weight_sum += 30.0

            # 3. 30일 평균 대비 비율 (가중치: 20%)
            if current_volume and avg_30d:
                ratio_30d = float(current_volume / avg_30d)
                ratio_score = min(20.0, (ratio_30d - 1.0) * 20.0)
                score += max(0.0, ratio_score)
                weight_sum += 20.0

            # 4. 가격 변동률 연동 (가중치: 10%)
            price_change_rate = current_data.get('fluctate_rate_24h')
            if price_change_rate:
                # 절댓값 5% 이상 변동을 만점으로 설정
                price_score = min(10.0, abs(float(price_change_rate)) * 2.0)
                score += price_score
                weight_sum += 10.0

            # 가중 평균 계산
            if weight_sum > 0:
                final_score = score / weight_sum * 100.0
            else:
                final_score = 0.0

            return max(0.0, min(100.0, final_score))

        except Exception as e:
            logger.error(f"종합 점수 계산 중 오류: {str(e)}")
            return 0.0

    def _calculate_ratio(self, current: Optional[Decimal], average: Optional[Decimal]) -> Optional[float]:
        """안전한 비율 계산"""
        try:
            if current and average and average > 0:
                return float(current / average)
            return None
        except Exception:
            return None

    def _safe_decimal(self, value: Any) -> Optional[Decimal]:
        """안전한 Decimal 변환"""
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (ValueError, TypeError, InvalidOperation):
            logger.warning(f"Decimal 변환 실패: {value}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """분석 서비스 통계 정보 반환"""
        return self._stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """
        서비스 상태 확인

        Returns:
            상태 정보 딕셔너리
        """
        http_client_status = False
        try:
            await self._ensure_http_client()
            http_client_status = not self._http_client.is_closed
        except Exception as e:
            logger.error(f"HTTP 클라이언트 상태 확인 중 오류: {str(e)}")

        return {
            'service_name': 'VolumeAnalyzer',
            'http_client_available': http_client_status,
            'database_connected': await self.db_config.health_check(),
            'stats': self.get_stats()
        }

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.close()


# 전역 인스턴스를 위한 팩토리 함수
_global_volume_analyzer: Optional[VolumeAnalyzer] = None


async def get_volume_analyzer(db_config: Optional[DatabaseConfig] = None) -> VolumeAnalyzer:
    """
    전역 VolumeAnalyzer 인스턴스 반환

    Args:
        db_config: 데이터베이스 설정 객체 (None이면 기본 설정 사용)

    Returns:
        VolumeAnalyzer 인스턴스
    """
    global _global_volume_analyzer

    if _global_volume_analyzer is None:
        if db_config is None:
            db_config = DatabaseConfig()
        _global_volume_analyzer = VolumeAnalyzer(db_config)

    return _global_volume_analyzer


async def close_volume_analyzer():
    """전역 VolumeAnalyzer 인스턴스 정리"""
    global _global_volume_analyzer

    if _global_volume_analyzer is not None:
        await _global_volume_analyzer.__aexit__(None, None, None)
        _global_volume_analyzer = None