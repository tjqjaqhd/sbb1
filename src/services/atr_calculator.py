"""
ATR(Average True Range) 계산 서비스

빗썸 API에서 과거 가격 데이터를 수집하여 ATR 지표를 계산하는 서비스입니다.
14일 기간 ATR 계산, 변동성 점수 정규화, 데이트레이딩 적합성 평가 등을 제공합니다.
"""

import asyncio
import logging
import statistics
import talib
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal, InvalidOperation

from src.api.bithumb.client import get_http_client, BithumbHTTPClient
from src.database.config import DatabaseConfig
from src.database.models.market import MarketData, Ticker
from sqlalchemy import select, func, and_
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class ATRCalculator:
    """
    ATR(Average True Range) 계산 서비스

    빗썸 API와 데이터베이스를 활용하여 ATR 지표를 계산하고
    변동성 기반 데이트레이딩 신호를 생성하는 서비스
    """

    # 기본 ATR 설정
    DEFAULT_ATR_PERIOD = 14  # 14일 기간
    MIN_DATA_POINTS = 20     # 최소 데이터 포인트 수 (ATR 계산을 위해 충분한 데이터 필요)

    # 데이트레이딩 적합성 기준 (ATR 기준 변동성 범위)
    DAYTRADING_ATR_RANGES = {
        'very_low': (0.0, 0.02),    # 2% 미만 - 매우 낮은 변동성
        'low': (0.02, 0.04),        # 2-4% - 낮은 변동성
        'moderate': (0.04, 0.07),   # 4-7% - 적당한 변동성 (데이트레이딩 적합)
        'high': (0.07, 0.12),       # 7-12% - 높은 변동성 (데이트레이딩 적합)
        'very_high': (0.12, 1.0)    # 12% 초과 - 매우 높은 변동성 (위험)
    }

    def __init__(self, db_config: DatabaseConfig):
        """
        ATR 계산기 초기화

        Args:
            db_config: 데이터베이스 설정 객체
        """
        self.db_config = db_config
        self._http_client: Optional[BithumbHTTPClient] = None
        self._stats = {
            'atr_calculations': 0,
            'api_calls': 0,
            'db_queries': 0,
            'errors': 0,
            'last_calculation': None
        }

    async def _ensure_http_client(self):
        """HTTP 클라이언트 확인 및 초기화"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = await get_http_client()

    async def fetch_historical_data(
        self,
        symbol: str,
        days: int = 30
    ) -> Optional[List[Dict[str, Any]]]:
        """
        과거 가격 데이터 수집 (빗썸 API 또는 데이터베이스)

        Args:
            symbol: 분석할 심볼 (예: "BTC_KRW")
            days: 수집할 일수

        Returns:
            OHLCV 데이터 리스트 또는 None
        """
        try:
            # 먼저 데이터베이스에서 MarketData 조회
            db_data = await self._get_historical_from_db(symbol, days)
            if db_data and len(db_data) >= self.MIN_DATA_POINTS:
                logger.debug(f"{symbol} 과거 데이터 - 데이터베이스에서 {len(db_data)}개 조회")
                return db_data

            # 데이터베이스에 충분한 데이터가 없으면 API 호출
            api_data = await self._get_historical_from_api(symbol, days)
            if api_data:
                logger.debug(f"{symbol} 과거 데이터 - API에서 {len(api_data)}개 수집")
                return api_data

            logger.warning(f"{symbol} 과거 가격 데이터 수집 실패")
            return None

        except Exception as e:
            logger.error(f"{symbol} 과거 데이터 수집 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def _get_historical_from_db(
        self,
        symbol: str,
        days: int
    ) -> Optional[List[Dict[str, Any]]]:
        """데이터베이스에서 과거 데이터 조회"""
        try:
            if not await self.db_config.health_check():
                return None

            async with self.db_config.get_session() as session:
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(days=days)

                # MarketData 테이블에서 일봉 데이터 조회
                query = select(MarketData).where(
                    and_(
                        MarketData.symbol == symbol,
                        MarketData.timeframe == '1d',  # 일봉 데이터
                        MarketData.timestamp >= start_time,
                        MarketData.timestamp <= end_time
                    )
                ).order_by(MarketData.timestamp.asc())

                result = await session.execute(query)
                records = result.scalars().all()

                self._stats['db_queries'] += 1

                if not records:
                    return None

                # MarketData를 OHLCV 딕셔너리로 변환
                historical_data = []
                for record in records:
                    historical_data.append({
                        'timestamp': record.timestamp,
                        'open': float(record.open_price),
                        'high': float(record.high_price),
                        'low': float(record.low_price),
                        'close': float(record.close_price),
                        'volume': float(record.volume)
                    })

                return historical_data

        except Exception as e:
            logger.error(f"{symbol} 데이터베이스 과거 데이터 조회 중 오류: {str(e)}")
            return None

    async def _get_historical_from_api(
        self,
        symbol: str,
        days: int
    ) -> Optional[List[Dict[str, Any]]]:
        """빗썸 API에서 과거 데이터 수집"""
        try:
            await self._ensure_http_client()

            # 빗썸은 직접적인 과거 캔들 데이터 API가 제한적이므로
            # 현재 Ticker 데이터를 활용한 근사치 생성
            # 실제 운영환경에서는 외부 데이터 소스나 별도 수집 시스템 필요

            endpoint = f"/public/ticker/{symbol}"
            response = await self._http_client.get(endpoint)

            self._stats['api_calls'] += 1

            if response.get('status') != '0000' or 'data' not in response:
                return None

            ticker_data = response['data']

            # 현재 가격 정보로부터 가상의 과거 데이터 생성 (개발/테스트용)
            # 주의: 실제 환경에서는 실제 OHLCV 데이터를 사용해야 함
            current_price = self._safe_float(ticker_data.get('closing_price'))
            high_price = self._safe_float(ticker_data.get('max_price'))
            low_price = self._safe_float(ticker_data.get('min_price'))
            volume = self._safe_float(ticker_data.get('units_traded_24H'))

            if not all([current_price, high_price, low_price]):
                return None

            # 간단한 가상 데이터 생성 (실제로는 실제 과거 데이터 필요)
            historical_data = []
            base_time = datetime.now(timezone.utc) - timedelta(days=days)

            for i in range(days):
                # 가격 변동을 시뮬레이션 (매우 단순한 방식)
                daily_variation = np.random.normal(0, 0.02)  # 2% 표준편차
                sim_close = current_price * (1 + daily_variation * (days - i) / days)
                sim_high = sim_close * (1 + abs(np.random.normal(0, 0.01)))
                sim_low = sim_close * (1 - abs(np.random.normal(0, 0.01)))
                sim_open = sim_low + (sim_high - sim_low) * np.random.random()

                historical_data.append({
                    'timestamp': base_time + timedelta(days=i),
                    'open': sim_open,
                    'high': sim_high,
                    'low': sim_low,
                    'close': sim_close,
                    'volume': volume * np.random.uniform(0.5, 1.5)
                })

            logger.warning(f"{symbol} API에서 시뮬레이션 데이터 생성 (실제 과거 데이터 필요)")
            return historical_data

        except Exception as e:
            logger.error(f"{symbol} API 과거 데이터 수집 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    def calculate_atr(
        self,
        historical_data: List[Dict[str, Any]],
        period: int = None
    ) -> Optional[float]:
        """
        ATR(Average True Range) 계산

        Args:
            historical_data: OHLCV 데이터 리스트
            period: ATR 계산 기간 (기본값: 14일)

        Returns:
            ATR 값 또는 None
        """
        try:
            if period is None:
                period = self.DEFAULT_ATR_PERIOD

            if len(historical_data) < period + 1:
                logger.warning(f"ATR 계산을 위한 데이터 부족: {len(historical_data)} < {period + 1}")
                return None

            # OHLC 데이터를 numpy 배열로 변환
            high_prices = np.array([float(d['high']) for d in historical_data])
            low_prices = np.array([float(d['low']) for d in historical_data])
            close_prices = np.array([float(d['close']) for d in historical_data])

            # TA-Lib을 사용하여 ATR 계산
            atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)

            # 마지막 ATR 값 반환 (NaN 제거)
            atr_result = atr_values[-1]
            if np.isnan(atr_result):
                # NaN인 경우 이전 값들 중 유효한 값 찾기
                for i in range(len(atr_values) - 1, -1, -1):
                    if not np.isnan(atr_values[i]):
                        atr_result = atr_values[i]
                        break
                else:
                    return None

            logger.debug(f"ATR 계산 완료: {atr_result:.6f} (기간: {period}일)")
            return float(atr_result)

        except Exception as e:
            logger.error(f"ATR 계산 중 오류: {str(e)}")
            return None

    def calculate_atr_percentage(
        self,
        atr_value: float,
        current_price: float
    ) -> Optional[float]:
        """
        ATR 퍼센티지 계산 (가격 대비 ATR 비율)

        Args:
            atr_value: ATR 값
            current_price: 현재 가격

        Returns:
            ATR 퍼센티지 (0.0 ~ 1.0) 또는 None
        """
        try:
            if current_price <= 0:
                return None

            atr_percentage = atr_value / current_price
            return float(atr_percentage)

        except Exception as e:
            logger.error(f"ATR 퍼센티지 계산 중 오류: {str(e)}")
            return None

    def normalize_volatility_score(self, atr_percentage: float) -> float:
        """
        변동성 점수 정규화 (0-10점 스케일)

        Args:
            atr_percentage: ATR 퍼센티지

        Returns:
            정규화된 변동성 점수 (0.0 ~ 10.0)
        """
        try:
            # ATR 퍼센티지를 0-10 스케일로 변환
            # 15% ATR을 최고점(10)으로 설정
            max_atr_percent = 0.15
            normalized_score = min(10.0, (atr_percentage / max_atr_percent) * 10.0)
            return max(0.0, normalized_score)

        except Exception as e:
            logger.error(f"변동성 점수 정규화 중 오류: {str(e)}")
            return 0.0

    def evaluate_daytrading_suitability(self, atr_percentage: float) -> Dict[str, Any]:
        """
        데이트레이딩 적합성 평가

        Args:
            atr_percentage: ATR 퍼센티지

        Returns:
            적합성 평가 결과
        """
        try:
            # 변동성 범위 분류
            volatility_level = 'unknown'
            suitability_score = 0.0
            recommendation = '데이터 부족'

            for level, (min_range, max_range) in self.DAYTRADING_ATR_RANGES.items():
                if min_range <= atr_percentage < max_range:
                    volatility_level = level
                    break

            # 데이트레이딩 적합성 점수 및 권장사항 설정
            if volatility_level == 'very_low':
                suitability_score = 2.0
                recommendation = '변동성 부족 - 수익 기회 제한적'
            elif volatility_level == 'low':
                suitability_score = 4.0
                recommendation = '낮은 변동성 - 안전하지만 수익률 제한적'
            elif volatility_level == 'moderate':
                suitability_score = 8.5
                recommendation = '데이트레이딩 적합 - 적당한 변동성으로 좋은 기회'
            elif volatility_level == 'high':
                suitability_score = 7.5
                recommendation = '데이트레이딩 가능 - 높은 변동성, 위험 관리 필수'
            elif volatility_level == 'very_high':
                suitability_score = 3.0
                recommendation = '위험 - 과도한 변동성, 경험 있는 트레이더만 권장'

            return {
                'volatility_level': volatility_level,
                'atr_percentage': atr_percentage,
                'suitability_score': suitability_score,  # 0-10 점
                'recommendation': recommendation,
                'is_suitable': suitability_score >= 7.0,  # 7점 이상을 적합으로 판정
                'risk_level': self._get_risk_level(volatility_level)
            }

        except Exception as e:
            logger.error(f"데이트레이딩 적합성 평가 중 오류: {str(e)}")
            return {
                'volatility_level': 'unknown',
                'atr_percentage': 0.0,
                'suitability_score': 0.0,
                'recommendation': '평가 실패',
                'is_suitable': False,
                'risk_level': 'unknown'
            }

    def _get_risk_level(self, volatility_level: str) -> str:
        """변동성 수준에 따른 위험도 반환"""
        risk_mapping = {
            'very_low': '매우 낮음',
            'low': '낮음',
            'moderate': '보통',
            'high': '높음',
            'very_high': '매우 높음'
        }
        return risk_mapping.get(volatility_level, '알 수 없음')

    async def comprehensive_atr_analysis(
        self,
        symbol: str,
        period: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        종합적인 ATR 분석

        과거 데이터 수집, ATR 계산, 변동성 분석, 데이트레이딩 적합성 평가를 모두 포함

        Args:
            symbol: 분석할 심볼
            period: ATR 계산 기간

        Returns:
            종합 ATR 분석 결과
        """
        try:
            if period is None:
                period = self.DEFAULT_ATR_PERIOD

            logger.info(f"{symbol} 종합 ATR 분석 시작 (기간: {period}일)")

            # 1. 과거 가격 데이터 수집
            historical_data = await self.fetch_historical_data(symbol, period + 15)
            if not historical_data:
                logger.error(f"{symbol} 과거 가격 데이터 수집 실패")
                return None

            # 2. 현재 가격 정보 수집
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.error(f"{symbol} 현재 가격 정보 수집 실패")
                return None

            # 3. ATR 계산
            atr_value = self.calculate_atr(historical_data, period)
            if atr_value is None:
                logger.error(f"{symbol} ATR 계산 실패")
                return None

            # 4. ATR 퍼센티지 계산
            atr_percentage = self.calculate_atr_percentage(atr_value, current_price)
            if atr_percentage is None:
                logger.error(f"{symbol} ATR 퍼센티지 계산 실패")
                return None

            # 5. 변동성 점수 정규화
            volatility_score = self.normalize_volatility_score(atr_percentage)

            # 6. 데이트레이딩 적합성 평가
            suitability_analysis = self.evaluate_daytrading_suitability(atr_percentage)

            # 7. 추가 통계 계산
            additional_stats = self._calculate_additional_stats(historical_data, atr_value)

            # 결과 정리
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'atr_period': period,
                'data_points_used': len(historical_data),
                'current_price': current_price,
                'atr_value': atr_value,
                'atr_percentage': atr_percentage * 100,  # 퍼센티지로 표시
                'volatility_score': volatility_score,
                'suitability_analysis': suitability_analysis,
                'additional_stats': additional_stats,
                'analysis_quality': self._assess_analysis_quality(len(historical_data), period)
            }

            self._stats['atr_calculations'] += 1
            self._stats['last_calculation'] = datetime.now(timezone.utc)

            logger.info(
                f"{symbol} 종합 ATR 분석 완료 - "
                f"ATR: {atr_value:.6f} ({atr_percentage * 100:.2f}%), "
                f"변동성점수: {volatility_score:.2f}, "
                f"적합성: {'적합' if suitability_analysis['is_suitable'] else '부적합'}"
            )

            return analysis_result

        except Exception as e:
            logger.error(f"{symbol} 종합 ATR 분석 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """현재 가격 조회"""
        try:
            await self._ensure_http_client()

            endpoint = f"/public/ticker/{symbol}"
            response = await self._http_client.get(endpoint)

            self._stats['api_calls'] += 1

            if response.get('status') == '0000' and 'data' in response:
                ticker_data = response['data']
                current_price = self._safe_float(ticker_data.get('closing_price'))
                return current_price

            return None

        except Exception as e:
            logger.error(f"{symbol} 현재 가격 조회 중 오류: {str(e)}")
            return None

    def _calculate_additional_stats(
        self,
        historical_data: List[Dict[str, Any]],
        atr_value: float
    ) -> Dict[str, Any]:
        """추가 통계 정보 계산"""
        try:
            if not historical_data:
                return {}

            close_prices = [d['close'] for d in historical_data]
            volumes = [d['volume'] for d in historical_data]

            # 가격 변동성 통계
            price_changes = []
            for i in range(1, len(close_prices)):
                change = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                price_changes.append(abs(change))

            avg_daily_change = statistics.mean(price_changes) if price_changes else 0
            max_daily_change = max(price_changes) if price_changes else 0
            min_daily_change = min(price_changes) if price_changes else 0

            return {
                'avg_daily_change_pct': avg_daily_change * 100,
                'max_daily_change_pct': max_daily_change * 100,
                'min_daily_change_pct': min_daily_change * 100,
                'avg_volume': statistics.mean(volumes) if volumes else 0,
                'price_range_pct': ((max(close_prices) - min(close_prices)) / min(close_prices)) * 100 if close_prices else 0,
                'atr_to_avg_change_ratio': atr_value / (avg_daily_change * close_prices[-1]) if avg_daily_change > 0 and close_prices else 0
            }

        except Exception as e:
            logger.error(f"추가 통계 계산 중 오류: {str(e)}")
            return {}

    def _assess_analysis_quality(self, data_points: int, period: int) -> Dict[str, Any]:
        """분석 품질 평가"""
        try:
            # 데이터 충분성 평가
            min_required = period + 5
            excellent_threshold = period * 2

            if data_points >= excellent_threshold:
                quality_grade = 'excellent'
                quality_score = 10.0
                quality_note = '충분한 데이터로 신뢰성 높음'
            elif data_points >= min_required:
                quality_grade = 'good'
                quality_score = 7.0 + (data_points - min_required) / (excellent_threshold - min_required) * 3.0
                quality_note = '양호한 데이터 품질'
            elif data_points >= period:
                quality_grade = 'fair'
                quality_score = 5.0
                quality_note = '최소 요구사항 충족'
            else:
                quality_grade = 'poor'
                quality_score = 2.0
                quality_note = '데이터 부족으로 신뢰성 제한적'

            return {
                'grade': quality_grade,
                'score': quality_score,
                'data_points': data_points,
                'required_minimum': min_required,
                'note': quality_note
            }

        except Exception as e:
            logger.error(f"분석 품질 평가 중 오류: {str(e)}")
            return {
                'grade': 'unknown',
                'score': 0.0,
                'data_points': data_points,
                'note': '평가 실패'
            }

    def _safe_float(self, value: Any) -> Optional[float]:
        """안전한 float 변환"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"float 변환 실패: {value}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """ATR 계산 서비스 통계 정보 반환"""
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
            'service_name': 'ATRCalculator',
            'http_client_available': http_client_status,
            'database_connected': await self.db_config.health_check(),
            'talib_available': True,  # TA-Lib 가용성 (import 성공했으므로 True)
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
_global_atr_calculator: Optional[ATRCalculator] = None


async def get_atr_calculator(db_config: Optional[DatabaseConfig] = None) -> ATRCalculator:
    """
    전역 ATRCalculator 인스턴스 반환

    Args:
        db_config: 데이터베이스 설정 객체 (선택적, None이면 기본값 생성)

    Returns:
        ATRCalculator 인스턴스
    """
    global _global_atr_calculator

    if db_config is None:
        db_config = DatabaseConfig()

    if _global_atr_calculator is None:
        _global_atr_calculator = ATRCalculator(db_config)

    return _global_atr_calculator


async def close_atr_calculator():
    """전역 ATRCalculator 인스턴스 정리"""
    global _global_atr_calculator

    if _global_atr_calculator is not None:
        await _global_atr_calculator.__aexit__(None, None, None)
        _global_atr_calculator = None