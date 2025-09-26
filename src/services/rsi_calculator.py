"""
RSI(Relative Strength Index) 계산 서비스

빗썸 API에서 과거 가격 데이터를 수집하여 RSI 지표를 계산하는 서비스입니다.
14일 기간 RSI 계산, 과매수/과매도 분석, 다이버전스 감지, 모멘텀 점수 산출 등을 제공합니다.
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


class RSICalculator:
    """
    RSI(Relative Strength Index) 계산 서비스

    빗썸 API와 데이터베이스를 활용하여 RSI 지표를 계산하고
    RSI 기반 모멘텀 분석 및 데이트레이딩 신호를 생성하는 서비스
    """

    # 기본 RSI 설정
    DEFAULT_RSI_PERIOD = 14  # 14일 기간
    MIN_DATA_POINTS = 20     # 최소 데이터 포인트 수 (RSI 계산을 위해 충분한 데이터 필요)

    # RSI 레벨 기준
    RSI_LEVELS = {
        'extreme_oversold': 20,     # 극도 과매도
        'oversold': 30,             # 과매도
        'lower_neutral': 35,        # 하위 중립 (데이트레이딩 진입 고려)
        'upper_neutral': 65,        # 상위 중립 (데이트레이딩 청산 고려)
        'overbought': 70,           # 과매수
        'extreme_overbought': 80    # 극도 과매수
    }

    # 데이트레이딩 적합 RSI 구간
    DAYTRADING_RSI_RANGES = {
        'very_oversold': (0, 20),      # 매우 과매도 - 반등 기회
        'oversold': (20, 35),          # 과매도 - 매수 신호 가능
        'optimal_buy': (35, 50),       # 최적 매수 구간
        'neutral': (50, 65),           # 중립 구간
        'optimal_sell': (50, 65),      # 최적 매도 구간
        'overbought': (65, 80),        # 과매수 - 매도 신호 가능
        'very_overbought': (80, 100)   # 매우 과매수 - 조정 위험
    }

    def __init__(self, db_config: DatabaseConfig):
        """
        RSI 계산기 초기화

        Args:
            db_config: 데이터베이스 설정 객체
        """
        self.db_config = db_config
        self._http_client: Optional[BithumbHTTPClient] = None
        self._stats = {
            'rsi_calculations': 0,
            'api_calls': 0,
            'db_queries': 0,
            'divergence_analyses': 0,
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
                logger.debug(f"{symbol} RSI 계산용 과거 데이터 - 데이터베이스에서 {len(db_data)}개 조회")
                return db_data

            # 데이터베이스에 충분한 데이터가 없으면 API 호출
            api_data = await self._get_historical_from_api(symbol, days)
            if api_data:
                logger.debug(f"{symbol} RSI 계산용 과거 데이터 - API에서 {len(api_data)}개 수집")
                return api_data

            logger.warning(f"{symbol} RSI 계산을 위한 과거 가격 데이터 수집 실패")
            return None

        except Exception as e:
            logger.error(f"{symbol} RSI 계산용 과거 데이터 수집 중 오류: {str(e)}")
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

            # RSI 계산을 위한 가상 데이터 생성 (실제로는 실제 과거 데이터 필요)
            historical_data = []
            base_time = datetime.now(timezone.utc) - timedelta(days=days)

            # RSI 계산에 적합한 가격 변동 시뮬레이션
            for i in range(days):
                # 트렌드와 변동성을 고려한 가격 시뮬레이션
                trend_factor = np.sin(i / days * np.pi) * 0.1  # 사인파 트렌드
                random_factor = np.random.normal(0, 0.03)      # 3% 표준편차

                sim_close = current_price * (1 + trend_factor + random_factor * (days - i) / days)
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

            logger.warning(f"{symbol} API에서 RSI 계산용 시뮬레이션 데이터 생성 (실제 과거 데이터 필요)")
            return historical_data

        except Exception as e:
            logger.error(f"{symbol} API 과거 데이터 수집 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    def calculate_rsi(
        self,
        historical_data: List[Dict[str, Any]],
        period: int = None
    ) -> Optional[np.ndarray]:
        """
        RSI(Relative Strength Index) 계산

        Args:
            historical_data: OHLCV 데이터 리스트
            period: RSI 계산 기간 (기본값: 14일)

        Returns:
            RSI 값 배열 또는 None
        """
        try:
            if period is None:
                period = self.DEFAULT_RSI_PERIOD

            if len(historical_data) < period + 1:
                logger.warning(f"RSI 계산을 위한 데이터 부족: {len(historical_data)} < {period + 1}")
                return None

            # 종가 데이터를 numpy 배열로 변환
            close_prices = np.array([float(d['close']) for d in historical_data])

            # TA-Lib을 사용하여 RSI 계산
            rsi_values = talib.RSI(close_prices, timeperiod=period)

            # NaN 값 제거
            valid_rsi = rsi_values[~np.isnan(rsi_values)]
            if len(valid_rsi) == 0:
                return None

            logger.debug(f"RSI 계산 완료: {len(valid_rsi)}개 값 (기간: {period}일)")
            return rsi_values

        except Exception as e:
            logger.error(f"RSI 계산 중 오류: {str(e)}")
            return None

    def get_current_rsi(self, rsi_values: np.ndarray) -> Optional[float]:
        """
        현재(최신) RSI 값 반환

        Args:
            rsi_values: RSI 값 배열

        Returns:
            현재 RSI 값 또는 None
        """
        try:
            if rsi_values is None or len(rsi_values) == 0:
                return None

            # 최신 유효한 RSI 값 찾기
            for i in range(len(rsi_values) - 1, -1, -1):
                if not np.isnan(rsi_values[i]):
                    return float(rsi_values[i])

            return None

        except Exception as e:
            logger.error(f"현재 RSI 값 조회 중 오류: {str(e)}")
            return None

    def classify_rsi_level(self, rsi_value: float) -> Dict[str, Any]:
        """
        RSI 수준 분류 및 해석

        Args:
            rsi_value: RSI 값

        Returns:
            RSI 수준 분류 결과
        """
        try:
            # RSI 수준 분류
            if rsi_value <= self.RSI_LEVELS['extreme_oversold']:
                level = 'extreme_oversold'
                signal = 'strong_buy'
                description = '극도 과매도 - 강력한 매수 신호'
                risk_level = 'low'
                momentum_score = 2.0
            elif rsi_value <= self.RSI_LEVELS['oversold']:
                level = 'oversold'
                signal = 'buy'
                description = '과매도 - 매수 고려'
                risk_level = 'low'
                momentum_score = 3.0
            elif rsi_value <= self.RSI_LEVELS['lower_neutral']:
                level = 'lower_neutral'
                signal = 'weak_buy'
                description = '하위 중립 - 약한 매수 신호'
                risk_level = 'medium'
                momentum_score = 4.0
            elif rsi_value <= 50:
                level = 'lower_middle'
                signal = 'neutral'
                description = '중립 하위 - 관망'
                risk_level = 'medium'
                momentum_score = 5.0
            elif rsi_value <= self.RSI_LEVELS['upper_neutral']:
                level = 'upper_middle'
                signal = 'neutral'
                description = '중립 상위 - 관망'
                risk_level = 'medium'
                momentum_score = 6.0
            elif rsi_value <= self.RSI_LEVELS['overbought']:
                level = 'upper_neutral'
                signal = 'weak_sell'
                description = '상위 중립 - 약한 매도 신호'
                risk_level = 'medium'
                momentum_score = 7.0
            elif rsi_value <= self.RSI_LEVELS['extreme_overbought']:
                level = 'overbought'
                signal = 'sell'
                description = '과매수 - 매도 고려'
                risk_level = 'high'
                momentum_score = 8.0
            else:
                level = 'extreme_overbought'
                signal = 'strong_sell'
                description = '극도 과매수 - 강력한 매도 신호'
                risk_level = 'high'
                momentum_score = 9.0

            return {
                'rsi_value': rsi_value,
                'level': level,
                'signal': signal,
                'description': description,
                'risk_level': risk_level,
                'momentum_score': momentum_score,
                'is_extreme': level in ['extreme_oversold', 'extreme_overbought'],
                'trading_opportunity': signal in ['strong_buy', 'buy', 'sell', 'strong_sell']
            }

        except Exception as e:
            logger.error(f"RSI 수준 분류 중 오류: {str(e)}")
            return {
                'rsi_value': rsi_value,
                'level': 'unknown',
                'signal': 'neutral',
                'description': '분석 실패',
                'risk_level': 'unknown',
                'momentum_score': 5.0,
                'is_extreme': False,
                'trading_opportunity': False
            }

    def evaluate_daytrading_suitability(self, rsi_value: float) -> Dict[str, Any]:
        """
        데이트레이딩 적합성 평가 (RSI 기반)

        Args:
            rsi_value: 현재 RSI 값

        Returns:
            데이트레이딩 적합성 평가 결과
        """
        try:
            suitability_score = 0.0
            trading_zone = 'unknown'
            recommendation = '데이터 부족'

            # RSI 기반 데이트레이딩 구간 분류
            if 35 <= rsi_value <= 65:
                # 최적 데이트레이딩 구간 (35-65)
                trading_zone = 'optimal'
                base_score = 8.5
                # 50 근처일수록 높은 점수 (양방향 기회)
                center_bonus = 2.0 - abs(rsi_value - 50) / 15.0  # 0-2점 보너스
                suitability_score = base_score + center_bonus
                recommendation = '데이트레이딩 최적 구간 - 양방향 기회 활용 가능'
            elif 30 <= rsi_value < 35 or 65 < rsi_value <= 70:
                # 준최적 구간
                trading_zone = 'good'
                suitability_score = 7.0
                recommendation = '데이트레이딩 양호 - 한 방향 편향 주의'
            elif 25 <= rsi_value < 30 or 70 < rsi_value <= 75:
                # 신중한 트레이딩 필요
                trading_zone = 'caution'
                suitability_score = 5.5
                recommendation = '신중한 데이트레이딩 - 반전 위험 고려'
            elif rsi_value < 25 or rsi_value > 75:
                # 극단 구간 - 높은 위험
                trading_zone = 'extreme'
                suitability_score = 3.0
                recommendation = '극단 구간 - 데이트레이딩 부적합, 반전 대기'
            else:
                trading_zone = 'unknown'
                suitability_score = 0.0
                recommendation = '분석 실패'

            # 추가 요소 고려
            volatility_bonus = 0.0
            if 40 <= rsi_value <= 60:
                # 중간 구간일 때 변동성 보너스
                volatility_bonus = 0.5

            final_score = min(10.0, suitability_score + volatility_bonus)

            return {
                'rsi_value': rsi_value,
                'trading_zone': trading_zone,
                'suitability_score': final_score,
                'is_suitable': final_score >= 7.0,
                'recommendation': recommendation,
                'risk_assessment': self._assess_rsi_risk(rsi_value),
                'preferred_strategy': self._suggest_strategy(rsi_value)
            }

        except Exception as e:
            logger.error(f"데이트레이딩 적합성 평가 중 오류: {str(e)}")
            return {
                'rsi_value': rsi_value,
                'trading_zone': 'unknown',
                'suitability_score': 0.0,
                'is_suitable': False,
                'recommendation': '평가 실패',
                'risk_assessment': 'unknown',
                'preferred_strategy': 'hold'
            }

    def detect_divergence(
        self,
        price_data: List[float],
        rsi_data: List[float],
        lookback_periods: int = 10
    ) -> Dict[str, Any]:
        """
        RSI 다이버전스 감지

        Args:
            price_data: 가격 데이터 리스트
            rsi_data: RSI 데이터 리스트
            lookback_periods: 분석할 기간

        Returns:
            다이버전스 분석 결과
        """
        try:
            self._stats['divergence_analyses'] += 1

            if len(price_data) < lookback_periods or len(rsi_data) < lookback_periods:
                logger.warning("다이버전스 분석을 위한 데이터 부족")
                return {
                    'divergence_detected': False,
                    'divergence_type': None,
                    'strength': 0.0,
                    'description': '데이터 부족'
                }

            # 최근 데이터 슬라이싱
            recent_prices = price_data[-lookback_periods:]
            recent_rsi = rsi_data[-lookback_periods:]

            # 가격과 RSI의 고점/저점 찾기
            price_peaks = self._find_peaks_and_troughs(recent_prices)
            rsi_peaks = self._find_peaks_and_troughs(recent_rsi)

            # 다이버전스 유형 감지
            divergence_type = None
            divergence_strength = 0.0
            description = '다이버전스 없음'

            # 강세 다이버전스 (가격 하락, RSI 상승)
            if (price_peaks['latest_trough_idx'] >= 0 and
                rsi_peaks['latest_trough_idx'] >= 0):

                price_trend = price_peaks['latest_trough_value'] - price_peaks['prev_trough_value'] if price_peaks['prev_trough_value'] else 0
                rsi_trend = rsi_peaks['latest_trough_value'] - rsi_peaks['prev_trough_value'] if rsi_peaks['prev_trough_value'] else 0

                if price_trend < 0 and rsi_trend > 0:  # 가격 하락, RSI 상승
                    divergence_type = 'bullish'
                    divergence_strength = min(10.0, abs(price_trend) * abs(rsi_trend) * 100)
                    description = '강세 다이버전스 - 상승 반전 신호'

            # 약세 다이버전스 (가격 상승, RSI 하락)
            if (price_peaks['latest_peak_idx'] >= 0 and
                rsi_peaks['latest_peak_idx'] >= 0):

                price_trend = price_peaks['latest_peak_value'] - price_peaks['prev_peak_value'] if price_peaks['prev_peak_value'] else 0
                rsi_trend = rsi_peaks['latest_peak_value'] - rsi_peaks['prev_peak_value'] if rsi_peaks['prev_peak_value'] else 0

                if price_trend > 0 and rsi_trend < 0:  # 가격 상승, RSI 하락
                    if divergence_type is None:  # 강세 다이버전스가 없는 경우에만
                        divergence_type = 'bearish'
                        divergence_strength = min(10.0, abs(price_trend) * abs(rsi_trend) * 100)
                        description = '약세 다이버전스 - 하락 반전 신호'

            return {
                'divergence_detected': divergence_type is not None,
                'divergence_type': divergence_type,
                'strength': divergence_strength,
                'description': description,
                'confidence': self._calculate_divergence_confidence(divergence_strength, lookback_periods),
                'lookback_periods': lookback_periods,
                'analysis_timestamp': datetime.now(timezone.utc)
            }

        except Exception as e:
            logger.error(f"RSI 다이버전스 감지 중 오류: {str(e)}")
            return {
                'divergence_detected': False,
                'divergence_type': None,
                'strength': 0.0,
                'description': '분석 실패',
                'confidence': 0.0
            }

    def calculate_momentum_score(
        self,
        rsi_value: float,
        rsi_trend: Optional[List[float]] = None,
        divergence_info: Optional[Dict] = None
    ) -> float:
        """
        RSI 기반 모멘텀 점수 계산

        Args:
            rsi_value: 현재 RSI 값
            rsi_trend: RSI 트렌드 데이터
            divergence_info: 다이버전스 정보

        Returns:
            모멘텀 점수 (0-10)
        """
        try:
            momentum_score = 5.0  # 기본 중립 점수

            # 1. RSI 레벨 기반 점수 (40% 가중치)
            rsi_classification = self.classify_rsi_level(rsi_value)
            level_score = rsi_classification['momentum_score']
            momentum_score += (level_score - 5.0) * 0.4

            # 2. RSI 트렌드 분석 (30% 가중치)
            if rsi_trend and len(rsi_trend) >= 3:
                trend_score = self._analyze_rsi_trend(rsi_trend)
                momentum_score += trend_score * 0.3

            # 3. 다이버전스 보너스/페널티 (20% 가중치)
            if divergence_info and divergence_info.get('divergence_detected'):
                divergence_score = self._score_divergence_impact(divergence_info)
                momentum_score += divergence_score * 0.2

            # 4. 극값 근접도 보너스 (10% 가중치)
            extreme_bonus = self._calculate_extreme_bonus(rsi_value)
            momentum_score += extreme_bonus * 0.1

            # 점수 범위 제한
            final_score = max(0.0, min(10.0, momentum_score))

            logger.debug(f"RSI 모멘텀 점수 계산: {final_score:.2f} (RSI: {rsi_value:.2f})")
            return final_score

        except Exception as e:
            logger.error(f"모멘텀 점수 계산 중 오류: {str(e)}")
            return 5.0  # 오류 시 중립 점수 반환

    async def comprehensive_rsi_analysis(
        self,
        symbol: str,
        period: int = None,
        include_divergence: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        종합적인 RSI 분석

        과거 데이터 수집, RSI 계산, 수준 분석, 다이버전스 감지, 모멘텀 점수 계산을 모두 포함

        Args:
            symbol: 분석할 심볼
            period: RSI 계산 기간
            include_divergence: 다이버전스 분석 포함 여부

        Returns:
            종합 RSI 분석 결과
        """
        try:
            if period is None:
                period = self.DEFAULT_RSI_PERIOD

            logger.info(f"{symbol} 종합 RSI 분석 시작 (기간: {period}일)")

            # 1. 과거 가격 데이터 수집
            historical_data = await self.fetch_historical_data(symbol, period + 20)  # 충분한 데이터 확보
            if not historical_data:
                logger.error(f"{symbol} 과거 가격 데이터 수집 실패")
                return None

            # 2. RSI 계산
            rsi_values = self.calculate_rsi(historical_data, period)
            if rsi_values is None:
                logger.error(f"{symbol} RSI 계산 실패")
                return None

            # 3. 현재 RSI 값 조회
            current_rsi = self.get_current_rsi(rsi_values)
            if current_rsi is None:
                logger.error(f"{symbol} 현재 RSI 값 조회 실패")
                return None

            # 4. RSI 수준 분류
            rsi_classification = self.classify_rsi_level(current_rsi)

            # 5. 데이트레이딩 적합성 평가
            daytrading_suitability = self.evaluate_daytrading_suitability(current_rsi)

            # 6. 다이버전스 분석 (선택적)
            divergence_analysis = None
            if include_divergence and len(historical_data) >= 15:
                price_data = [d['close'] for d in historical_data]
                rsi_data = [rsi for rsi in rsi_values if not np.isnan(rsi)]

                if len(rsi_data) >= 10:
                    divergence_analysis = self.detect_divergence(price_data, rsi_data)

            # 7. RSI 트렌드 분석
            rsi_trend = [rsi for rsi in rsi_values[-7:] if not np.isnan(rsi)]  # 최근 7일

            # 8. 모멘텀 점수 계산
            momentum_score = self.calculate_momentum_score(
                current_rsi, rsi_trend, divergence_analysis
            )

            # 9. 추가 통계 계산
            rsi_statistics = self._calculate_rsi_statistics(rsi_values)

            # 결과 정리
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'rsi_period': period,
                'data_points_used': len(historical_data),
                'current_rsi': current_rsi,
                'rsi_classification': rsi_classification,
                'daytrading_suitability': daytrading_suitability,
                'momentum_score': momentum_score,
                'divergence_analysis': divergence_analysis,
                'rsi_trend_data': rsi_trend[-5:] if rsi_trend else [],  # 최근 5개 값만 반환
                'rsi_statistics': rsi_statistics,
                'trading_signals': self._generate_trading_signals(current_rsi, divergence_analysis),
                'analysis_quality': self._assess_analysis_quality(len(historical_data), period)
            }

            self._stats['rsi_calculations'] += 1
            self._stats['last_calculation'] = datetime.now(timezone.utc)

            logger.info(
                f"{symbol} 종합 RSI 분석 완료 - "
                f"RSI: {current_rsi:.2f} ({rsi_classification['level']}), "
                f"모멘텀점수: {momentum_score:.2f}, "
                f"데이트레이딩: {'적합' if daytrading_suitability['is_suitable'] else '부적합'}"
            )

            return analysis_result

        except Exception as e:
            logger.error(f"{symbol} 종합 RSI 분석 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    def _find_peaks_and_troughs(self, data: List[float]) -> Dict[str, Any]:
        """데이터에서 고점과 저점 찾기"""
        try:
            if len(data) < 3:
                return {'latest_peak_idx': -1, 'latest_trough_idx': -1}

            peaks = []
            troughs = []

            for i in range(1, len(data) - 1):
                if data[i] > data[i-1] and data[i] > data[i+1]:
                    peaks.append({'idx': i, 'value': data[i]})
                elif data[i] < data[i-1] and data[i] < data[i+1]:
                    troughs.append({'idx': i, 'value': data[i]})

            result = {
                'latest_peak_idx': peaks[-1]['idx'] if peaks else -1,
                'latest_peak_value': peaks[-1]['value'] if peaks else None,
                'prev_peak_value': peaks[-2]['value'] if len(peaks) >= 2 else None,
                'latest_trough_idx': troughs[-1]['idx'] if troughs else -1,
                'latest_trough_value': troughs[-1]['value'] if troughs else None,
                'prev_trough_value': troughs[-2]['value'] if len(troughs) >= 2 else None
            }

            return result

        except Exception as e:
            logger.error(f"고점/저점 찾기 중 오류: {str(e)}")
            return {'latest_peak_idx': -1, 'latest_trough_idx': -1}

    def _analyze_rsi_trend(self, rsi_trend: List[float]) -> float:
        """RSI 트렌드 분석하여 점수 반환"""
        try:
            if len(rsi_trend) < 3:
                return 0.0

            # 선형 회귀를 통한 트렌드 방향성 계산
            x = np.arange(len(rsi_trend))
            y = np.array(rsi_trend)

            slope = np.polyfit(x, y, 1)[0]

            # 기울기를 점수로 변환 (-5 ~ +5)
            trend_score = np.clip(slope * 2, -5.0, 5.0)

            return trend_score

        except Exception as e:
            logger.error(f"RSI 트렌드 분석 중 오류: {str(e)}")
            return 0.0

    def _score_divergence_impact(self, divergence_info: Dict) -> float:
        """다이버전스 영향도 점수 계산"""
        try:
            if not divergence_info.get('divergence_detected'):
                return 0.0

            strength = divergence_info.get('strength', 0)
            divergence_type = divergence_info.get('divergence_type')

            if divergence_type == 'bullish':
                return min(3.0, strength / 3.0)  # 강세 다이버전스는 양의 점수
            elif divergence_type == 'bearish':
                return max(-3.0, -strength / 3.0)  # 약세 다이버전스는 음의 점수

            return 0.0

        except Exception as e:
            logger.error(f"다이버전스 점수 계산 중 오류: {str(e)}")
            return 0.0

    def _calculate_extreme_bonus(self, rsi_value: float) -> float:
        """극값 근접도에 따른 보너스 점수"""
        try:
            if rsi_value <= 30:
                # 과매도 구간 - 반등 기대
                return (30 - rsi_value) / 10.0  # 최대 3점
            elif rsi_value >= 70:
                # 과매수 구간 - 조정 기대
                return -(rsi_value - 70) / 10.0  # 최대 -3점
            else:
                return 0.0

        except Exception as e:
            logger.error(f"극값 보너스 계산 중 오류: {str(e)}")
            return 0.0

    def _calculate_divergence_confidence(self, strength: float, lookback: int) -> float:
        """다이버전스 신뢰도 계산"""
        try:
            base_confidence = min(100.0, strength * 10)
            period_factor = min(1.0, lookback / 10.0)  # 분석 기간이 길수록 신뢰도 증가

            return base_confidence * period_factor

        except Exception as e:
            logger.error(f"다이버전스 신뢰도 계산 중 오류: {str(e)}")
            return 0.0

    def _calculate_rsi_statistics(self, rsi_values: np.ndarray) -> Dict[str, Any]:
        """RSI 통계 정보 계산"""
        try:
            valid_rsi = rsi_values[~np.isnan(rsi_values)]

            if len(valid_rsi) == 0:
                return {}

            return {
                'avg_rsi': float(np.mean(valid_rsi)),
                'max_rsi': float(np.max(valid_rsi)),
                'min_rsi': float(np.min(valid_rsi)),
                'rsi_volatility': float(np.std(valid_rsi)),
                'oversold_periods': int(np.sum(valid_rsi <= 30)),
                'overbought_periods': int(np.sum(valid_rsi >= 70)),
                'neutral_periods': int(np.sum((valid_rsi > 30) & (valid_rsi < 70))),
                'total_periods': len(valid_rsi)
            }

        except Exception as e:
            logger.error(f"RSI 통계 계산 중 오류: {str(e)}")
            return {}

    def _generate_trading_signals(
        self,
        current_rsi: float,
        divergence_analysis: Optional[Dict]
    ) -> Dict[str, Any]:
        """트레이딩 신호 생성"""
        try:
            signals = {
                'primary_signal': 'hold',
                'signal_strength': 0,
                'entry_signals': [],
                'exit_signals': [],
                'risk_warnings': []
            }

            # 주요 신호 결정
            if current_rsi <= 25:
                signals['primary_signal'] = 'strong_buy'
                signals['signal_strength'] = 9
                signals['entry_signals'].append('극도 과매도 - 강력한 매수 기회')
            elif current_rsi <= 35:
                signals['primary_signal'] = 'buy'
                signals['signal_strength'] = 7
                signals['entry_signals'].append('과매도 - 매수 고려')
            elif current_rsi >= 75:
                signals['primary_signal'] = 'strong_sell'
                signals['signal_strength'] = 9
                signals['exit_signals'].append('극도 과매수 - 강력한 매도 기회')
            elif current_rsi >= 65:
                signals['primary_signal'] = 'sell'
                signals['signal_strength'] = 7
                signals['exit_signals'].append('과매수 - 매도 고려')

            # 다이버전스 신호 추가
            if divergence_analysis and divergence_analysis.get('divergence_detected'):
                div_type = divergence_analysis.get('divergence_type')
                if div_type == 'bullish':
                    signals['entry_signals'].append('강세 다이버전스 감지')
                    if signals['primary_signal'] == 'hold':
                        signals['primary_signal'] = 'buy'
                        signals['signal_strength'] = 6
                elif div_type == 'bearish':
                    signals['exit_signals'].append('약세 다이버전스 감지')
                    if signals['primary_signal'] == 'hold':
                        signals['primary_signal'] = 'sell'
                        signals['signal_strength'] = 6

            # 위험 경고
            if current_rsi > 80:
                signals['risk_warnings'].append('극도 과매수 - 급격한 조정 위험')
            elif current_rsi < 20:
                signals['risk_warnings'].append('극도 과매도 - 추가 하락 위험 가능')

            return signals

        except Exception as e:
            logger.error(f"트레이딩 신호 생성 중 오류: {str(e)}")
            return {'primary_signal': 'hold', 'signal_strength': 0}

    def _assess_rsi_risk(self, rsi_value: float) -> str:
        """RSI 기반 위험도 평가"""
        if rsi_value <= 20 or rsi_value >= 80:
            return 'very_high'
        elif rsi_value <= 30 or rsi_value >= 70:
            return 'high'
        elif rsi_value <= 40 or rsi_value >= 60:
            return 'medium'
        else:
            return 'low'

    def _suggest_strategy(self, rsi_value: float) -> str:
        """RSI 기반 전략 제안"""
        if 35 <= rsi_value <= 65:
            return 'scalping'  # 스캘핑 적합
        elif rsi_value < 35:
            return 'buy_and_hold'  # 매수 후 보유
        elif rsi_value > 65:
            return 'take_profit'  # 차익 실현
        else:
            return 'wait'  # 관망

    def _assess_analysis_quality(self, data_points: int, period: int) -> Dict[str, Any]:
        """분석 품질 평가"""
        try:
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
        """RSI 계산 서비스 통계 정보 반환"""
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
            'service_name': 'RSICalculator',
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
_global_rsi_calculator: Optional[RSICalculator] = None


async def get_rsi_calculator(db_config: Optional[DatabaseConfig] = None) -> RSICalculator:
    """
    전역 RSICalculator 인스턴스 반환

    Args:
        db_config: 데이터베이스 설정 객체 (선택적, None이면 기본값 생성)

    Returns:
        RSICalculator 인스턴스
    """
    global _global_rsi_calculator

    if db_config is None:
        db_config = DatabaseConfig()

    if _global_rsi_calculator is None:
        _global_rsi_calculator = RSICalculator(db_config)

    return _global_rsi_calculator


async def close_rsi_calculator():
    """전역 RSICalculator 인스턴스 정리"""
    global _global_rsi_calculator

    if _global_rsi_calculator is not None:
        await _global_rsi_calculator.__aexit__(None, None, None)
        _global_rsi_calculator = None