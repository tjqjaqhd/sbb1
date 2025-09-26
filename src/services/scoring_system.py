"""
가중 점수 시스템

각 지표별 가중치를 적용하여 종합 점수를 계산하는 시스템입니다.
거래량, ATR, RSI, 볼린저밴드, 스프레드 분석 결과를 통합하여
데이트레이딩 적합성을 평가합니다.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal, InvalidOperation

from src.database.config import DatabaseConfig
from src.services.volume_analyzer import VolumeAnalyzer, get_volume_analyzer
from src.services.atr_calculator import ATRCalculator, get_atr_calculator
from src.services.rsi_calculator import RSICalculator, get_rsi_calculator
from src.services.bollinger_analyzer import BollingerAnalyzer, get_bollinger_analyzer
from src.services.spread_analyzer import SpreadAnalyzer, get_spread_analyzer

logger = logging.getLogger(__name__)


class ScoringSystem:
    """
    가중 점수 시스템

    여러 기술적 지표를 통합하여 종합적인 투자 점수를 계산하는 시스템
    각 지표별 가중치를 적용하고 이상치를 필터링하여 신뢰도 높은 점수 제공
    """

    # 기본 가중치 설정 (총 100%)
    DEFAULT_WEIGHTS = {
        'volume': 0.30,      # 거래량 30%
        'atr': 0.25,         # ATR (변동성) 25%
        'rsi': 0.20,         # RSI (모멘텀) 20%
        'bollinger': 0.15,   # 볼린저밴드 15%
        'spread': 0.10       # 스프레드 (유동성) 10%
    }

    # 동적 가중치 조정 규칙
    DYNAMIC_WEIGHT_RULES = {
        'high_volatility': {  # 고변동성 시장
            'condition': 'atr_percentage > 0.08',  # ATR 8% 초과
            'weights': {
                'volume': 0.25,     # 거래량 비중 감소
                'atr': 0.35,        # ATR 비중 증가
                'rsi': 0.15,        # RSI 비중 감소
                'bollinger': 0.20,  # 볼린저밴드 비중 증가
                'spread': 0.05      # 스프레드 비중 감소
            }
        },
        'low_volatility': {  # 저변동성 시장
            'condition': 'atr_percentage < 0.03',  # ATR 3% 미만
            'weights': {
                'volume': 0.40,     # 거래량 비중 증가
                'atr': 0.10,        # ATR 비중 감소
                'rsi': 0.30,        # RSI 비중 증가
                'bollinger': 0.10,  # 볼린저밴드 비중 감소
                'spread': 0.10      # 스프레드 비중 유지
            }
        },
        'low_liquidity': {  # 유동성 부족
            'condition': 'spread_rate > 0.005',  # 스프레드 0.5% 초과
            'weights': {
                'volume': 0.25,     # 거래량 비중 감소
                'atr': 0.20,        # ATR 비중 감소
                'rsi': 0.15,        # RSI 비중 감소
                'bollinger': 0.15,  # 볼린저밴드 비중 유지
                'spread': 0.25      # 스프레드 비중 크게 증가
            }
        }
    }

    # 이상치 필터링 임계값
    OUTLIER_Z_SCORE_THRESHOLD = 2.5  # Z-score 2.5 이상을 이상치로 판정

    # 최소 품질 기준
    MIN_QUALITY_THRESHOLD = 0.6  # 60% 이상 신뢰도 필요

    def __init__(self, db_config: DatabaseConfig):
        """
        점수 시스템 초기화

        Args:
            db_config: 데이터베이스 설정 객체
        """
        self.db_config = db_config

        # 분석 서비스들 초기화
        self._volume_analyzer: Optional[VolumeAnalyzer] = None
        self._atr_calculator: Optional[ATRCalculator] = None
        self._rsi_calculator: Optional[RSICalculator] = None
        self._bollinger_analyzer: Optional[BollingerAnalyzer] = None
        self._spread_analyzer: Optional[SpreadAnalyzer] = None

        # 통계 정보
        self._stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'filtered_outliers': 0,
            'dynamic_weight_adjustments': 0,
            'last_calculation': None,
            'errors': 0
        }

        # 점수 이력 (이상치 필터링용)
        self._score_history: Dict[str, List[float]] = {}

    async def _ensure_analyzers(self):
        """모든 분석 서비스 초기화 확인"""
        try:
            if self._volume_analyzer is None:
                self._volume_analyzer = await get_volume_analyzer(self.db_config)

            if self._atr_calculator is None:
                self._atr_calculator = await get_atr_calculator(self.db_config)

            if self._rsi_calculator is None:
                self._rsi_calculator = await get_rsi_calculator(self.db_config)

            if self._bollinger_analyzer is None:
                self._bollinger_analyzer = await get_bollinger_analyzer(self.db_config)

            if self._spread_analyzer is None:
                self._spread_analyzer = await get_spread_analyzer(self.db_config)

        except Exception as e:
            logger.error(f"분석 서비스 초기화 중 오류: {str(e)}")
            raise

    async def calculate_comprehensive_score(
        self,
        symbol: str,
        custom_weights: Optional[Dict[str, float]] = None,
        enable_dynamic_weights: bool = True,
        filter_outliers: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        종합 점수 계산

        모든 분석 서비스를 활용하여 가중 평균 종합 점수를 계산

        Args:
            symbol: 분석할 심볼
            custom_weights: 사용자 정의 가중치
            enable_dynamic_weights: 동적 가중치 조정 활성화 여부
            filter_outliers: 이상치 필터링 활성화 여부

        Returns:
            종합 점수 분석 결과
        """
        try:
            logger.info(f"{symbol} 종합 점수 계산 시작")
            self._stats['total_calculations'] += 1

            # 분석 서비스 초기화
            await self._ensure_analyzers()

            # 각 분석 서비스에서 데이터 수집
            analysis_results = await self._collect_all_analysis_data(symbol)

            if not analysis_results:
                logger.error(f"{symbol} 분석 데이터 수집 실패")
                return None

            # 정규화된 점수 추출
            normalized_scores = self._extract_normalized_scores(analysis_results)

            if not normalized_scores:
                logger.error(f"{symbol} 정규화된 점수 추출 실패")
                return None

            # 가중치 결정 (동적 조정 포함)
            weights = await self._determine_weights(
                analysis_results,
                custom_weights,
                enable_dynamic_weights
            )

            # 이상치 필터링
            if filter_outliers:
                normalized_scores = self._filter_outliers(symbol, normalized_scores)

            # 종합 점수 계산
            comprehensive_score = self._calculate_weighted_score(normalized_scores, weights)

            # 품질 평가
            quality_assessment = self._assess_score_quality(normalized_scores, analysis_results)

            # 결과 정리
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'comprehensive_score': comprehensive_score,
                'individual_scores': normalized_scores,
                'weights_used': weights,
                'quality_assessment': quality_assessment,
                'analysis_results': analysis_results,
                'is_reliable': quality_assessment['overall_quality'] >= self.MIN_QUALITY_THRESHOLD,
                'recommendation': self._generate_recommendation(comprehensive_score, quality_assessment)
            }

            # 통계 업데이트
            self._stats['successful_calculations'] += 1
            self._stats['last_calculation'] = datetime.now(timezone.utc)

            # 점수 이력 업데이트 (이상치 필터링용)
            self._update_score_history(symbol, comprehensive_score)

            logger.info(
                f"{symbol} 종합 점수 계산 완료 - "
                f"점수: {comprehensive_score:.2f}, "
                f"품질: {quality_assessment['overall_quality']:.2f}, "
                f"신뢰도: {'높음' if result['is_reliable'] else '낮음'}"
            )

            return result

        except Exception as e:
            logger.error(f"{symbol} 종합 점수 계산 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def _collect_all_analysis_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        모든 분석 서비스에서 데이터 수집

        Args:
            symbol: 분석할 심볼

        Returns:
            모든 분석 결과 딕셔너리
        """
        try:
            # 병렬로 모든 분석 실행
            tasks = [
                self._volume_analyzer.comprehensive_volume_analysis(symbol),
                self._atr_calculator.comprehensive_atr_analysis(symbol),
                self._rsi_calculator.comprehensive_rsi_analysis(symbol),
                self._bollinger_analyzer.comprehensive_bollinger_analysis(symbol),
                self._spread_analyzer.comprehensive_spread_analysis(symbol)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            analysis_data = {
                'volume': results[0] if not isinstance(results[0], Exception) else None,
                'atr': results[1] if not isinstance(results[1], Exception) else None,
                'rsi': results[2] if not isinstance(results[2], Exception) else None,
                'bollinger': results[3] if not isinstance(results[3], Exception) else None,
                'spread': results[4] if not isinstance(results[4], Exception) else None
            }

            # 오류 로깅
            for i, (name, result) in enumerate(zip(['volume', 'atr', 'rsi', 'bollinger', 'spread'], results)):
                if isinstance(result, Exception):
                    logger.warning(f"{symbol} {name} 분석 실패: {str(result)}")

            return analysis_data

        except Exception as e:
            logger.error(f"{symbol} 전체 분석 데이터 수집 중 오류: {str(e)}")
            return None

    def _extract_normalized_scores(self, analysis_results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        분석 결과에서 정규화된 점수 추출 (0-100 스케일)

        Args:
            analysis_results: 각 분석 서비스 결과

        Returns:
            정규화된 점수 딕셔너리
        """
        try:
            normalized_scores = {}

            # 거래량 점수 추출
            if analysis_results.get('volume'):
                volume_data = analysis_results['volume']
                # comprehensive_score는 이미 0-100 범위
                normalized_scores['volume'] = float(volume_data.get('comprehensive_score', 0.0))

            # ATR 점수 추출
            if analysis_results.get('atr'):
                atr_data = analysis_results['atr']
                # volatility_score를 0-100 범위로 변환 (원래 0-10 범위)
                volatility_score = float(atr_data.get('volatility_score', 0.0))
                normalized_scores['atr'] = min(100.0, volatility_score * 10.0)

            # RSI 점수 추출
            if analysis_results.get('rsi'):
                rsi_data = analysis_results['rsi']
                # momentum_score를 그대로 사용 (이미 0-100 범위로 설계됨)
                normalized_scores['rsi'] = float(rsi_data.get('momentum_score', 0.0))

            # 볼린저밴드 점수 추출
            if analysis_results.get('bollinger'):
                bollinger_data = analysis_results['bollinger']
                # breakout_score를 0-100 범위로 변환 (원래 0-10 범위)
                breakout_score = float(bollinger_data.get('breakout_score', 0.0))
                normalized_scores['bollinger'] = min(100.0, breakout_score * 10.0)

            # 스프레드 점수 추출
            if analysis_results.get('spread'):
                spread_data = analysis_results['spread']
                # comprehensive_score를 그대로 사용 (이미 0-100 범위)
                normalized_scores['spread'] = float(spread_data.get('comprehensive_score', 0.0))

            # 최소 2개 이상의 점수가 있어야 신뢰할 수 있음
            if len(normalized_scores) < 2:
                logger.warning("정규화된 점수가 부족합니다 (최소 2개 필요)")
                return None

            return normalized_scores

        except Exception as e:
            logger.error(f"정규화된 점수 추출 중 오류: {str(e)}")
            return None

    async def _determine_weights(
        self,
        analysis_results: Dict[str, Any],
        custom_weights: Optional[Dict[str, float]] = None,
        enable_dynamic: bool = True
    ) -> Dict[str, float]:
        """
        가중치 결정 (동적 조정 포함)

        Args:
            analysis_results: 분석 결과
            custom_weights: 사용자 정의 가중치
            enable_dynamic: 동적 조정 활성화

        Returns:
            최종 가중치 딕셔너리
        """
        try:
            # 기본 가중치에서 시작
            if custom_weights:
                weights = custom_weights.copy()
            else:
                weights = self.DEFAULT_WEIGHTS.copy()

            # 동적 가중치 조정
            if enable_dynamic:
                adjusted_weights = self._apply_dynamic_weight_adjustment(weights, analysis_results)
                if adjusted_weights:
                    weights = adjusted_weights
                    self._stats['dynamic_weight_adjustments'] += 1

            # 사용 가능한 지표에 따라 가중치 정규화
            available_indicators = set()
            for indicator in ['volume', 'atr', 'rsi', 'bollinger', 'spread']:
                if analysis_results.get(indicator):
                    available_indicators.add(indicator)

            # 사용 가능한 지표의 가중치만 추출하고 정규화
            filtered_weights = {k: v for k, v in weights.items() if k in available_indicators}

            if filtered_weights:
                total_weight = sum(filtered_weights.values())
                if total_weight > 0:
                    # 가중치 정규화 (합을 1.0으로)
                    normalized_weights = {k: v / total_weight for k, v in filtered_weights.items()}
                    return normalized_weights

            # 기본값 반환 (균등 가중치)
            equal_weight = 1.0 / len(available_indicators) if available_indicators else 0.0
            return {indicator: equal_weight for indicator in available_indicators}

        except Exception as e:
            logger.error(f"가중치 결정 중 오류: {str(e)}")
            # 오류 시 균등 가중치 반환
            available = [k for k in self.DEFAULT_WEIGHTS.keys() if analysis_results.get(k)]
            equal_weight = 1.0 / len(available) if available else 0.0
            return {k: equal_weight for k in available}

    def _apply_dynamic_weight_adjustment(
        self,
        base_weights: Dict[str, float],
        analysis_results: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """
        동적 가중치 조정 적용

        시장 상황에 따라 가중치를 자동 조정

        Args:
            base_weights: 기본 가중치
            analysis_results: 분석 결과

        Returns:
            조정된 가중치 또는 None (조정 불필요)
        """
        try:
            # ATR 데이터에서 변동성 정보 추출
            atr_data = analysis_results.get('atr')
            atr_percentage = 0.0
            if atr_data:
                atr_percentage = float(atr_data.get('atr_percentage', 0.0)) / 100.0

            # 스프레드 데이터에서 유동성 정보 추출
            spread_data = analysis_results.get('spread')
            spread_rate = 0.0
            if spread_data:
                current_analysis = spread_data.get('current_analysis', {})
                spread_rate = float(current_analysis.get('spread_rate', 0.0))

            # 조건 확인 및 가중치 조정
            for rule_name, rule in self.DYNAMIC_WEIGHT_RULES.items():
                condition = rule['condition']

                # 조건 평가
                condition_met = False
                if 'atr_percentage' in condition and atr_percentage > 0:
                    if 'atr_percentage > 0.08' in condition and atr_percentage > 0.08:
                        condition_met = True
                    elif 'atr_percentage < 0.03' in condition and atr_percentage < 0.03:
                        condition_met = True

                if 'spread_rate' in condition and spread_rate > 0:
                    if 'spread_rate > 0.005' in condition and spread_rate > 0.005:
                        condition_met = True

                if condition_met:
                    logger.info(f"동적 가중치 조정 적용: {rule_name}")
                    return rule['weights'].copy()

            return None  # 조정 불필요

        except Exception as e:
            logger.error(f"동적 가중치 조정 중 오류: {str(e)}")
            return None

    def _filter_outliers(self, symbol: str, scores: Dict[str, float]) -> Dict[str, float]:
        """
        이상치 필터링 (Z-score 기반)

        Args:
            symbol: 심볼
            scores: 정규화된 점수들

        Returns:
            필터링된 점수들
        """
        try:
            filtered_scores = scores.copy()

            # 심볼별 점수 이력이 충분한 경우에만 이상치 필터링 적용
            if symbol not in self._score_history:
                return filtered_scores

            history = self._score_history[symbol]
            if len(history) < 5:  # 최소 5개 이력 필요
                return filtered_scores

            # 각 지표별로 이상치 검사
            for indicator, score in scores.items():
                # 해당 지표의 과거 점수들 (추후 확장 시 지표별 이력 관리)
                # 현재는 종합 점수 이력만 사용
                if len(history) > 1:
                    mean_score = statistics.mean(history)
                    stdev_score = statistics.stdev(history) if len(history) > 1 else 0

                    if stdev_score > 0:
                        z_score = abs(score - mean_score) / stdev_score

                        if z_score > self.OUTLIER_Z_SCORE_THRESHOLD:
                            logger.warning(
                                f"{symbol} {indicator} 지표에서 이상치 감지 "
                                f"(Z-score: {z_score:.2f}, 점수: {score:.2f})"
                            )
                            # 이상치를 평균값으로 대체
                            filtered_scores[indicator] = mean_score
                            self._stats['filtered_outliers'] += 1

            return filtered_scores

        except Exception as e:
            logger.error(f"이상치 필터링 중 오류: {str(e)}")
            return scores

    def _calculate_weighted_score(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        가중 평균 점수 계산

        Args:
            scores: 정규화된 점수들
            weights: 가중치들

        Returns:
            가중 평균 점수 (0-100)
        """
        try:
            total_score = 0.0
            total_weight = 0.0

            for indicator, score in scores.items():
                if indicator in weights:
                    weight = weights[indicator]
                    total_score += score * weight
                    total_weight += weight

            if total_weight > 0:
                weighted_score = total_score / total_weight
                return max(0.0, min(100.0, weighted_score))
            else:
                return 0.0

        except Exception as e:
            logger.error(f"가중 점수 계산 중 오류: {str(e)}")
            return 0.0

    def _assess_score_quality(
        self,
        scores: Dict[str, float],
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        점수 품질 평가

        Args:
            scores: 정규화된 점수들
            analysis_results: 원본 분석 결과들

        Returns:
            품질 평가 결과
        """
        try:
            # 사용 가능한 지표 수
            available_indicators = len(scores)
            total_indicators = len(self.DEFAULT_WEIGHTS)

            # 데이터 품질 점수 계산
            data_quality_scores = []

            # 각 분석 결과의 품질 평가
            for indicator, result in analysis_results.items():
                if result and indicator in scores:
                    # 각 분석 서비스별 데이터 품질 확인
                    quality_score = self._evaluate_individual_quality(indicator, result)
                    data_quality_scores.append(quality_score)

            # 전체 품질 점수 계산
            coverage_score = available_indicators / total_indicators  # 커버리지 점수
            avg_data_quality = statistics.mean(data_quality_scores) if data_quality_scores else 0.0

            # 종합 품질 점수 (커버리지 40% + 데이터품질 60%)
            overall_quality = (coverage_score * 0.4) + (avg_data_quality * 0.6)

            return {
                'available_indicators': available_indicators,
                'total_indicators': total_indicators,
                'coverage_score': coverage_score,
                'avg_data_quality': avg_data_quality,
                'overall_quality': overall_quality,
                'individual_qualities': dict(zip(analysis_results.keys(), data_quality_scores)),
                'reliability_level': self._get_reliability_level(overall_quality)
            }

        except Exception as e:
            logger.error(f"점수 품질 평가 중 오류: {str(e)}")
            return {
                'available_indicators': 0,
                'total_indicators': 5,
                'coverage_score': 0.0,
                'avg_data_quality': 0.0,
                'overall_quality': 0.0,
                'individual_qualities': {},
                'reliability_level': '낮음'
            }

    def _evaluate_individual_quality(self, indicator: str, result: Dict[str, Any]) -> float:
        """
        개별 지표의 데이터 품질 평가

        Args:
            indicator: 지표 이름
            result: 분석 결과

        Returns:
            품질 점수 (0.0-1.0)
        """
        try:
            quality_score = 0.0

            if indicator == 'volume':
                # 거래량 데이터 품질 확인
                if result.get('current_volume_24h') and result.get('avg_volume_7d'):
                    quality_score += 0.5
                if result.get('surge_score') is not None:
                    quality_score += 0.3
                if result.get('pattern_analysis'):
                    quality_score += 0.2

            elif indicator == 'atr':
                # ATR 데이터 품질 확인
                if result.get('data_points_used', 0) >= ATRCalculator.MIN_DATA_POINTS:
                    quality_score += 0.6
                if result.get('atr_value') and result.get('current_price'):
                    quality_score += 0.4

            elif indicator == 'rsi':
                # RSI 데이터 품질 확인
                if result.get('data_points_used', 0) >= 14:  # RSI 계산 최소 기간
                    quality_score += 0.5
                if result.get('current_rsi') is not None:
                    quality_score += 0.3
                if result.get('trend_analysis'):
                    quality_score += 0.2

            elif indicator == 'bollinger':
                # 볼린저밴드 데이터 품질 확인
                if result.get('data_points_used', 0) >= 20:  # 볼린저밴드 계산 최소 기간
                    quality_score += 0.5
                if result.get('current_analysis', {}).get('band_position') is not None:
                    quality_score += 0.3
                if result.get('breakout_signals'):
                    quality_score += 0.2

            elif indicator == 'spread':
                # 스프레드 데이터 품질 확인
                if result.get('current_analysis', {}).get('spread_rate') is not None:
                    quality_score += 0.4
                if result.get('historical_analysis'):
                    quality_score += 0.3
                if result.get('market_depth_analysis'):
                    quality_score += 0.3

            return min(1.0, quality_score)

        except Exception as e:
            logger.error(f"{indicator} 개별 품질 평가 중 오류: {str(e)}")
            return 0.0

    def _get_reliability_level(self, quality_score: float) -> str:
        """품질 점수에 따른 신뢰도 레벨 반환"""
        if quality_score >= 0.8:
            return '매우 높음'
        elif quality_score >= 0.6:
            return '높음'
        elif quality_score >= 0.4:
            return '보통'
        elif quality_score >= 0.2:
            return '낮음'
        else:
            return '매우 낮음'

    def _generate_recommendation(
        self,
        comprehensive_score: float,
        quality_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        종합 점수와 품질을 바탕으로 투자 권고 생성

        Args:
            comprehensive_score: 종합 점수
            quality_assessment: 품질 평가

        Returns:
            투자 권고 정보
        """
        try:
            # 기본 등급 결정
            if comprehensive_score >= 80:
                grade = '매우 좋음'
                action = 'strong_buy'
            elif comprehensive_score >= 65:
                grade = '좋음'
                action = 'buy'
            elif comprehensive_score >= 50:
                grade = '보통'
                action = 'hold'
            elif comprehensive_score >= 35:
                grade = '나쁨'
                action = 'sell'
            else:
                grade = '매우 나쁨'
                action = 'strong_sell'

            # 품질을 고려한 신뢰도 조정
            reliability = quality_assessment.get('overall_quality', 0.0)
            if reliability < self.MIN_QUALITY_THRESHOLD:
                action = 'hold'  # 품질이 낮으면 보수적 판단
                grade += ' (저신뢰도)'

            return {
                'score': comprehensive_score,
                'grade': grade,
                'action': action,
                'reliability': quality_assessment.get('reliability_level', '알 수 없음'),
                'confidence': reliability * 100,
                'reasoning': self._generate_reasoning(comprehensive_score, quality_assessment)
            }

        except Exception as e:
            logger.error(f"투자 권고 생성 중 오류: {str(e)}")
            return {
                'score': comprehensive_score,
                'grade': '판정 불가',
                'action': 'hold',
                'reliability': '낮음',
                'confidence': 0.0,
                'reasoning': '분석 중 오류가 발생했습니다.'
            }

    def _generate_reasoning(
        self,
        score: float,
        quality: Dict[str, Any]
    ) -> str:
        """권고 근거 생성"""
        try:
            reasoning_parts = []

            # 점수 기반 근거
            if score >= 70:
                reasoning_parts.append("높은 종합 점수로 긍정적 신호")
            elif score <= 30:
                reasoning_parts.append("낮은 종합 점수로 부정적 신호")
            else:
                reasoning_parts.append("중간 수준의 종합 점수")

            # 품질 기반 근거
            coverage = quality.get('coverage_score', 0.0)
            if coverage >= 0.8:
                reasoning_parts.append("충분한 지표 커버리지")
            elif coverage <= 0.4:
                reasoning_parts.append("제한적인 지표 정보")

            # 신뢰도 기반 근거
            reliability = quality.get('overall_quality', 0.0)
            if reliability >= 0.7:
                reasoning_parts.append("높은 데이터 신뢰도")
            elif reliability <= 0.4:
                reasoning_parts.append("낮은 데이터 신뢰도로 신중한 판단 필요")

            return ". ".join(reasoning_parts) + "."

        except Exception:
            return "분석 결과를 바탕으로 한 종합 판단."

    def _update_score_history(self, symbol: str, score: float):
        """점수 이력 업데이트 (이상치 필터링용)"""
        try:
            if symbol not in self._score_history:
                self._score_history[symbol] = []

            self._score_history[symbol].append(score)

            # 최근 50개 점수만 보관 (메모리 관리)
            if len(self._score_history[symbol]) > 50:
                self._score_history[symbol] = self._score_history[symbol][-50:]

        except Exception as e:
            logger.error(f"점수 이력 업데이트 중 오류: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계 정보 반환"""
        return self._stats.copy()

    def get_score_history(self, symbol: str) -> List[float]:
        """특정 심볼의 점수 이력 반환"""
        return self._score_history.get(symbol, []).copy()

    async def health_check(self) -> Dict[str, Any]:
        """
        시스템 상태 확인

        Returns:
            상태 정보 딕셔너리
        """
        try:
            await self._ensure_analyzers()

            # 각 분석 서비스 상태 확인
            service_status = {}
            services = [
                ('volume', self._volume_analyzer),
                ('atr', self._atr_calculator),
                ('rsi', self._rsi_calculator),
                ('bollinger', self._bollinger_analyzer),
                ('spread', self._spread_analyzer)
            ]

            for name, service in services:
                if service:
                    try:
                        status = await service.health_check()
                        service_status[name] = status.get('database_connected', False)
                    except Exception as e:
                        service_status[name] = False
                        logger.error(f"{name} 서비스 상태 확인 실패: {str(e)}")
                else:
                    service_status[name] = False

            return {
                'service_name': 'ScoringSystem',
                'database_connected': await self.db_config.health_check(),
                'analyzer_services': service_status,
                'all_services_healthy': all(service_status.values()),
                'stats': self.get_stats()
            }

        except Exception as e:
            logger.error(f"상태 확인 중 오류: {str(e)}")
            return {
                'service_name': 'ScoringSystem',
                'database_connected': False,
                'analyzer_services': {},
                'all_services_healthy': False,
                'stats': self.get_stats(),
                'error': str(e)
            }

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        # 각 분석 서비스 정리
        services = [
            self._volume_analyzer,
            self._atr_calculator,
            self._rsi_calculator,
            self._bollinger_analyzer,
            self._spread_analyzer
        ]

        for service in services:
            if service:
                try:
                    await service.__aexit__(exc_type, exc_val, exc_tb)
                except Exception as e:
                    logger.error(f"분석 서비스 정리 중 오류: {str(e)}")


# 전역 인스턴스를 위한 팩토리 함수
_global_scoring_system: Optional[ScoringSystem] = None


async def get_scoring_system(db_config: Optional[DatabaseConfig] = None) -> ScoringSystem:
    """
    전역 ScoringSystem 인스턴스 반환

    Args:
        db_config: 데이터베이스 설정 객체 (선택적, None이면 기본값 생성)

    Returns:
        ScoringSystem 인스턴스
    """
    global _global_scoring_system

    if db_config is None:
        db_config = DatabaseConfig()

    if _global_scoring_system is None:
        _global_scoring_system = ScoringSystem(db_config)

    return _global_scoring_system


async def close_scoring_system():
    """전역 ScoringSystem 인스턴스 정리"""
    global _global_scoring_system

    if _global_scoring_system is not None:
        await _global_scoring_system.__aexit__(None, None, None)
        _global_scoring_system = None