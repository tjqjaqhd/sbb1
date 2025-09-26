"""
스프레드 분석 서비스

빗썸 API에서 실시간 호가창 데이터를 수집하고 스프레드 분석을 통한 유동성 평가 시스템입니다.
Bid-Ask 스프레드 계산, 스프레드 변화율 분석, 유동성 점수 산출, 슬리피지 예측 등을 제공합니다.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal, InvalidOperation
from enum import Enum

from src.api.bithumb.client import get_http_client, BithumbHTTPClient
from src.database.config import DatabaseConfig
from src.database.models.market import OrderBook, Ticker
from sqlalchemy import select, func, and_
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class LiquidityLevel(Enum):
    """유동성 등급"""
    VERY_HIGH = "매우 높음"      # ≤ 0.1%
    HIGH = "높음"               # 0.1% ~ 0.3%
    MEDIUM = "보통"             # 0.3% ~ 0.5%
    LOW = "낮음"                # 0.5% ~ 1.0%
    VERY_LOW = "매우 낮음"      # > 1.0%


class SpreadAnalyzer:
    """
    스프레드 분석 서비스

    빗썸 API와 데이터베이스를 활용하여 호가창 스프레드를 분석하고
    유동성 평가 및 슬리피지 예측을 제공하는 서비스
    """

    def __init__(self, db_config: DatabaseConfig):
        """
        스프레드 분석기 초기화

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

    async def get_orderbook_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        실시간 호가창 데이터 수집

        Args:
            symbol: 분석할 심볼 (예: "BTC_KRW")

        Returns:
            호가창 데이터 딕셔너리 또는 None
        """
        try:
            await self._ensure_http_client()

            # 빗썸 공개 API에서 호가창 데이터 수집
            endpoint = f"/public/orderbook/{symbol}"
            response = await self._http_client.get(endpoint)

            self._stats['api_calls'] += 1

            if response.get('status') == '0000' and 'data' in response:
                orderbook_data = response['data']

                # 호가창 데이터 추출 및 정제
                bids = []
                asks = []

                # 매수 호가 (bids)
                if 'bids' in orderbook_data:
                    for bid in orderbook_data['bids']:
                        price = self._safe_decimal(bid.get('price'))
                        quantity = self._safe_decimal(bid.get('quantity'))
                        if price and quantity:
                            bids.append({'price': price, 'quantity': quantity})

                # 매도 호가 (asks)
                if 'asks' in orderbook_data:
                    for ask in orderbook_data['asks']:
                        price = self._safe_decimal(ask.get('price'))
                        quantity = self._safe_decimal(ask.get('quantity'))
                        if price and quantity:
                            asks.append({'price': price, 'quantity': quantity})

                processed_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now(timezone.utc),
                    'bids': bids,
                    'asks': asks,
                    'payment_total': self._safe_decimal(orderbook_data.get('payment_total')),
                    'order_currency': orderbook_data.get('order_currency'),
                    'payment_currency': orderbook_data.get('payment_currency')
                }

                logger.debug(f"{symbol} 호가창 데이터 수집 완료: 매수호가 {len(bids)}개, 매도호가 {len(asks)}개")
                return processed_data
            else:
                logger.warning(f"{symbol} 호가창 데이터 수집 실패: {response}")
                return None

        except Exception as e:
            logger.error(f"{symbol} 호가창 데이터 수집 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def calculate_spread_metrics(self, orderbook_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        스프레드 메트릭 계산

        Args:
            orderbook_data: 호가창 데이터

        Returns:
            스프레드 분석 결과 또는 None
        """
        try:
            symbol = orderbook_data.get('symbol')
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])

            if not bids or not asks:
                logger.warning(f"{symbol} 호가 데이터 부족: 매수호가 {len(bids)}개, 매도호가 {len(asks)}개")
                return None

            # 최우선 매수/매도 호가 추출
            best_bid = bids[0]['price']
            best_ask = asks[0]['price']
            best_bid_qty = bids[0]['quantity']
            best_ask_qty = asks[0]['quantity']

            # 중간가격 계산
            mid_price = (best_bid + best_ask) / Decimal('2')

            # 스프레드 계산
            spread_absolute = best_ask - best_bid
            spread_rate = (spread_absolute / mid_price) * Decimal('100')  # 백분율

            # 유동성 등급 결정
            liquidity_level = self._determine_liquidity_level(float(spread_rate))

            # 시장 깊이 분석 (상위 5호가)
            market_depth = await self._analyze_market_depth(bids, asks)

            spread_metrics = {
                'symbol': symbol,
                'timestamp': orderbook_data.get('timestamp'),
                'best_bid': best_bid,
                'best_ask': best_ask,
                'best_bid_quantity': best_bid_qty,
                'best_ask_quantity': best_ask_qty,
                'mid_price': mid_price,
                'spread_absolute': spread_absolute,
                'spread_rate': spread_rate,
                'liquidity_level': liquidity_level.value,
                'liquidity_score': self._calculate_liquidity_score(float(spread_rate)),
                'market_depth': market_depth,
                'imbalance_ratio': self._calculate_imbalance_ratio(best_bid_qty, best_ask_qty)
            }

            logger.debug(
                f"{symbol} 스프레드 분석 - "
                f"스프레드율: {spread_rate:.4f}%, "
                f"유동성: {liquidity_level.value}, "
                f"중간가: {mid_price}"
            )

            return spread_metrics

        except Exception as e:
            logger.error(f"스프레드 메트릭 계산 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def _analyze_market_depth(self, bids: List[Dict], asks: List[Dict]) -> Dict[str, Any]:
        """
        시장 깊이 분석 (상위 5호가)

        Args:
            bids: 매수 호가 리스트
            asks: 매도 호가 리스트

        Returns:
            시장 깊이 분석 결과
        """
        try:
            # 상위 5호가 분석
            depth_levels = min(5, len(bids), len(asks))

            bid_depth = sum(bid['quantity'] for bid in bids[:depth_levels])
            ask_depth = sum(ask['quantity'] for ask in asks[:depth_levels])
            total_depth = bid_depth + ask_depth

            # 가격 범위 계산
            if depth_levels > 0:
                bid_price_range = bids[0]['price'] - bids[min(4, len(bids)-1)]['price']
                ask_price_range = asks[min(4, len(asks)-1)]['price'] - asks[0]['price']
                total_price_range = ask_price_range + bid_price_range
            else:
                bid_price_range = ask_price_range = total_price_range = Decimal('0')

            return {
                'depth_levels': depth_levels,
                'total_quantity': total_depth,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'depth_imbalance': self._calculate_imbalance_ratio(bid_depth, ask_depth),
                'bid_price_range': bid_price_range,
                'ask_price_range': ask_price_range,
                'total_price_range': total_price_range,
                'depth_score': self._calculate_depth_score(total_depth, total_price_range)
            }

        except Exception as e:
            logger.error(f"시장 깊이 분석 중 오류: {str(e)}")
            return {
                'depth_levels': 0,
                'total_quantity': Decimal('0'),
                'bid_depth': Decimal('0'),
                'ask_depth': Decimal('0'),
                'depth_imbalance': 0.0,
                'bid_price_range': Decimal('0'),
                'ask_price_range': Decimal('0'),
                'total_price_range': Decimal('0'),
                'depth_score': 0.0
            }

    async def predict_slippage(
        self,
        orderbook_data: Dict[str, Any],
        order_size: Decimal,
        side: str = "BUY"
    ) -> Optional[Dict[str, Any]]:
        """
        슬리피지 예측

        Args:
            orderbook_data: 호가창 데이터
            order_size: 주문 크기 (매수/매도 수량)
            side: 주문 방향 ("BUY" 또는 "SELL")

        Returns:
            슬리피지 예측 결과 또는 None
        """
        try:
            symbol = orderbook_data.get('symbol')
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])

            if side.upper() == "BUY":
                # 매수 주문 - 매도 호가에서 체결
                orders = asks
                best_price = asks[0]['price'] if asks else None
            else:
                # 매도 주문 - 매수 호가에서 체결
                orders = bids
                best_price = bids[0]['price'] if bids else None

            if not orders or not best_price:
                logger.warning(f"{symbol} 슬리피지 예측: 호가 데이터 부족")
                return None

            # 슬리피지 계산
            remaining_size = order_size
            total_cost = Decimal('0')
            filled_quantity = Decimal('0')
            levels_used = 0

            for order in orders:
                if remaining_size <= 0:
                    break

                available_qty = min(remaining_size, order['quantity'])
                total_cost += available_qty * order['price']
                filled_quantity += available_qty
                remaining_size -= available_qty
                levels_used += 1

            if filled_quantity > 0:
                average_fill_price = total_cost / filled_quantity
                slippage_absolute = abs(average_fill_price - best_price)
                slippage_rate = (slippage_absolute / best_price) * Decimal('100')

                fill_ratio = (filled_quantity / order_size) * Decimal('100')
            else:
                average_fill_price = best_price
                slippage_absolute = Decimal('0')
                slippage_rate = Decimal('0')
                fill_ratio = Decimal('0')

            slippage_prediction = {
                'symbol': symbol,
                'order_size': order_size,
                'order_side': side.upper(),
                'best_price': best_price,
                'average_fill_price': average_fill_price,
                'total_cost': total_cost,
                'filled_quantity': filled_quantity,
                'remaining_quantity': remaining_size,
                'fill_ratio': fill_ratio,
                'slippage_absolute': slippage_absolute,
                'slippage_rate': slippage_rate,
                'levels_used': levels_used,
                'is_fully_fillable': remaining_size <= 0,
                'slippage_category': self._categorize_slippage(float(slippage_rate))
            }

            logger.debug(
                f"{symbol} 슬리피지 예측 - "
                f"주문크기: {order_size}, "
                f"슬리피지율: {slippage_rate:.4f}%, "
                f"체결률: {fill_ratio:.2f}%"
            )

            return slippage_prediction

        except Exception as e:
            logger.error(f"{symbol} 슬리피지 예측 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def get_historical_spread_analysis(
        self,
        symbol: str,
        hours_back: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        과거 스프레드 변화 분석

        Args:
            symbol: 분석할 심볼
            hours_back: 분석할 시간 범위 (시간)

        Returns:
            과거 스프레드 분석 결과 또는 None
        """
        try:
            # 데이터베이스 초기화 확인
            if not await self.db_config.health_check():
                logger.warning(f"{symbol} 과거 스프레드 분석: 데이터베이스 연결 비활성")
                return None

            async with self.db_config.get_session() as session:
                # 과거 N시간 호가 데이터 조회
                end_time = datetime.now(timezone.utc)
                start_time = end_time - timedelta(hours=hours_back)

                query = select(
                    OrderBook.timestamp,
                    OrderBook.bid_prices,
                    OrderBook.ask_prices,
                    OrderBook.bid_quantities,
                    OrderBook.ask_quantities
                ).where(
                    and_(
                        OrderBook.symbol == symbol,
                        OrderBook.timestamp >= start_time,
                        OrderBook.timestamp <= end_time,
                        OrderBook.bid_prices.is_not(None),
                        OrderBook.ask_prices.is_not(None)
                    )
                ).order_by(OrderBook.timestamp.desc()).limit(100)

                result = await session.execute(query)
                historical_data = result.fetchall()

                self._stats['db_queries'] += 1

                if not historical_data:
                    logger.warning(f"{symbol} 과거 스프레드 데이터가 없습니다")
                    return None

                # 스프레드 변화 분석
                spreads = []
                liquidity_scores = []

                for row in historical_data:
                    if row.bid_prices and row.ask_prices and len(row.bid_prices) > 0 and len(row.ask_prices) > 0:
                        best_bid = row.bid_prices[0]
                        best_ask = row.ask_prices[0]
                        mid_price = (best_bid + best_ask) / Decimal('2')
                        spread_rate = ((best_ask - best_bid) / mid_price) * Decimal('100')

                        spreads.append(float(spread_rate))
                        liquidity_scores.append(self._calculate_liquidity_score(float(spread_rate)))

                if not spreads:
                    return None

                # 통계 계산
                avg_spread = statistics.mean(spreads)
                min_spread = min(spreads)
                max_spread = max(spreads)
                spread_volatility = statistics.stdev(spreads) if len(spreads) > 1 else 0.0

                avg_liquidity = statistics.mean(liquidity_scores)

                # 트렌드 분석 (최근 25% vs 전체 평균)
                recent_count = max(1, len(spreads) // 4)
                recent_spreads = spreads[:recent_count]
                recent_avg = statistics.mean(recent_spreads)

                trend = "개선" if recent_avg < avg_spread else "악화" if recent_avg > avg_spread else "안정"

                historical_analysis = {
                    'symbol': symbol,
                    'analysis_period_hours': hours_back,
                    'sample_count': len(spreads),
                    'avg_spread_rate': avg_spread,
                    'min_spread_rate': min_spread,
                    'max_spread_rate': max_spread,
                    'spread_volatility': spread_volatility,
                    'avg_liquidity_score': avg_liquidity,
                    'recent_avg_spread': recent_avg,
                    'trend': trend,
                    'trend_strength': abs(recent_avg - avg_spread) / avg_spread if avg_spread > 0 else 0,
                    'timestamp': datetime.now(timezone.utc)
                }

                logger.debug(f"{symbol} 과거 스프레드 분석 완료: 평균 {avg_spread:.4f}%, 트렌드 {trend}")
                return historical_analysis

        except Exception as e:
            logger.error(f"{symbol} 과거 스프레드 분석 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    async def comprehensive_spread_analysis(
        self,
        symbol: str,
        order_sizes: Optional[List[Decimal]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        종합적인 스프레드 분석

        실시간 스프레드, 유동성 평가, 슬리피지 예측, 과거 분석을 모두 포함하는 종합 분석

        Args:
            symbol: 분석할 심볼
            order_sizes: 슬리피지 예측할 주문 크기 리스트

        Returns:
            종합 스프레드 분석 결과
        """
        try:
            logger.info(f"{symbol} 종합 스프레드 분석 시작")

            # 기본 주문 크기 설정 (지정되지 않은 경우)
            if order_sizes is None:
                order_sizes = [Decimal('1'), Decimal('5'), Decimal('10'), Decimal('50')]

            # 1. 실시간 호가창 데이터 수집
            orderbook_data = await self.get_orderbook_data(symbol)
            if not orderbook_data:
                logger.error(f"{symbol} 호가창 데이터 수집 실패")
                return None

            # 2. 스프레드 메트릭 계산
            spread_metrics = await self.calculate_spread_metrics(orderbook_data)
            if not spread_metrics:
                logger.error(f"{symbol} 스프레드 메트릭 계산 실패")
                return None

            # 3. 슬리피지 예측 (여러 주문 크기별)
            slippage_predictions = {}
            for size in order_sizes:
                buy_slippage = await self.predict_slippage(orderbook_data, size, "BUY")
                sell_slippage = await self.predict_slippage(orderbook_data, size, "SELL")

                slippage_predictions[str(size)] = {
                    'buy': buy_slippage,
                    'sell': sell_slippage
                }

            # 4. 과거 스프레드 분석
            historical_analysis = await self.get_historical_spread_analysis(symbol, 24)

            # 5. 종합 평가 점수 계산
            comprehensive_score = self._calculate_comprehensive_spread_score(
                spread_metrics, historical_analysis
            )

            # 결과 정리
            analysis_result = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc),
                'spread_metrics': spread_metrics,
                'slippage_predictions': slippage_predictions,
                'historical_analysis': historical_analysis,
                'comprehensive_score': comprehensive_score,
                'trading_recommendation': self._generate_trading_recommendation(
                    spread_metrics, comprehensive_score
                )
            }

            self._stats['analysis_count'] += 1
            self._stats['last_analysis'] = datetime.now(timezone.utc)

            logger.info(
                f"{symbol} 종합 스프레드 분석 완료 - "
                f"스프레드율: {spread_metrics.get('spread_rate', 0):.4f}%, "
                f"유동성: {spread_metrics.get('liquidity_level', 'N/A')}, "
                f"종합점수: {comprehensive_score:.2f}"
            )

            return analysis_result

        except Exception as e:
            logger.error(f"{symbol} 종합 스프레드 분석 중 오류: {str(e)}")
            self._stats['errors'] += 1
            return None

    def _determine_liquidity_level(self, spread_rate: float) -> LiquidityLevel:
        """스프레드율에 따른 유동성 등급 결정"""
        if spread_rate <= 0.1:
            return LiquidityLevel.VERY_HIGH
        elif spread_rate <= 0.3:
            return LiquidityLevel.HIGH
        elif spread_rate <= 0.5:
            return LiquidityLevel.MEDIUM
        elif spread_rate <= 1.0:
            return LiquidityLevel.LOW
        else:
            return LiquidityLevel.VERY_LOW

    def _calculate_liquidity_score(self, spread_rate: float) -> float:
        """
        유동성 점수 계산 (0 ~ 100)

        스프레드율이 낮을수록 높은 점수
        """
        if spread_rate <= 0.1:
            return 100.0
        elif spread_rate <= 0.3:
            return 100.0 - ((spread_rate - 0.1) / 0.2) * 20.0  # 100 -> 80
        elif spread_rate <= 0.5:
            return 80.0 - ((spread_rate - 0.3) / 0.2) * 30.0   # 80 -> 50
        elif spread_rate <= 1.0:
            return 50.0 - ((spread_rate - 0.5) / 0.5) * 30.0   # 50 -> 20
        else:
            return max(0.0, 20.0 - (spread_rate - 1.0) * 10.0)  # 20 -> 0

    def _calculate_imbalance_ratio(self, bid_qty: Decimal, ask_qty: Decimal) -> float:
        """
        호가 불균형 비율 계산

        양수: 매수 우세, 음수: 매도 우세, 0: 균형
        """
        try:
            total = bid_qty + ask_qty
            if total > 0:
                return float((bid_qty - ask_qty) / total)
            return 0.0
        except Exception:
            return 0.0

    def _calculate_depth_score(self, total_quantity: Decimal, price_range: Decimal) -> float:
        """
        시장 깊이 점수 계산 (0 ~ 100)

        거래량이 많고 가격 범위가 좁을수록 높은 점수
        """
        try:
            if total_quantity <= 0:
                return 0.0

            # 거래량 점수 (로그 스케일)
            import math
            quantity_score = min(50.0, math.log10(float(total_quantity) + 1) * 10.0)

            # 가격 범위 점수 (범위가 좁을수록 높은 점수)
            if price_range > 0:
                range_score = min(50.0, 50.0 / (float(price_range) + 1))
            else:
                range_score = 50.0

            return quantity_score + range_score

        except Exception:
            return 0.0

    def _categorize_slippage(self, slippage_rate: float) -> str:
        """슬리피지 범주 분류"""
        if slippage_rate <= 0.05:
            return "매우 낮음"
        elif slippage_rate <= 0.1:
            return "낮음"
        elif slippage_rate <= 0.3:
            return "보통"
        elif slippage_rate <= 0.5:
            return "높음"
        else:
            return "매우 높음"

    def _calculate_comprehensive_spread_score(
        self,
        spread_metrics: Dict[str, Any],
        historical_analysis: Optional[Dict[str, Any]]
    ) -> float:
        """
        종합 스프레드 점수 계산 (0 ~ 100)

        현재 스프레드, 시장 깊이, 과거 트렌드를 종합하여 최종 점수 계산
        """
        try:
            score = 0.0
            weight_sum = 0.0

            # 1. 현재 유동성 점수 (가중치: 50%)
            liquidity_score = spread_metrics.get('liquidity_score', 0)
            score += liquidity_score * 0.5
            weight_sum += 0.5

            # 2. 시장 깊이 점수 (가중치: 30%)
            market_depth = spread_metrics.get('market_depth', {})
            depth_score = market_depth.get('depth_score', 0)
            score += depth_score * 0.3
            weight_sum += 0.3

            # 3. 과거 트렌드 점수 (가중치: 20%)
            if historical_analysis:
                trend = historical_analysis.get('trend', '안정')
                trend_strength = historical_analysis.get('trend_strength', 0)

                if trend == '개선':
                    trend_score = 80.0 + min(20.0, trend_strength * 100)
                elif trend == '악화':
                    trend_score = 20.0 - min(20.0, trend_strength * 100)
                else:
                    trend_score = 50.0

                score += trend_score * 0.2
                weight_sum += 0.2

            # 가중 평균 계산
            if weight_sum > 0:
                final_score = score / weight_sum
            else:
                final_score = liquidity_score

            return max(0.0, min(100.0, final_score))

        except Exception as e:
            logger.error(f"종합 스프레드 점수 계산 중 오류: {str(e)}")
            return 0.0

    def _generate_trading_recommendation(
        self,
        spread_metrics: Dict[str, Any],
        comprehensive_score: float
    ) -> Dict[str, Any]:
        """거래 추천 생성"""
        try:
            spread_rate = float(spread_metrics.get('spread_rate', 0))
            liquidity_level = spread_metrics.get('liquidity_level', '매우 낮음')
            imbalance_ratio = spread_metrics.get('imbalance_ratio', 0)

            # 추천 등급 결정
            if comprehensive_score >= 80:
                recommendation = "매우 추천"
                reason = "높은 유동성과 좁은 스프레드로 거래 비용이 낮음"
            elif comprehensive_score >= 60:
                recommendation = "추천"
                reason = "양호한 유동성으로 적절한 거래 환경"
            elif comprehensive_score >= 40:
                recommendation = "보통"
                reason = "보통 수준의 유동성, 주의 깊은 거래 필요"
            elif comprehensive_score >= 20:
                recommendation = "비추천"
                reason = "낮은 유동성으로 거래 비용이 높음"
            else:
                recommendation = "매우 비추천"
                reason = "매우 낮은 유동성으로 슬리피지 위험 높음"

            # 추가 주의사항
            warnings = []
            if spread_rate > 0.5:
                warnings.append("스프레드가 높아 거래 비용 증가")
            if abs(imbalance_ratio) > 0.3:
                warnings.append("호가 불균형으로 가격 변동 가능성")

            return {
                'recommendation': recommendation,
                'score': comprehensive_score,
                'reason': reason,
                'warnings': warnings,
                'optimal_order_size': self._suggest_optimal_order_size(spread_metrics),
                'timing_suggestion': "즉시" if comprehensive_score >= 70 else "주의 깊게"
            }

        except Exception as e:
            logger.error(f"거래 추천 생성 중 오류: {str(e)}")
            return {
                'recommendation': "분석 불가",
                'score': 0.0,
                'reason': "데이터 부족으로 분석 실패",
                'warnings': ["분석 오류 발생"],
                'optimal_order_size': "소량",
                'timing_suggestion': "대기"
            }

    def _suggest_optimal_order_size(self, spread_metrics: Dict[str, Any]) -> str:
        """최적 주문 크기 제안"""
        try:
            market_depth = spread_metrics.get('market_depth', {})
            total_quantity = market_depth.get('total_quantity', 0)

            if total_quantity > 100:
                return "대량 가능"
            elif total_quantity > 50:
                return "중량"
            elif total_quantity > 10:
                return "소량"
            else:
                return "최소량"

        except Exception:
            return "소량"

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
            'service_name': 'SpreadAnalyzer',
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
_global_spread_analyzer: Optional[SpreadAnalyzer] = None


async def get_spread_analyzer(db_config: Optional[DatabaseConfig] = None) -> SpreadAnalyzer:
    """
    전역 SpreadAnalyzer 인스턴스 반환

    Args:
        db_config: 데이터베이스 설정 객체 (선택적, None이면 기본값 생성)

    Returns:
        SpreadAnalyzer 인스턴스
    """
    global _global_spread_analyzer

    if db_config is None:
        db_config = DatabaseConfig()

    if _global_spread_analyzer is None:
        _global_spread_analyzer = SpreadAnalyzer(db_config)

    return _global_spread_analyzer


async def close_spread_analyzer():
    """전역 SpreadAnalyzer 인스턴스 정리"""
    global _global_spread_analyzer

    if _global_spread_analyzer is not None:
        await _global_spread_analyzer.__aexit__(None, None, None)
        _global_spread_analyzer = None