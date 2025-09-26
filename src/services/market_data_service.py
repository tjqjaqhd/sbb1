"""
시장 데이터 서비스

빗썸 WebSocket에서 수신한 실시간 데이터를 데이터베이스에 저장하는 서비스입니다.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from decimal import Decimal

from src.api.bithumb.websocket_client import (
    BithumbWebSocketClient, SubscriptionType, get_websocket_client
)
from src.api.bithumb.message_parser import TickerData, OrderBookData, TransactionData
from src.database.config import DatabaseConfig
from src.database.models.market import MarketData, Ticker, OrderBook, Transaction
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select

logger = logging.getLogger(__name__)


class MarketDataService:
    """
    시장 데이터 서비스

    빗썸 WebSocket 데이터를 수신하여 데이터베이스에 저장하는 서비스
    """

    def __init__(self, db_config: DatabaseConfig):
        """
        서비스 초기화

        Args:
            db_config: 데이터베이스 설정 객체
        """
        self.db_config = db_config
        self.websocket_client: Optional[BithumbWebSocketClient] = None
        self.is_running = False
        self._stats = {
            'tickers_saved': 0,
            'orderbooks_saved': 0,
            'transactions_saved': 0,
            'errors': 0,
            'last_update': None
        }

    async def start(self, symbols: List[str] = None) -> bool:
        """
        시장 데이터 수집 시작

        Args:
            symbols: 수집할 심볼 리스트 (기본값: ["BTC_KRW", "ETH_KRW"])

        Returns:
            시작 성공 여부
        """
        if self.is_running:
            logger.warning("시장 데이터 서비스가 이미 실행 중입니다")
            return True

        if symbols is None:
            symbols = ["BTC_KRW", "ETH_KRW"]

        try:
            # WebSocket 클라이언트 가져오기
            self.websocket_client = await get_websocket_client()

            # 메시지 핸들러 등록
            self.websocket_client.add_message_handler(
                SubscriptionType.TICKER, self._handle_ticker_data
            )
            self.websocket_client.add_message_handler(
                SubscriptionType.ORDERBOOK, self._handle_orderbook_data
            )
            self.websocket_client.add_message_handler(
                SubscriptionType.TRANSACTION, self._handle_transaction_data
            )

            # WebSocket 연결
            if not await self.websocket_client.connect():
                logger.error("WebSocket 연결 실패")
                return False

            # 데이터 구독
            success_ticker = await self.websocket_client.subscribe(
                SubscriptionType.TICKER, symbols
            )
            success_orderbook = await self.websocket_client.subscribe(
                SubscriptionType.ORDERBOOK, symbols
            )
            success_transaction = await self.websocket_client.subscribe(
                SubscriptionType.TRANSACTION, symbols
            )

            if not all([success_ticker, success_orderbook, success_transaction]):
                logger.error("일부 데이터 구독이 실패했습니다")
                return False

            self.is_running = True
            logger.info(f"시장 데이터 서비스 시작 완료: {symbols}")

            # 메시지 수신 시작 (백그라운드 태스크)
            asyncio.create_task(self._start_receiving())

            return True

        except Exception as e:
            logger.error(f"시장 데이터 서비스 시작 실패: {str(e)}")
            return False

    async def stop(self):
        """시장 데이터 수집 중지"""
        if not self.is_running:
            return

        self.is_running = False

        if self.websocket_client:
            await self.websocket_client.disconnect()

        logger.info("시장 데이터 서비스 중지 완료")

    async def _start_receiving(self):
        """메시지 수신 시작 (백그라운드 태스크)"""
        try:
            await self.websocket_client.start_receiving()
        except Exception as e:
            logger.error(f"메시지 수신 중 오류: {str(e)}")
            self.is_running = False

    async def _handle_ticker_data(self, ticker_data, message):
        """
        Ticker 데이터 처리 및 저장

        Args:
            ticker_data: 파싱된 Ticker 데이터 또는 원본 dict
            message: 원본 WebSocket 메시지
        """
        try:
            # ticker_data가 dict인 경우 (파싱 실패 시 raw_data가 전달됨)
            if isinstance(ticker_data, dict):
                if ticker_data.get('type') == 'ticker' and 'content' in ticker_data:
                    # 빗썸 ticker 메시지 구조에서 데이터 추출
                    content = ticker_data['content']
                    symbol = content.get('symbol', 'BTC_KRW')

                    # 실제 빗썸 ticker 데이터 필드명 사용
                    ticker_info = {
                        'symbol': symbol,
                        'opening_price': content.get('openPrice'),
                        'closing_price': content.get('closePrice'),
                        'min_price': content.get('lowPrice'),
                        'max_price': content.get('highPrice'),
                        'average_price': content.get('averagePrice'),
                        'units_traded': content.get('unitsTraded'),  # 또는 volume
                        'volume_1day': content.get('volume1Day', content.get('volume')),
                        'volume_7day': content.get('volume7Day'),
                        'buy_price': content.get('buyPrice'),
                        'sell_price': content.get('sellPrice'),
                        'fluctate_24h': content.get('chgAmt'),  # 변동액
                        'fluctate_rate_24h': content.get('chgRate')  # 변동률
                    }

                    logger.debug(f"추출된 ticker 데이터 - symbol: {ticker_info['symbol']}, price: {ticker_info['closing_price']}")
                else:
                    logger.warning(f"Ticker 형태가 아닌 메시지: {ticker_data}")
                    return
            else:
                # TickerData 객체인 경우 (정상 파싱된 경우)
                ticker_info = {
                    'symbol': ticker_data.symbol,
                    'opening_price': ticker_data.opening_price,
                    'closing_price': ticker_data.closing_price,
                    'min_price': ticker_data.min_price,
                    'max_price': ticker_data.max_price,
                    'average_price': None,  # TickerData 모델에 없는 필드
                    'units_traded': ticker_data.units_traded,
                    'volume_1day': ticker_data.volume_1day,
                    'volume_7day': ticker_data.volume_7day,
                    'buy_price': ticker_data.buy_price,
                    'sell_price': ticker_data.sell_price,
                    'fluctate_24h': ticker_data.fluctate_24h,
                    'fluctate_rate_24h': ticker_data.fluctate_rate_24h
                }
            async with self.db_config.get_session() as session:
                # 기존 Ticker 데이터 조회
                existing_ticker = await session.execute(
                    select(Ticker).filter(Ticker.symbol == ticker_info['symbol'])
                )
                ticker_record = existing_ticker.scalar_one_or_none()

                # 안전한 Decimal 변환 함수
                def safe_decimal(value):
                    if value is None:
                        return None
                    try:
                        return Decimal(str(value))
                    except (ValueError, TypeError):
                        logger.warning(f"Decimal 변환 실패: {value}")
                        return None

                if ticker_record:
                    # 기존 레코드 업데이트
                    if ticker_info['opening_price']:
                        ticker_record.opening_price = safe_decimal(ticker_info['opening_price'])
                    if ticker_info['closing_price']:
                        ticker_record.closing_price = safe_decimal(ticker_info['closing_price'])
                    if ticker_info['min_price']:
                        ticker_record.min_price = safe_decimal(ticker_info['min_price'])
                    if ticker_info['max_price']:
                        ticker_record.max_price = safe_decimal(ticker_info['max_price'])

                    ticker_record.average_price = safe_decimal(ticker_info['average_price'])
                    ticker_record.units_traded = safe_decimal(ticker_info['units_traded'])
                    ticker_record.volume_1day = safe_decimal(ticker_info['volume_1day'])
                    ticker_record.volume_7day = safe_decimal(ticker_info['volume_7day'])
                    ticker_record.buy_price = safe_decimal(ticker_info['buy_price'])
                    ticker_record.sell_price = safe_decimal(ticker_info['sell_price'])
                    ticker_record.change_24h = safe_decimal(ticker_info['fluctate_24h'])
                    ticker_record.change_rate = safe_decimal(ticker_info['fluctate_rate_24h'])
                    ticker_record.updated_at = datetime.now(timezone.utc)
                else:
                    # 새 레코드 생성
                    ticker_record = Ticker(
                        symbol=ticker_info['symbol'],
                        opening_price=safe_decimal(ticker_info['opening_price']) or Decimal('0'),
                        closing_price=safe_decimal(ticker_info['closing_price']) or Decimal('0'),
                        min_price=safe_decimal(ticker_info['min_price']) or Decimal('0'),
                        max_price=safe_decimal(ticker_info['max_price']) or Decimal('0'),
                        average_price=safe_decimal(ticker_info['average_price']),
                        units_traded=safe_decimal(ticker_info['units_traded']),
                        volume_1day=safe_decimal(ticker_info['volume_1day']),
                        volume_7day=safe_decimal(ticker_info['volume_7day']),
                        buy_price=safe_decimal(ticker_info['buy_price']),
                        sell_price=safe_decimal(ticker_info['sell_price']),
                        change_24h=safe_decimal(ticker_info['fluctate_24h']),
                        change_rate=safe_decimal(ticker_info['fluctate_rate_24h'])
                    )
                    session.add(ticker_record)

                await session.commit()
                self._stats['tickers_saved'] += 1
                self._stats['last_update'] = datetime.now(timezone.utc)

                logger.debug(f"Ticker 데이터 저장 완료: {ticker_info['symbol']}")

        except SQLAlchemyError as e:
            logger.error(f"Ticker 데이터 저장 중 DB 오류: {str(e)}")
            self._stats['errors'] += 1
        except Exception as e:
            logger.error(f"Ticker 데이터 처리 중 오류: {str(e)}")
            self._stats['errors'] += 1

    async def _handle_orderbook_data(self, orderbook_raw_data, message):
        """
        OrderBook 데이터 처리 및 저장 (orderbookdepth 타입 처리)

        Args:
            orderbook_raw_data: 원본 OrderBook 메시지 (orderbookdepth)
            message: 원본 WebSocket 메시지
        """
        try:
            # orderbookdepth 메시지 구조 처리
            if isinstance(orderbook_raw_data, dict) and 'content' in orderbook_raw_data:
                content = orderbook_raw_data['content']
                entries = content.get('list', [])
            else:
                logger.warning("OrderBook 데이터 구조를 파싱할 수 없습니다")
                return

            if not entries:
                return

            # 첫 번째 엔트리에서 심볼 가져오기
            symbol = entries[0].get('symbol', 'BTC_KRW')

            async with self.db_config.get_session() as session:
                # 현재 시간 사용
                timestamp = datetime.now(timezone.utc)

                # bids/asks 데이터를 분류
                bids_json = []
                asks_json = []

                for entry in entries:
                    order_type = entry.get('orderType', 'bid')
                    order_data = {
                        "price": entry.get('price', '0'),
                        "quantity": entry.get('quantity', '0'),
                        "total": entry.get('total', '0')
                    }

                    if order_type == 'bid':
                        bids_json.append(order_data)
                    else:
                        asks_json.append(order_data)

                orderbook_record = OrderBook(
                    symbol=symbol,
                    timestamp=timestamp,
                    bids_data=bids_json,  # JSON 필드
                    asks_data=asks_json   # JSON 필드
                )

                session.add(orderbook_record)
                await session.commit()

                self._stats['orderbooks_saved'] += 1
                self._stats['last_update'] = datetime.now(timezone.utc)

                logger.debug(f"OrderBook 데이터 저장 완료: {symbol}")

        except SQLAlchemyError as e:
            logger.error(f"OrderBook 데이터 저장 중 DB 오류: {str(e)}")
            self._stats['errors'] += 1
        except Exception as e:
            logger.error(f"OrderBook 데이터 처리 중 오류: {str(e)}")
            self._stats['errors'] += 1

    async def _handle_transaction_data(self, transaction_raw_data, message):
        """
        Transaction 데이터 처리 및 저장 (빗썸 transaction 타입 처리)

        Args:
            transaction_raw_data: 원본 Transaction 메시지
            message: 원본 WebSocket 메시지
        """
        try:
            # transaction 메시지 구조 처리
            if isinstance(transaction_raw_data, dict) and 'content' in transaction_raw_data:
                content = transaction_raw_data['content']
                entries = content.get('list', [])
            else:
                logger.warning("Transaction 데이터 구조를 파싱할 수 없습니다")
                return

            if not entries:
                return

            async with self.db_config.get_session() as session:
                # Transaction 데이터는 리스트로 올 수 있으므로 각각 처리
                for entry in entries:
                    symbol = entry.get('symbol', 'BTC_KRW')

                    # Transaction ID는 심볼+시간+가격으로 생성 (고유성 보장)
                    cont_dtm = entry.get('contDtm', '')
                    cont_price = entry.get('contPrice', '0')
                    transaction_id = f"{symbol}_{cont_dtm}_{cont_price}"

                    # 체결 타입 변환 ('1' -> 'buy', '2' -> 'sell')
                    buy_sell_gb = entry.get('buySellGb', '1')
                    trade_type = 'buy' if buy_sell_gb == '1' else 'sell'

                    # 체결 시간 파싱
                    if cont_dtm:
                        try:
                            # YYYYMMDDHHMISS 형태를 파싱
                            timestamp = datetime.strptime(cont_dtm, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                        except ValueError:
                            timestamp = datetime.now(timezone.utc)
                    else:
                        timestamp = datetime.now(timezone.utc)

                    # 수량과 금액 처리
                    cont_qty = entry.get('contQty', '0')
                    cont_amt = entry.get('contAmt', '0')

                    # 총액이 없으면 가격 * 수량으로 계산
                    if not cont_amt or cont_amt == '0':
                        total = Decimal(str(cont_price)) * Decimal(str(cont_qty))
                    else:
                        total = Decimal(str(cont_amt))

                    transaction_record = Transaction(
                        symbol=symbol,
                        transaction_id=transaction_id,
                        price=Decimal(str(cont_price)),
                        quantity=Decimal(str(cont_qty)),
                        total=total,
                        type=trade_type,  # 'buy' 또는 'sell'
                        timestamp=timestamp
                    )

                    session.add(transaction_record)

                # 모든 Transaction을 한 번에 커밋
                await session.commit()

                self._stats['transactions_saved'] += len(entries)
                self._stats['last_update'] = datetime.now(timezone.utc)

                logger.debug(f"Transaction 데이터 저장 완료: {len(entries)}건")

        except SQLAlchemyError as e:
            logger.error(f"Transaction 데이터 저장 중 DB 오류: {str(e)}")
            self._stats['errors'] += 1
        except Exception as e:
            logger.error(f"Transaction 데이터 처리 중 오류: {str(e)}")
            self._stats['errors'] += 1

    async def save_market_data_to_timeseries(
        self,
        symbol: str,
        timeframe: str,
        ohlcv_data: Dict[str, Any]
    ) -> bool:
        """
        시계열 MarketData 테이블에 캔들 데이터 저장 (선택적 기능)

        Args:
            symbol: 심볼 (예: "BTC_KRW")
            timeframe: 시간 간격 (예: "1h", "1d")
            ohlcv_data: OHLCV 데이터 딕셔너리

        Returns:
            저장 성공 여부
        """
        try:
            async with self.db_config.get_session() as session:
                market_data = MarketData(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.fromtimestamp(ohlcv_data['timestamp'] / 1000, timezone.utc),
                    open_price=Decimal(str(ohlcv_data['open'])),
                    high_price=Decimal(str(ohlcv_data['high'])),
                    low_price=Decimal(str(ohlcv_data['low'])),
                    close_price=Decimal(str(ohlcv_data['close'])),
                    volume=Decimal(str(ohlcv_data['volume'])),
                    quote_volume=Decimal(str(ohlcv_data.get('quote_volume', 0))) if ohlcv_data.get('quote_volume') else None,
                    trade_count=ohlcv_data.get('trade_count')
                )

                session.add(market_data)
                await session.commit()

                logger.debug(f"MarketData 저장 완료: {symbol} {timeframe}")
                return True

        except SQLAlchemyError as e:
            logger.error(f"MarketData 저장 중 DB 오류: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"MarketData 저장 중 오류: {str(e)}")
            return False

    async def get_latest_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        특정 심볼의 최신 Ticker 데이터 조회

        Args:
            symbol: 심볼 (예: "BTC_KRW")

        Returns:
            Ticker 객체 또는 None
        """
        try:
            async with self.db_config.get_session() as session:
                result = await session.execute(
                    select(Ticker).filter(Ticker.symbol == symbol)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Ticker 조회 중 오류: {str(e)}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """서비스 통계 정보 반환"""
        return self._stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """
        서비스 상태 확인

        Returns:
            상태 정보 딕셔너리
        """
        websocket_status = {}
        if self.websocket_client:
            websocket_status = await self.websocket_client.health_check()

        return {
            'service_running': self.is_running,
            'websocket': websocket_status,
            'stats': self.get_stats(),
            'database_connected': await self.db_config.health_check()
        }

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.stop()