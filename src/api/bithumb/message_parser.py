"""
WebSocket 메시지 파싱 및 검증 시스템

빗썸 WebSocket에서 수신한 JSON 메시지를 파싱하고 검증하는 시스템입니다.
Pydantic을 활용하여 데이터 모델을 정의하고 타입 안전성을 보장합니다.
"""

import json
import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, List, Optional, Union, Literal, Type
from enum import Enum
import traceback

from pydantic import BaseModel, Field, validator, ValidationError, model_validator
from pydantic.types import PositiveInt, NonNegativeFloat

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket 메시지 타입"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRANSACTION = "transaction"
    ERROR = "error"
    STATUS = "status"


class OrderType(str, Enum):
    """주문 타입"""
    BUY = "1"  # 매수
    SELL = "2"  # 매도


class TickerData(BaseModel):
    """
    실시간 시세(Ticker) 데이터 모델

    빗썸 WebSocket의 ticker 메시지 구조를 정의합니다.
    """
    symbol: str = Field(..., description="거래 심볼 (예: BTC_KRW)")

    # 가격 정보
    opening_price: Optional[Decimal] = Field(None, alias="openPrice", description="시가")
    closing_price: Optional[Decimal] = Field(None, alias="closePrice", description="현재가")
    min_price: Optional[Decimal] = Field(None, alias="minPrice", description="최저가")
    max_price: Optional[Decimal] = Field(None, alias="maxPrice", description="최고가")

    # 거래량 정보
    units_traded: Optional[Decimal] = Field(None, alias="unitsTraded", description="거래량")
    volume_1day: Optional[Decimal] = Field(None, alias="volume1day", description="1일 거래량")
    volume_7day: Optional[Decimal] = Field(None, alias="volume7day", description="7일 거래량")

    # 변동 정보
    buy_price: Optional[Decimal] = Field(None, alias="buyPrice", description="매수 호가")
    sell_price: Optional[Decimal] = Field(None, alias="sellPrice", description="매도 호가")

    # 변동률 정보 (문자열로 받아서 Decimal로 변환)
    fluctate_24h: Optional[Decimal] = Field(None, alias="fluctate24H", description="24시간 변동률")
    fluctate_rate_24h: Optional[Decimal] = Field(None, alias="fluctateRate24H", description="24시간 변동률(%)")

    # 타임스탬프
    date: Optional[str] = Field(None, description="일자")

    class Config:
        # 알 수 없는 필드 허용
        extra = "allow"
        # 별칭 사용 허용
        validate_by_name = True

    @validator('symbol')
    def validate_symbol(cls, v):
        """심볼 형식 검증"""
        if not v or "_" not in v:
            raise ValueError("심볼은 'BASE_QUOTE' 형식이어야 합니다 (예: BTC_KRW)")
        return v.upper()

    @validator('opening_price', 'closing_price', 'min_price', 'max_price',
              'buy_price', 'sell_price', pre=True)
    def validate_price_fields(cls, v):
        """가격 필드 검증 및 변환"""
        if v is None or v == "":
            return None

        try:
            return Decimal(str(v))
        except (InvalidOperation, ValueError):
            logger.warning(f"유효하지 않은 가격 값: {v}")
            return None

    @validator('units_traded', 'volume_1day', 'volume_7day', pre=True)
    def validate_volume_fields(cls, v):
        """거래량 필드 검증 및 변환"""
        if v is None or v == "":
            return None

        try:
            decimal_value = Decimal(str(v))
            if decimal_value < 0:
                logger.warning(f"음수 거래량 값: {v}")
                return None
            return decimal_value
        except (InvalidOperation, ValueError):
            logger.warning(f"유효하지 않은 거래량 값: {v}")
            return None

    @validator('fluctate_24h', 'fluctate_rate_24h', pre=True)
    def validate_fluctation_fields(cls, v):
        """변동률 필드 검증 및 변환"""
        if v is None or v == "":
            return None

        try:
            return Decimal(str(v))
        except (InvalidOperation, ValueError):
            logger.warning(f"유효하지 않은 변동률 값: {v}")
            return None


class OrderBookEntry(BaseModel):
    """호가 엔트리 모델"""
    price: Decimal = Field(..., description="가격")
    quantity: Decimal = Field(..., description="수량")

    @validator('price', 'quantity', pre=True)
    def validate_decimal_fields(cls, v):
        """Decimal 필드 검증"""
        if v is None or v == "":
            raise ValueError("가격과 수량은 필수 값입니다")

        try:
            decimal_value = Decimal(str(v))
            if decimal_value <= 0:
                raise ValueError("가격과 수량은 양수여야 합니다")
            return decimal_value
        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"유효하지 않은 숫자 형식: {v}")


class OrderBookData(BaseModel):
    """
    실시간 호가(Orderbook) 데이터 모델

    빗썸 WebSocket의 orderbook 메시지 구조를 정의합니다.
    """
    symbol: str = Field(..., description="거래 심볼")

    # 매수/매도 호가 리스트
    bids: List[OrderBookEntry] = Field(default_factory=list, description="매수 호가")
    asks: List[OrderBookEntry] = Field(default_factory=list, description="매도 호가")

    # 타임스탬프
    timestamp: Optional[str] = Field(None, description="타임스탬프")

    class Config:
        extra = "allow"

    @validator('symbol')
    def validate_symbol(cls, v):
        """심볼 형식 검증"""
        if not v or "_" not in v:
            raise ValueError("심볼은 'BASE_QUOTE' 형식이어야 합니다")
        return v.upper()

    @validator('bids', 'asks', pre=True)
    def validate_order_entries(cls, v):
        """호가 엔트리 리스트 검증"""
        if not isinstance(v, list):
            logger.warning("호가 데이터가 리스트 형식이 아닙니다")
            return []

        valid_entries = []
        for entry in v:
            try:
                if isinstance(entry, dict) and 'price' in entry and 'quantity' in entry:
                    valid_entries.append(OrderBookEntry(**entry))
                elif isinstance(entry, list) and len(entry) >= 2:
                    # [price, quantity] 형태인 경우
                    valid_entries.append(OrderBookEntry(price=entry[0], quantity=entry[1]))
            except ValidationError as e:
                logger.warning(f"호가 엔트리 검증 실패: {entry}, 오류: {e}")
                continue

        return valid_entries

    @model_validator(mode='after')
    def validate_orderbook_integrity(self):
        """호가북 무결성 검증"""
        # 매수 호가는 내림차순, 매도 호가는 오름차순으로 정렬되어야 함
        if self.bids:
            self.bids.sort(key=lambda x: x.price, reverse=True)  # 높은 가격 순
        if self.asks:
            self.asks.sort(key=lambda x: x.price)  # 낮은 가격 순

        return self


class TransactionData(BaseModel):
    """
    실시간 체결(Transaction) 데이터 모델

    빗썸 WebSocket의 transaction 메시지 구조를 정의합니다.
    """
    symbol: str = Field(..., description="거래 심볼")

    # 체결 정보
    buy_sell_gb: OrderType = Field(..., alias="buySellGb", description="매수/매도 구분")
    cont_price: Decimal = Field(..., alias="contPrice", description="체결 가격")
    cont_qty: Decimal = Field(..., alias="contQty", description="체결 수량")
    cont_amt: Optional[Decimal] = Field(None, alias="contAmt", description="체결 금액")

    # 타임스탬프
    cont_dtm: Optional[str] = Field(None, alias="contDtm", description="체결 시간")
    updn: Optional[str] = Field(None, description="전일대비 구분")

    class Config:
        extra = "allow"
        allow_population_by_field_name = True

    @validator('symbol')
    def validate_symbol(cls, v):
        """심볼 형식 검증"""
        if not v or "_" not in v:
            raise ValueError("심볼은 'BASE_QUOTE' 형식이어야 합니다")
        return v.upper()

    @validator('cont_price', 'cont_qty', 'cont_amt', pre=True)
    def validate_transaction_amounts(cls, v):
        """거래 금액 필드 검증"""
        if v is None or v == "":
            return None

        try:
            decimal_value = Decimal(str(v))
            if decimal_value <= 0:
                raise ValueError("체결 가격과 수량은 양수여야 합니다")
            return decimal_value
        except (InvalidOperation, ValueError) as e:
            raise ValueError(f"유효하지 않은 체결 정보: {v}")


class ErrorData(BaseModel):
    """에러 메시지 데이터 모델"""
    error_code: Optional[str] = Field(None, alias="errorCode", description="에러 코드")
    error_message: Optional[str] = Field(None, alias="errorMessage", description="에러 메시지")

    class Config:
        extra = "allow"
        allow_population_by_field_name = True


class StatusData(BaseModel):
    """상태 메시지 데이터 모델"""
    status: Optional[str] = Field(None, description="상태")
    message: Optional[str] = Field(None, description="메시지")

    class Config:
        extra = "allow"


class WebSocketMessage(BaseModel):
    """
    WebSocket 메시지 래퍼 모델

    모든 WebSocket 메시지의 공통 구조를 정의합니다.
    """
    type: MessageType = Field(..., description="메시지 타입")
    content: Optional[Dict[str, Any]] = Field(None, description="메시지 내용")

    # 파싱된 데이터 (타입에 따라 다른 모델)
    ticker_data: Optional[TickerData] = Field(None, description="Ticker 데이터")
    orderbook_data: Optional[OrderBookData] = Field(None, description="Orderbook 데이터")
    transaction_data: Optional[List[TransactionData]] = Field(None, description="Transaction 데이터")
    error_data: Optional[ErrorData] = Field(None, description="에러 데이터")
    status_data: Optional[StatusData] = Field(None, description="상태 데이터")

    # 메타데이터
    parsed_at: datetime = Field(default_factory=datetime.now, description="파싱 시간")
    raw_data: Optional[Dict[str, Any]] = Field(None, description="원본 데이터")

    class Config:
        extra = "allow"


class MessageParsingError(Exception):
    """메시지 파싱 오류"""

    def __init__(self, message: str, raw_data: Any = None, cause: Exception = None):
        self.message = message
        self.raw_data = raw_data
        self.cause = cause
        super().__init__(message)


class MessageParser:
    """
    WebSocket 메시지 파서

    빗썸 WebSocket 메시지를 파싱하고 검증하는 클래스입니다.
    """

    def __init__(self, strict_mode: bool = False):
        """
        파서 초기화

        Args:
            strict_mode: 엄격 모드 (검증 실패시 예외 발생)
        """
        self.strict_mode = strict_mode
        self.stats = {
            'total_messages': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'by_type': {
                'ticker': 0,
                'orderbook': 0,
                'transaction': 0,
                'error': 0,
                'status': 0,
                'unknown': 0
            },
            'errors': []
        }

    def parse_message(self, message_str: str) -> Optional[WebSocketMessage]:
        """
        메시지 문자열을 파싱하여 WebSocketMessage 객체로 변환

        Args:
            message_str: JSON 형태의 메시지 문자열

        Returns:
            파싱된 WebSocketMessage 객체 또는 None

        Raises:
            MessageParsingError: strict_mode에서 파싱 실패시
        """
        self.stats['total_messages'] += 1

        try:
            # JSON 파싱
            raw_data = json.loads(message_str)
            return self._parse_raw_data(raw_data)

        except json.JSONDecodeError as e:
            error_msg = f"JSON 파싱 오류: {str(e)}"
            logger.error(f"{error_msg}, 원본 메시지: {message_str[:200]}...")

            self.stats['failed_parses'] += 1
            self.stats['errors'].append({
                'timestamp': datetime.now(),
                'error': error_msg,
                'raw_data': message_str[:500]  # 처음 500자만 저장
            })

            if self.strict_mode:
                raise MessageParsingError(error_msg, message_str, e)
            return None

        except Exception as e:
            error_msg = f"메시지 파싱 중 예외 발생: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            self.stats['failed_parses'] += 1
            self.stats['errors'].append({
                'timestamp': datetime.now(),
                'error': error_msg,
                'raw_data': message_str[:500]
            })

            if self.strict_mode:
                raise MessageParsingError(error_msg, message_str, e)
            return None

    def _parse_raw_data(self, raw_data: Dict[str, Any]) -> Optional[WebSocketMessage]:
        """
        원본 데이터를 파싱하여 WebSocketMessage 객체로 변환

        Args:
            raw_data: 파싱된 JSON 데이터

        Returns:
            WebSocketMessage 객체 또는 None
        """
        try:
            # 메시지 타입 확인
            message_type_str = raw_data.get('type', '').lower()

            if not message_type_str:
                logger.warning("메시지에 type 필드가 없습니다")
                message_type_str = 'unknown'

            # 메시지 타입 매핑
            try:
                message_type = MessageType(message_type_str)
            except ValueError:
                logger.warning(f"알 수 없는 메시지 타입: {message_type_str}")
                message_type = MessageType.STATUS  # 기본값으로 처리

            self.stats['by_type'].get(message_type_str, 0)
            if message_type_str in self.stats['by_type']:
                self.stats['by_type'][message_type_str] += 1
            else:
                self.stats['by_type']['unknown'] += 1

            # 메시지 기본 구조 생성
            message = WebSocketMessage(
                type=message_type,
                content=raw_data.get('content'),
                raw_data=raw_data
            )

            # 타입별 데이터 파싱
            if message_type == MessageType.TICKER:
                message.ticker_data = self._parse_ticker_data(raw_data)
            elif message_type == MessageType.ORDERBOOK:
                message.orderbook_data = self._parse_orderbook_data(raw_data)
            elif message_type == MessageType.TRANSACTION:
                message.transaction_data = self._parse_transaction_data(raw_data)
            elif message_type == MessageType.ERROR:
                message.error_data = self._parse_error_data(raw_data)
            elif message_type == MessageType.STATUS:
                message.status_data = self._parse_status_data(raw_data)

            self.stats['successful_parses'] += 1
            return message

        except Exception as e:
            error_msg = f"구조화된 데이터 파싱 실패: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")

            self.stats['failed_parses'] += 1
            self.stats['errors'].append({
                'timestamp': datetime.now(),
                'error': error_msg,
                'raw_data': raw_data
            })

            if self.strict_mode:
                raise MessageParsingError(error_msg, raw_data, e)
            return None

    def _parse_ticker_data(self, raw_data: Dict[str, Any]) -> Optional[TickerData]:
        """Ticker 데이터 파싱"""
        try:
            content = raw_data.get('content', {})
            if isinstance(content, dict):
                return TickerData(**content)
            else:
                logger.warning("Ticker content가 dict 형태가 아닙니다")
                return None
        except ValidationError as e:
            logger.warning(f"Ticker 데이터 검증 실패: {e}")
            if not self.strict_mode:
                return None
            raise

    def _parse_orderbook_data(self, raw_data: Dict[str, Any]) -> Optional[OrderBookData]:
        """Orderbook 데이터 파싱"""
        try:
            content = raw_data.get('content', {})
            if isinstance(content, dict):
                return OrderBookData(**content)
            else:
                logger.warning("Orderbook content가 dict 형태가 아닙니다")
                return None
        except ValidationError as e:
            logger.warning(f"Orderbook 데이터 검증 실패: {e}")
            if not self.strict_mode:
                return None
            raise

    def _parse_transaction_data(self, raw_data: Dict[str, Any]) -> Optional[List[TransactionData]]:
        """Transaction 데이터 파싱"""
        try:
            content = raw_data.get('content', {})
            transaction_list = content.get('list', [])

            if not isinstance(transaction_list, list):
                logger.warning("Transaction list가 리스트 형태가 아닙니다")
                return []

            parsed_transactions = []
            for transaction in transaction_list:
                try:
                    parsed_transactions.append(TransactionData(**transaction))
                except ValidationError as e:
                    logger.warning(f"Transaction 데이터 검증 실패: {e}")
                    if self.strict_mode:
                        raise

            return parsed_transactions

        except Exception as e:
            logger.warning(f"Transaction 데이터 파싱 실패: {e}")
            if not self.strict_mode:
                return []
            raise

    def _parse_error_data(self, raw_data: Dict[str, Any]) -> Optional[ErrorData]:
        """Error 데이터 파싱"""
        try:
            return ErrorData(**raw_data)
        except ValidationError as e:
            logger.warning(f"Error 데이터 검증 실패: {e}")
            if not self.strict_mode:
                return None
            raise

    def _parse_status_data(self, raw_data: Dict[str, Any]) -> Optional[StatusData]:
        """Status 데이터 파싱"""
        try:
            return StatusData(**raw_data)
        except ValidationError as e:
            logger.warning(f"Status 데이터 검증 실패: {e}")
            if not self.strict_mode:
                return None
            raise

    def get_stats(self) -> Dict[str, Any]:
        """파싱 통계 반환"""
        return self.stats.copy()

    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            'total_messages': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'by_type': {
                'ticker': 0,
                'orderbook': 0,
                'transaction': 0,
                'error': 0,
                'status': 0,
                'unknown': 0
            },
            'errors': []
        }


# 전역 파서 인스턴스
_global_parser: Optional[MessageParser] = None


def get_message_parser(strict_mode: bool = False) -> MessageParser:
    """
    전역 메시지 파서 인스턴스 반환

    Args:
        strict_mode: 엄격 모드 설정

    Returns:
        MessageParser 인스턴스
    """
    global _global_parser

    if _global_parser is None:
        _global_parser = MessageParser(strict_mode=strict_mode)

    return _global_parser


def parse_websocket_message(message_str: str, strict_mode: bool = False) -> Optional[WebSocketMessage]:
    """
    WebSocket 메시지 파싱 유틸리티 함수

    Args:
        message_str: JSON 형태의 메시지 문자열
        strict_mode: 엄격 모드

    Returns:
        파싱된 WebSocketMessage 객체 또는 None
    """
    parser = get_message_parser(strict_mode)
    return parser.parse_message(message_str)