# 빗썸 자동매매 시스템 데이터베이스 스키마 설계

## ERD 개요

자동매매 시스템을 위한 PostgreSQL 데이터베이스 스키마 설계입니다.
시계열 데이터 처리, 실시간 거래 데이터 저장, 매매 신호 관리를 위한 테이블들을 정의합니다.

## 테이블 구조

### 1. market_data (시장 데이터 - OHLCV)
시계열 캔들 데이터를 저장하는 메인 테이블입니다.

```sql
CREATE TABLE market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,           -- 거래 종목 (BTC-KRW, ETH-KRW 등)
    timeframe VARCHAR(10) NOT NULL,        -- 시간 간격 (1m, 5m, 15m, 1h, 1d 등)
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,  -- 캔들 시간

    -- OHLCV 데이터
    open_price DECIMAL(20, 8) NOT NULL,    -- 시가
    high_price DECIMAL(20, 8) NOT NULL,    -- 고가
    low_price DECIMAL(20, 8) NOT NULL,     -- 저가
    close_price DECIMAL(20, 8) NOT NULL,   -- 종가
    volume DECIMAL(20, 8) NOT NULL,        -- 거래량

    -- 부가 정보
    quote_volume DECIMAL(20, 8),           -- 거래 금액
    trade_count INTEGER,                   -- 거래 건수

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(symbol, timeframe, timestamp)
) PARTITION BY RANGE (timestamp);
```

### 2. tickers (실시간 시세)
WebSocket으로 수신한 실시간 시세 데이터를 저장합니다.

```sql
CREATE TABLE tickers (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,

    -- 가격 정보
    opening_price DECIMAL(20, 8),
    closing_price DECIMAL(20, 8),
    min_price DECIMAL(20, 8),
    max_price DECIMAL(20, 8),

    -- 거래량 정보
    volume_24h DECIMAL(20, 8),
    volume_7d DECIMAL(20, 8),

    -- 호가 정보
    best_bid DECIMAL(20, 8),
    best_ask DECIMAL(20, 8),

    -- 변동률 정보
    price_change_24h DECIMAL(20, 8),
    price_change_rate_24h DECIMAL(10, 4),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(symbol, timestamp)
);
```

### 3. orderbooks (호가 데이터)
실시간 호가창 데이터를 저장합니다.

```sql
CREATE TABLE orderbooks (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,

    -- 매수 호가 (상위 10개)
    bid_prices DECIMAL(20, 8)[],
    bid_quantities DECIMAL(20, 8)[],

    -- 매도 호가 (상위 10개)
    ask_prices DECIMAL(20, 8)[],
    ask_quantities DECIMAL(20, 8)[],

    -- 총 호가량
    total_bid_quantity DECIMAL(20, 8),
    total_ask_quantity DECIMAL(20, 8),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(symbol, timestamp)
);
```

### 4. transactions (체결 데이터)
실시간 거래 체결 정보를 저장합니다.

```sql
CREATE TABLE transactions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,

    -- 체결 정보
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    total DECIMAL(20, 8) NOT NULL,

    -- 거래 타입
    transaction_type VARCHAR(10) NOT NULL, -- BUY, SELL

    -- 빗썸 원본 데이터
    transaction_id VARCHAR(50),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_transactions_symbol_timestamp (symbol, timestamp),
    INDEX idx_transactions_timestamp (timestamp)
);
```

### 5. orders (주문 정보)
사용자가 실행한 주문 정보를 저장합니다.

```sql
CREATE TABLE orders (
    id BIGSERIAL PRIMARY KEY,

    -- 주문 식별
    order_id VARCHAR(50) UNIQUE NOT NULL,  -- 빗썸 주문 ID
    client_order_id VARCHAR(100),          -- 클라이언트 주문 ID

    -- 주문 기본 정보
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,             -- BUY, SELL
    order_type VARCHAR(20) NOT NULL,       -- LIMIT, MARKET

    -- 가격/수량 정보
    price DECIMAL(20, 8),                  -- 주문 가격
    quantity DECIMAL(20, 8) NOT NULL,      -- 주문 수량
    filled_quantity DECIMAL(20, 8) DEFAULT 0, -- 체결된 수량
    remaining_quantity DECIMAL(20, 8),     -- 미체결 수량

    -- 상태 정보
    status VARCHAR(20) NOT NULL,           -- PENDING, FILLED, PARTIAL, CANCELLED

    -- 시간 정보
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,

    -- 추가 정보
    commission DECIMAL(20, 8),             -- 수수료
    strategy_id INTEGER,                   -- 연관된 전략 ID

    INDEX idx_orders_symbol_status (symbol, status),
    INDEX idx_orders_created_at (created_at),
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);
```

### 6. balances (잔고 정보)
사용자 계정의 잔고 정보를 저장합니다.

```sql
CREATE TABLE balances (
    id BIGSERIAL PRIMARY KEY,

    -- 자산 정보
    currency VARCHAR(20) NOT NULL,         -- KRW, BTC, ETH 등
    available DECIMAL(20, 8) NOT NULL,     -- 사용 가능한 잔고
    locked DECIMAL(20, 8) DEFAULT 0,       -- 주문 중인 잔고
    total DECIMAL(20, 8) GENERATED ALWAYS AS (available + locked) STORED,

    -- 시간 정보
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(currency)
);
```

### 7. trading_signals (매매 신호)
기술적 분석을 통해 생성된 매매 신호를 저장합니다.

```sql
CREATE TABLE trading_signals (
    id BIGSERIAL PRIMARY KEY,

    -- 신호 기본 정보
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20) NOT NULL,      -- BUY, SELL, HOLD
    strategy_name VARCHAR(100) NOT NULL,   -- 전략명

    -- 신호 상세 정보
    confidence_score DECIMAL(5, 4),        -- 신뢰도 점수 (0-1)
    price DECIMAL(20, 8) NOT NULL,         -- 신호 발생 시점 가격
    target_price DECIMAL(20, 8),           -- 목표 가격
    stop_price DECIMAL(20, 8),             -- 손절 가격

    -- 기술적 지표 값들
    rsi_value DECIMAL(5, 2),
    macd_value DECIMAL(10, 6),
    bollinger_position DECIMAL(5, 4),
    volume_ratio DECIMAL(10, 4),

    -- 상태 및 시간
    status VARCHAR(20) DEFAULT 'ACTIVE',   -- ACTIVE, EXECUTED, EXPIRED
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    executed_at TIMESTAMP WITH TIME ZONE,

    INDEX idx_signals_symbol_status (symbol, status),
    INDEX idx_signals_created_at (created_at)
);
```

### 8. strategies (거래 전략)
사용자가 설정한 거래 전략 정보를 저장합니다.

```sql
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,

    -- 전략 기본 정보
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    is_active BOOLEAN DEFAULT true,

    -- 전략 설정
    symbols TEXT[],                        -- 대상 종목들
    timeframes VARCHAR(10)[],              -- 사용할 시간 간격들

    -- 리스크 관리
    max_position_size DECIMAL(5, 4),       -- 최대 포지션 크기 (%)
    stop_loss_pct DECIMAL(5, 4),           -- 손절 비율
    take_profit_pct DECIMAL(5, 4),         -- 익절 비율

    -- 기술적 지표 파라미터 (JSON)
    technical_params JSONB,

    -- 시간 정보
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_run_at TIMESTAMP WITH TIME ZONE
);
```

### 9. user_settings (사용자 설정)
시스템 전반의 사용자 설정을 저장합니다.

```sql
CREATE TABLE user_settings (
    id SERIAL PRIMARY KEY,

    -- 설정 식별
    category VARCHAR(50) NOT NULL,         -- API, TRADING, NOTIFICATION 등
    key VARCHAR(100) NOT NULL,
    value TEXT,

    -- 설정 메타데이터
    description TEXT,
    is_encrypted BOOLEAN DEFAULT false,     -- 암호화 필요 여부

    -- 시간 정보
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(category, key)
);
```

### 10. trade_history (거래 내역)
실제 거래 실행 결과를 저장하는 로그 테이블입니다.

```sql
CREATE TABLE trade_history (
    id BIGSERIAL PRIMARY KEY,

    -- 거래 기본 정보
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,             -- BUY, SELL

    -- 가격/수량 정보
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    total DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 8) NOT NULL,

    -- 연관 정보
    order_id VARCHAR(50) NOT NULL,
    signal_id INTEGER,                     -- 매매 신호 ID
    strategy_id INTEGER,                   -- 전략 ID

    -- 손익 정보
    pnl DECIMAL(20, 8),                    -- 실현 손익
    pnl_pct DECIMAL(10, 6),                -- 손익률

    -- 시간 정보
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_trade_history_symbol_executed (symbol, executed_at),
    INDEX idx_trade_history_strategy (strategy_id),
    FOREIGN KEY (signal_id) REFERENCES trading_signals(id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);
```

## 인덱스 전략

### 1. 시계열 데이터 최적화
- market_data: (symbol, timeframe, timestamp) 복합 인덱스
- tickers: (symbol, timestamp) 복합 인덱스
- 파티셔닝: market_data 테이블을 월별로 파티셔닝

### 2. 조회 성능 최적화
- 거래 관련: (symbol, status) 복합 인덱스
- 시간 기반: timestamp 단일 인덱스
- 전략별: strategy_id 인덱스

## 파티셔닝 전략

### market_data 테이블 파티셔닝
```sql
-- 월별 파티셔닝 예시
CREATE TABLE market_data_202501 PARTITION OF market_data
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE market_data_202502 PARTITION OF market_data
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
```

## 제약 조건 및 트리거

### 1. 데이터 무결성
- 외래키 제약 조건
- CHECK 제약 조건 (가격 > 0, 수량 > 0 등)
- UNIQUE 제약 조건

### 2. 자동 업데이트
- updated_at 자동 갱신 트리거
- 잔고 자동 계산 트리거