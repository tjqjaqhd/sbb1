"""
시계열 데이터 파티셔닝 설정 스크립트
"""

import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime, timedelta


def setup_partitioning():
    """market_data 테이블에 시계열 파티셔닝 설정"""
    print("=" * 60)
    print("Time-series Data Partitioning Setup")
    print("=" * 60)

    target_username = 'tjqjaqhd'
    target_password = '*tj1748426'
    target_database = 'bithumb_trading'

    try:
        print("[STEP 1] Connecting to database...")

        # 사용자 계정으로 연결
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user=target_username,
            password=target_password,
            database=target_database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        print("[SUCCESS] Connected to database!")

        print("\n[STEP 2] Dropping existing market_data table...")

        # 기존 market_data 테이블 삭제 (파티셔닝을 위해)
        cursor.execute("DROP TABLE IF EXISTS market_data CASCADE;")
        print("[SUCCESS] Existing market_data table dropped!")

        print("\n[STEP 3] Creating partitioned market_data table...")

        # 파티셔닝된 market_data 테이블 생성
        cursor.execute("""
            CREATE TABLE market_data (
                id BIGSERIAL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                open_price DECIMAL(20, 8) NOT NULL,
                high_price DECIMAL(20, 8) NOT NULL,
                low_price DECIMAL(20, 8) NOT NULL,
                close_price DECIMAL(20, 8) NOT NULL,
                volume DECIMAL(20, 8) NOT NULL,
                quote_volume DECIMAL(20, 8),
                trade_count INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,

                -- 복합 기본 키 (파티션 키 포함)
                PRIMARY KEY (id, timestamp),

                -- 고유 제약조건
                CONSTRAINT uq_market_data_symbol_timeframe_timestamp
                    UNIQUE (symbol, timeframe, timestamp)
            ) PARTITION BY RANGE (timestamp);
        """)
        print("[SUCCESS] Partitioned market_data table created!")

        print("\n[STEP 4] Creating monthly partitions...")

        # 현재 월부터 향후 12개월까지 파티션 생성
        current_date = datetime.now()

        for i in range(-1, 13):  # 지난 달부터 향후 12개월
            # 파티션 시작/종료 날짜 계산
            if i < 0:
                target_date = current_date.replace(day=1) + timedelta(days=i*30)
            else:
                target_date = current_date.replace(day=1) + timedelta(days=i*32)

            year = target_date.year
            month = target_date.month

            # 다음 달 첫날 계산
            if month == 12:
                next_year = year + 1
                next_month = 1
            else:
                next_year = year
                next_month = month + 1

            partition_name = f"market_data_{year}_{month:02d}"
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{next_year}-{next_month:02d}-01"

            cursor.execute(f"""
                CREATE TABLE {partition_name} PARTITION OF market_data
                FOR VALUES FROM ('{start_date}') TO ('{end_date}');
            """)
            print(f"  - Created partition: {partition_name}")

        print("\n[STEP 5] Creating indexes on partitioned table...")

        # 인덱스 생성
        cursor.execute("""
            CREATE INDEX idx_market_data_symbol_timestamp
            ON market_data (symbol, timestamp);
        """)

        cursor.execute("""
            CREATE INDEX idx_market_data_timeframe_timestamp
            ON market_data (timeframe, timestamp);
        """)

        cursor.execute("""
            CREATE INDEX idx_market_data_symbol
            ON market_data (symbol);
        """)

        print("[SUCCESS] Indexes created on partitioned table!")

        print("\n[STEP 6] Testing partitioned table...")

        # 테스트 데이터 삽입
        test_timestamp = current_date.strftime('%Y-%m-%d %H:%M:%S+09')
        cursor.execute("""
            INSERT INTO market_data (
                symbol, timeframe, timestamp,
                open_price, high_price, low_price, close_price, volume
            ) VALUES (
                'BTC_KRW', '1h', %s,
                50000000, 50100000, 49900000, 50050000, 1.5
            );
        """, (test_timestamp,))

        # 데이터 조회
        cursor.execute("""
            SELECT COUNT(*) FROM market_data
            WHERE symbol = 'BTC_KRW';
        """)
        count = cursor.fetchone()[0]
        print(f"[SUCCESS] Test data inserted and retrieved! Count: {count}")

        # 파티션 정보 확인
        cursor.execute("""
            SELECT schemaname, tablename, tableowner
            FROM pg_tables
            WHERE tablename LIKE 'market_data_%'
            ORDER BY tablename;
        """)

        partitions = cursor.fetchall()
        print(f"\n[INFO] Created partitions ({len(partitions)}):")
        for partition in partitions:
            print(f"  - {partition[1]}")

        cursor.close()
        conn.close()

        print("\n" + "=" * 60)
        print("[SUCCESS] Time-series partitioning setup completed!")
        print("market_data table is now partitioned by timestamp (monthly)")
        print("=" * 60)
        return True

    except psycopg2.Error as e:
        print(f"[ERROR] PostgreSQL Error: {e}")
        return False

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False


if __name__ == "__main__":
    result = setup_partitioning()
    sys.exit(0 if result else 1)