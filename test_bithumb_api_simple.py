"""
간단한 빗썸 API 연결 테스트
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트 경로를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.bithumb.client import get_http_client, close_http_client


async def test_bithumb_http_api():
    """빗썸 HTTP API 연결 테스트"""
    print("=" * 60)
    print("Bithumb HTTP API Connection Test")
    print("=" * 60)

    try:
        # HTTP 클라이언트 가져오기
        print("\n[STEP 1] Getting HTTP client...")
        client = await get_http_client()
        print("[SUCCESS] HTTP client ready!")

        # API 상태 확인
        print("\n[STEP 2] Checking API health...")
        health = await client.health_check()
        if health:
            print("[SUCCESS] Bithumb API connection healthy!")
        else:
            print("[WARNING] Bithumb API health check failed!")
            return False

        # BTC 티커 조회 테스트
        print("\n[STEP 3] Testing BTC ticker query...")
        ticker_data = await client.get("/public/ticker/BTC_KRW")

        if ticker_data and "data" in ticker_data:
            closing_price = ticker_data["data"].get("closing_price", "N/A")
            opening_price = ticker_data["data"].get("opening_price", "N/A")
            max_price = ticker_data["data"].get("max_price", "N/A")
            min_price = ticker_data["data"].get("min_price", "N/A")

            print(f"[SUCCESS] BTC Price Data Retrieved:")
            print(f"  - Current Price: {closing_price}")
            print(f"  - Opening Price: {opening_price}")
            print(f"  - High: {max_price}")
            print(f"  - Low: {min_price}")
        else:
            print(f"[WARNING] Unexpected response format: {ticker_data}")

        # ETH 티커 조회 테스트
        print("\n[STEP 4] Testing ETH ticker query...")
        eth_data = await client.get("/public/ticker/ETH_KRW")

        if eth_data and "data" in eth_data:
            eth_price = eth_data["data"].get("closing_price", "N/A")
            print(f"[SUCCESS] ETH Current Price: {eth_price}")
        else:
            print(f"[WARNING] ETH ticker query failed")

        # Rate Limit 상태 확인
        print("\n[STEP 5] Checking rate limit status...")
        rate_status = client.get_rate_limit_status()
        print(f"[INFO] Rate limit categories: {len(rate_status)}")

        for category, status in rate_status.items():
            if status.get('available', 0) > 0:
                print(f"  - {category}: {status.get('available', 0)} requests available")

        print("\n" + "=" * 60)
        print("[SUCCESS] All HTTP API tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[ERROR] API test failed: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

    finally:
        # 클라이언트 정리
        await close_http_client()
        print("\n[INFO] HTTP client closed.")


if __name__ == "__main__":
    result = asyncio.run(test_bithumb_http_api())
    sys.exit(0 if result else 1)