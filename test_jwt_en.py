"""
JWT improvements test for private API
"""

import asyncio
import logging
from src.api.bithumb.client import BithumbHTTPClient
from src.api.bithumb.auth import get_api_key_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_private_api():
    """Test private API with JWT improvements"""
    print("Starting private API test")

    try:
        # Initialize API key manager
        key_manager = get_api_key_manager()
        print(f"API key configured: {key_manager.is_configured}")

        if not key_manager.is_configured:
            print("API key not configured")
            return False

        # Test with HTTP client - direct API calls
        async with BithumbHTTPClient() as client:
            print("Testing private API with JWT token...")

            try:
                # 빗썸 공식 계좌 조회 API 엔드포인트
                endpoint = "/v1/accounts"

                # 키 매니저에서 올바른 JWT 인증 헤더 생성
                headers = key_manager.get_auth_headers(endpoint)
                print(f"JWT auth headers generated successfully")
                print(f"Calling endpoint: {endpoint}")
                print(f"Headers: {headers}")

                response = await client.get(endpoint, headers=headers)
                print("Private API call SUCCESS!")
                print(f"Response: {response}")
                return True

            except Exception as e:
                print(f"Private API call failed: {e}")

                # 다른 개인 API 엔드포인트로 재시도
                print("Retrying with API key list endpoint...")
                try:
                    endpoint2 = "/v1/api-keys"
                    headers2 = key_manager.get_auth_headers(endpoint2)

                    response2 = await client.get(endpoint2, headers=headers2)
                    print("Alternative endpoint SUCCESS!")
                    print(f"Response: {response2}")
                    return True
                except Exception as e2:
                    print(f"Alternative endpoint also failed: {e2}")

                    # 마지막 시도: 다른 형식의 계좌 조회
                    print("Final attempt with orders endpoint...")
                    try:
                        endpoint3 = "/v1/orders"
                        headers3 = key_manager.get_auth_headers(endpoint3)

                        response3 = await client.get(endpoint3, headers=headers3)
                        print("Final endpoint SUCCESS!")
                        print(f"Response: {response3}")
                        return True
                    except Exception as e3:
                        print(f"All endpoints failed: {e3}")
                        return False

    except Exception as e:
        print(f"Test failed: {e}")
        return False


async def main():
    success = await test_private_api()
    print(f"Test result: {'SUCCESS' if success else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())