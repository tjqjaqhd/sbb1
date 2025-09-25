#!/usr/bin/env python3
"""
빗썸 개인 API 간단 테스트
"""

import asyncio
import logging
import traceback

logging.basicConfig(level=logging.INFO)

async def test_private_api():
    try:
        print("Testing Private API...")

        # Import modules individually to catch import errors
        print("1. Importing auth module...")
        from src.api.bithumb.auth import get_api_key_manager

        print("2. Importing client module...")
        from src.api.bithumb.client import get_http_client

        print("3. Getting API key manager...")
        key_manager = get_api_key_manager()

        if not key_manager.is_configured:
            print("API keys not configured")
            return False

        print("4. Getting HTTP client...")
        client = await get_http_client()

        print("5. Generating JWT headers...")
        auth_headers = key_manager.get_auth_headers("/v1/accounts", None)
        print(f"Headers generated: {list(auth_headers.keys())}")

        print("6. Making API request...")
        try:
            response = await client.get("/v1/accounts", headers=auth_headers)
            print(f"Response type: {type(response)}")
            print(f"Response: {str(response)[:200]}...")

        except Exception as api_error:
            print(f"API request error: {type(api_error).__name__}: {str(api_error)}")
            print("Full traceback:")
            traceback.print_exc()

        await client.close()
        return True

    except Exception as e:
        print(f"Test failed: {type(e).__name__}: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_private_api())
    print(f"Test result: {'SUCCESS' if success else 'FAILED'}")