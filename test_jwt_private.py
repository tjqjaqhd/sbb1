"""
JWT 개선사항 적용 후 개인 API 테스트
"""

import asyncio
import logging
from src.api.bithumb.client import BithumbHTTPClient
from src.api.bithumb.auth import get_api_key_manager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_private_api():
    """개인 API 테스트"""
    print("개인 API 연동 테스트 시작")

    try:
        # API 키 매니저 초기화
        key_manager = get_api_key_manager()
        print(f"API 키 설정 상태: {key_manager.is_configured}")

        if not key_manager.is_configured:
            print("API 키가 설정되지 않음")
            return False

        # HTTP 클라이언트로 계좌 조회 테스트
        async with BithumbHTTPClient() as client:
            print("계좌 정보 조회 시도...")

            try:
                response = await client.get_accounts()
                print("계좌 조회 성공!")
                print(f"응답: {response}")
                return True

            except Exception as e:
                print(f"계좌 조회 실패: {e}")

                # 잔고 조회로 재시도
                print("잔고 조회로 재시도...")
                try:
                    balance_response = await client.get_balance()
                    print("잔고 조회 성공!")
                    print(f"응답: {balance_response}")
                    return True
                except Exception as e2:
                    print(f"잔고 조회도 실패: {e2}")
                    return False

    except Exception as e:
        print(f"테스트 실패: {e}")
        return False


async def main():
    success = await test_private_api()
    print(f"테스트 결과: {'성공' if success else '실패'}")


if __name__ == "__main__":
    asyncio.run(main())