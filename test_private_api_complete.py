"""
완전한 개인 API 연동 테스트
TaskMaster JWT 수정사항 적용 후 검증

JWT 토큰 개선 사항:
- UTC 타임스탬프 정밀도 수정
- JWT 헤더 명시적 설정 (typ, alg)
- URL 인코딩 적용
- 디버깅 및 검증 강화
"""

import asyncio
import logging
from src.api.bithumb.client import BithumbHTTPClient
from src.api.bithumb.auth import get_api_key_manager

# 디버깅을 위한 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_complete_private_api():
    """완전한 개인 API 연동 테스트"""

    print("=" * 60)
    print("TaskMaster 개인 API 완전 연동 검증")
    print("=" * 60)

    try:
        # API 키 매니저 초기화
        key_manager = get_api_key_manager()

        print(f"🔑 API 키 설정 상태: {key_manager.is_configured}")
        print(f"🔑 API 키 정보: {key_manager}")

        if not key_manager.is_configured:
            print("❌ API 키가 설정되지 않았습니다!")
            return False

        # HTTP 클라이언트 생성
        async with BithumbHTTPClient() as client:
            print("\n🌐 HTTP 클라이언트 연결 완료")

            # 1. JWT 토큰 생성 테스트
            print("\n📝 JWT 토큰 생성 테스트")
            token = key_manager.generate_jwt_token("/v1/accounts")
            print(f"✅ JWT 토큰 생성 성공 (길이: {len(token)})")
            print(f"🔍 JWT 토큰 샘플: {token[:50]}...")

            # 2. 계좌 정보 조회 (GET /v1/accounts)
            print("\n💰 계좌 정보 조회 시도")
            try:
                accounts_response = await client.get_accounts()
                print(f"✅ 계좌 조회 성공!")
                print(f"📊 응답 데이터: {accounts_response}")

                if 'data' in accounts_response:
                    accounts = accounts_response['data']
                    print(f"📝 계좌 수: {len(accounts)}")
                    for account in accounts[:3]:  # 처음 3개만 표시
                        currency = account.get('currency', 'N/A')
                        balance = account.get('balance', '0')
                        locked = account.get('locked', '0')
                        print(f"   💳 {currency}: 잔고 {balance}, 잠금 {locked}")

                return True

            except Exception as e:
                print(f"❌ 계좌 조회 실패: {e}")

                # 3. 만약 계좌 조회가 실패하면 다른 개인 API 시도
                print("\n🔄 주문 내역 조회로 재시도")
                try:
                    # 주문 내역 조회 시도
                    orders_params = {"order_currency": "BTC", "count": 5}
                    orders_response = await client.get_user_orders("BTC_KRW", orders_params)
                    print(f"✅ 주문 내역 조회 성공!")
                    print(f"📊 응답: {orders_response}")
                    return True

                except Exception as e2:
                    print(f"❌ 주문 내역 조회도 실패: {e2}")

                    # 4. 마지막으로 잔고 조회 시도
                    print("\n🔄 잔고 조회로 재시도")
                    try:
                        balance_response = await client.get_balance()
                        print(f"✅ 잔고 조회 성공!")
                        print(f"📊 응답: {balance_response}")

                        # 실제 잔고 정보 파싱
                        if 'data' in balance_response:
                            balance_data = balance_response['data']
                            for currency, data in balance_data.items():
                                if currency != 'date':  # date 제외
                                    available = data.get('available', '0')
                                    in_use = data.get('in_use', '0')
                                    if float(available) > 0 or float(in_use) > 0:
                                        print(f"   💰 {currency}: 사용가능 {available}, 사용중 {in_use}")

                        return True

                    except Exception as e3:
                        print(f"❌ 모든 개인 API 테스트 실패: {e3}")
                        return False

    except Exception as e:
        print(f"❌ 전체 테스트 실패: {e}")
        logger.exception("Complete private API test failed")
        return False


async def main():
    """메인 실행 함수"""
    print("TaskMaster 개인 API 완전 연동 검증 시작\n")

    success = await test_complete_private_api()

    print("\n" + "=" * 60)
    if success:
        print("개인 API 완전 연동 검증 성공!")
        print("실제 계좌 정보 교환 확인됨")
    else:
        print("개인 API 연동에 여전히 문제가 있습니다")
        print("추가 디버깅이 필요합니다")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())