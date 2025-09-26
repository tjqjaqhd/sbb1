"""
JWT 토큰 디버깅 - 토큰 내용 분석
"""

import jwt
from src.api.bithumb.auth import get_api_key_manager

def debug_jwt_token():
    """JWT 토큰 내용 분석"""
    print("JWT Token Debugging")

    try:
        # API 키 매니저 생성
        key_manager = get_api_key_manager()

        if not key_manager.is_configured:
            print("API key not configured")
            return

        print(f"API Key Manager: {key_manager}")

        # JWT 토큰 생성
        endpoint = "/v1/accounts"
        jwt_token = key_manager.generate_jwt_token(endpoint)

        # 빈 문자열의 SHA512 해시 확인
        empty_hash = key_manager.generate_query_hash({})
        print(f"Empty string SHA512 hash: {empty_hash}")

        # 직접 빈 문자열 해시 계산
        import hashlib
        direct_hash = hashlib.sha512("".encode('utf-8')).hexdigest()
        print(f"Direct empty string SHA512: {direct_hash}")

        print(f"Generated JWT Token Length: {len(jwt_token)}")
        print(f"JWT Token: {jwt_token[:100]}...")

        # JWT 토큰 디코딩 (서명 검증 없이)
        try:
            decoded = jwt.decode(jwt_token, options={"verify_signature": False})
            print("Decoded JWT Payload:")
            for key, value in decoded.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"JWT decode error: {e}")

        # 헤더도 확인
        try:
            header = jwt.get_unverified_header(jwt_token)
            print("JWT Header:")
            for key, value in header.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"JWT header decode error: {e}")

    except Exception as e:
        print(f"Debug failed: {e}")

if __name__ == "__main__":
    debug_jwt_token()