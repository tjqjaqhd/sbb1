"""
빗썸 API 키 관리 시스템

API 키와 시크릿 키의 안전한 로드, 검증, 관리를 담당합니다.
"""

import os
import base64
import logging
import hashlib
import json
import time
import uuid
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt

from config.config import get_settings

logger = logging.getLogger(__name__)


class BithumbAPIKeyError(Exception):
    """빗썸 API 키 관련 오류"""
    pass


class BithumbAPIKeyManager:
    """
    빗썸 API 키 관리자

    환경변수 또는 암호화된 파일에서 API 키를 로드하고 관리합니다.
    """

    def __init__(self):
        """API 키 관리자 초기화"""
        self.settings = get_settings()
        self._api_key: Optional[str] = None
        self._secret_key: Optional[str] = None
        self._keys_loaded = False

    @property
    def is_configured(self) -> bool:
        """API 키가 설정되었는지 확인"""
        self._ensure_keys_loaded()
        return self._api_key is not None and self._secret_key is not None

    @property
    def api_key(self) -> str:
        """API 키 반환"""
        self._ensure_keys_loaded()
        if not self._api_key:
            raise BithumbAPIKeyError("빗썸 API 키가 설정되지 않았습니다")
        return self._api_key

    @property
    def secret_key(self) -> str:
        """시크릿 키 반환"""
        self._ensure_keys_loaded()
        if not self._secret_key:
            raise BithumbAPIKeyError("빗썸 시크릿 키가 설정되지 않았습니다")
        return self._secret_key

    def _ensure_keys_loaded(self) -> None:
        """키가 로드되었는지 확인하고, 없으면 로드"""
        if not self._keys_loaded:
            self._load_keys()

    def _load_keys(self) -> None:
        """환경변수 또는 암호화 파일에서 API 키 로드"""
        try:
            # 1. 환경변수에서 로드 시도
            self._load_from_environment()

            # 2. 환경변수에 없으면 암호화 파일에서 로드 시도
            if not self.is_keys_present():
                self._load_from_encrypted_file()

            self._keys_loaded = True

            if self.is_keys_present():
                logger.info("빗썸 API 키 로드 완료")
                self._validate_keys()
            else:
                logger.warning("빗썸 API 키가 설정되지 않았습니다. 공개 API만 사용 가능합니다.")

        except Exception as e:
            logger.error(f"API 키 로드 중 오류: {str(e)}")
            raise BithumbAPIKeyError(f"API 키 로드 실패: {str(e)}")

    def _load_from_environment(self) -> None:
        """환경변수에서 API 키 로드"""
        self._api_key = self.settings.BITHUMB_API_KEY
        self._secret_key = self.settings.BITHUMB_SECRET_KEY

        if self._api_key and self._secret_key:
            logger.debug("환경변수에서 API 키 로드 완료")

    def _load_from_encrypted_file(self) -> None:
        """암호화된 파일에서 API 키 로드"""
        encrypted_file_path = Path.home() / ".bithumb" / "credentials.enc"

        if not encrypted_file_path.exists():
            logger.debug("암호화된 키 파일을 찾을 수 없습니다")
            return

        try:
            password = os.environ.get("BITHUMB_KEYFILE_PASSWORD")
            if not password:
                logger.warning("암호화된 키 파일 패스워드가 설정되지 않았습니다")
                return

            # 키 파일 복호화
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()

            # 패스워드 기반 키 파생
            salt = encrypted_data[:16]  # 첫 16바이트는 salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

            # 복호화
            fernet = Fernet(key)
            decrypted_data = fernet.decrypt(encrypted_data[16:])

            # JSON 파싱
            import json
            credentials = json.loads(decrypted_data.decode())

            self._api_key = credentials.get('api_key')
            self._secret_key = credentials.get('secret_key')

            if self._api_key and self._secret_key:
                logger.debug("암호화된 파일에서 API 키 로드 완료")

        except Exception as e:
            logger.error(f"암호화된 키 파일 로드 실패: {str(e)}")

    def is_keys_present(self) -> bool:
        """API 키가 존재하는지 확인 (로드 여부와 무관)"""
        return bool(self._api_key and self._secret_key)

    def _validate_keys(self) -> None:
        """API 키 형식 검증"""
        if not self.is_keys_present():
            return

        # API 키 기본 형식 검증
        if not self._api_key or len(self._api_key) < 10:
            raise BithumbAPIKeyError("API 키 형식이 올바르지 않습니다")

        if not self._secret_key or len(self._secret_key) < 10:
            raise BithumbAPIKeyError("시크릿 키 형식이 올바르지 않습니다")

        logger.debug("API 키 형식 검증 완료")

    def save_encrypted_keys(self, api_key: str, secret_key: str, password: str) -> None:
        """API 키를 암호화하여 파일에 저장"""
        if not api_key or not secret_key:
            raise BithumbAPIKeyError("유효한 API 키와 시크릿 키가 필요합니다")

        if not password or len(password) < 8:
            raise BithumbAPIKeyError("패스워드는 8자 이상이어야 합니다")

        # 저장 디렉토리 생성
        key_dir = Path.home() / ".bithumb"
        key_dir.mkdir(exist_ok=True, mode=0o700)

        # 키 파일 경로
        key_file_path = key_dir / "credentials.enc"

        try:
            # Salt 생성
            salt = os.urandom(16)

            # 패스워드 기반 키 파생
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

            # 데이터 준비
            import json
            data = {
                'api_key': api_key,
                'secret_key': secret_key,
                'timestamp': str(int(os.times().elapsed))
            }

            # 암호화
            fernet = Fernet(key)
            encrypted_data = fernet.encrypt(json.dumps(data).encode())

            # 파일 저장 (salt + 암호화된 데이터)
            with open(key_file_path, 'wb') as f:
                f.write(salt + encrypted_data)

            # 파일 권한 설정 (소유자만 읽기/쓰기)
            key_file_path.chmod(0o600)

            logger.info(f"API 키가 암호화되어 {key_file_path}에 저장되었습니다")

        except Exception as e:
            logger.error(f"API 키 암호화 저장 실패: {str(e)}")
            raise BithumbAPIKeyError(f"키 저장 실패: {str(e)}")

    def clear_cached_keys(self) -> None:
        """캐시된 키 정보 클리어"""
        self._api_key = None
        self._secret_key = None
        self._keys_loaded = False
        logger.debug("API 키 캐시 클리어 완료")

    def generate_query_hash(self, params: Dict[str, Any]) -> str:
        """
        요청 파라미터의 SHA512 해시 생성

        Args:
            params: 요청 파라미터

        Returns:
            SHA512 해시 문자열
        """
        if not params:
            return ""

        # 파라미터를 정렬하고 쿼리 문자열로 변환
        sorted_params = sorted(params.items())
        query_string = "&".join(f"{k}={v}" for k, v in sorted_params)

        # SHA512 해시 생성
        return hashlib.sha512(query_string.encode('utf-8')).hexdigest()

    def generate_jwt_token(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        빗썸 API JWT 토큰 생성

        Args:
            endpoint: API 엔드포인트
            params: 요청 파라미터

        Returns:
            JWT 토큰 문자열
        """
        if not self.is_configured:
            raise BithumbAPIKeyError("API 키가 설정되지 않았습니다")

        # JWT 페이로드 구성
        payload = {
            "access_key": self.api_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000)  # 밀리초 단위
        }

        # 파라미터가 있는 경우 쿼리 해시 추가
        if params:
            payload["query_hash"] = self.generate_query_hash(params)
            payload["query_hash_alg"] = "SHA512"

        try:
            # HMAC-SHA256으로 JWT 토큰 생성
            token = jwt.encode(
                payload,
                self.secret_key,
                algorithm="HS256"
            )

            logger.debug(f"JWT 토큰 생성 완료: nonce={payload['nonce']}")
            return token

        except Exception as e:
            logger.error(f"JWT 토큰 생성 실패: {str(e)}")
            raise BithumbAPIKeyError(f"JWT 토큰 생성 실패: {str(e)}")

    def get_auth_headers(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        빗썸 API JWT 기반 인증 헤더 생성

        Args:
            endpoint: API 엔드포인트
            params: 요청 파라미터

        Returns:
            인증 헤더 딕셔너리
        """
        if not self.is_configured:
            raise BithumbAPIKeyError("API 키가 설정되지 않았습니다")

        # JWT 토큰 생성
        jwt_token = self.generate_jwt_token(endpoint, params)

        # JWT 기반 인증 헤더
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json"
        }

        logger.debug(f"JWT 인증 헤더 생성 완료: {endpoint}")
        return headers

    def test_connection(self) -> bool:
        """
        API 키 연결 테스트 (실제 API 호출 없이 기본 검증만)

        Returns:
            키 설정 상태
        """
        try:
            self._ensure_keys_loaded()

            if not self.is_configured:
                logger.warning("API 키가 설정되지 않아 개인 API 사용 불가")
                return False

            # 기본 검증만 수행 (실제 API 테스트는 Task 2.3 이후)
            self._validate_keys()
            logger.info("API 키 기본 검증 완료")
            return True

        except Exception as e:
            logger.error(f"API 키 테스트 실패: {str(e)}")
            return False

    def __str__(self) -> str:
        """문자열 표현 (민감한 정보 숨김)"""
        if self.is_configured:
            masked_api_key = f"{self.api_key[:8]}***{self.api_key[-4:]}" if self.api_key else "None"
            return f"BithumbAPIKeyManager(api_key={masked_api_key}, configured=True)"
        else:
            return "BithumbAPIKeyManager(configured=False)"


# 전역 API 키 매니저 인스턴스
_global_key_manager: Optional[BithumbAPIKeyManager] = None


def get_api_key_manager() -> BithumbAPIKeyManager:
    """
    전역 API 키 매니저 인스턴스 반환

    Returns:
        BithumbAPIKeyManager 인스턴스
    """
    global _global_key_manager

    if _global_key_manager is None:
        _global_key_manager = BithumbAPIKeyManager()

    return _global_key_manager


def reset_api_key_manager() -> None:
    """전역 API 키 매니저 리셋"""
    global _global_key_manager

    if _global_key_manager:
        _global_key_manager.clear_cached_keys()

    _global_key_manager = None