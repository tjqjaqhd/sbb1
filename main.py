#!/usr/bin/env python3
"""
빗썸 자동매매 메인 애플리케이션

실행 방법:
    python main.py
"""

import asyncio
import logging
from src import __version__

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """메인 애플리케이션 함수"""
    logger.info(f"빗썸 자동매매 시스템 v{__version__} 시작")

    # TODO: 시스템 구성 요소들을 여기에 추가
    print("🚀 빗썸 자동매매 시스템이 시작되었습니다!")
    print(f"📊 버전: {__version__}")

    # 개발 중... 추후 실제 구현 코드 추가 예정


if __name__ == "__main__":
    asyncio.run(main())