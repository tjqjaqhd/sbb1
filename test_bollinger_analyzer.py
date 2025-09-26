"""
볼린저 밴드 분석기 테스트 스크립트

BollingerAnalyzer 클래스의 주요 기능을 테스트합니다.
- 볼린저 밴드 계산
- 밴드 폭 분석
- 스퀴즈 패턴 감지
- 돌파 확률 계산
- 종합 분석
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.services.bollinger_analyzer import BollingerAnalyzer, get_bollinger_analyzer, close_bollinger_analyzer
from src.database.config import DatabaseConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_bollinger_analyzer.log')
    ]
)

logger = logging.getLogger(__name__)


def print_separator(title: str):
    """구분선 출력"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_bb_status(bb_status: dict):
    """볼린저 밴드 상태 정보 출력"""
    if not bb_status:
        print("볼린저 밴드 상태 정보 없음")
        return

    print(f"현재 가격: {bb_status['current_close']:,.2f}")
    print(f"상단 밴드: {bb_status['current_upper']:,.2f}")
    print(f"중간 밴드: {bb_status['current_middle']:,.2f}")
    print(f"하단 밴드: {bb_status['current_lower']:,.2f}")
    print(f"밴드 폭: {bb_status['current_band_width']:.2f}%")
    print(f"밴드 내 위치: {bb_status['band_position_percent']:.1f}%")
    print(f"밴드 폭 수준: {bb_status['band_width_level']}")
    print(f"가격 위치: {bb_status['price_position']}")


def print_squeeze_info(squeeze_info: dict):
    """스퀴즈 정보 출력"""
    if not squeeze_info:
        print("스퀴즈 정보 없음")
        return

    print(f"스퀴즈 감지: {'예' if squeeze_info['squeeze_detected'] else '아니오'}")
    print(f"스퀴즈 수준: {squeeze_info['squeeze_level']}")
    print(f"스퀴즈 강도: {squeeze_info['squeeze_strength']:.1f}/10")
    print(f"지속 기간: {squeeze_info['squeeze_duration']}일")
    print(f"설명: {squeeze_info['description']}")


def print_breakout_probability(breakout_prob: dict):
    """돌파 확률 정보 출력"""
    if not breakout_prob:
        print("돌파 확률 정보 없음")
        return

    print(f"돌파 확률: {breakout_prob['probability_percent']:.1f}%")
    print(f"예상 방향: {breakout_prob['expected_direction']}")
    print(f"방향 신뢰도: {breakout_prob['direction_confidence']:.1f}%")
    print(f"위험 수준: {breakout_prob['risk_level']}")
    print(f"권장 전략: {breakout_prob['recommended_strategy']}")
    print(f"고확률 돌파: {'예' if breakout_prob['is_high_probability'] else '아니오'}")

    # 점수 세부사항
    breakdown = breakout_prob.get('score_breakdown', {})
    if breakdown:
        print("\n점수 세부사항:")
        print(f"  밴드 폭 점수: {breakdown.get('band_width_score', 0):.2f}")
        print(f"  위치 점수: {breakdown.get('position_score', 0):.2f}")
        print(f"  지속 기간 점수: {breakdown.get('duration_score', 0):.2f}")
        print(f"  거래량 점수: {breakdown.get('volume_score', 0):.2f}")


def print_trading_signals(signals: dict):
    """트레이딩 신호 출력"""
    if not signals:
        print("트레이딩 신호 없음")
        return

    print(f"주요 신호: {signals['primary_signal']}")
    print(f"신호 강도: {signals['signal_strength']}/10")

    if signals.get('entry_signals'):
        print("진입 신호:")
        for signal in signals['entry_signals']:
            print(f"  - {signal}")

    if signals.get('exit_signals'):
        print("청산 신호:")
        for signal in signals['exit_signals']:
            print(f"  - {signal}")

    if signals.get('risk_warnings'):
        print("위험 경고:")
        for warning in signals['risk_warnings']:
            print(f"  ⚠️ {warning}")


async def test_basic_bb_calculation():
    """기본 볼린저 밴드 계산 테스트"""
    print_separator("기본 볼린저 밴드 계산 테스트")

    try:
        # 데이터베이스 설정
        db_config = DatabaseConfig()

        # 볼린저 밴드 분석기 생성
        analyzer = BollingerAnalyzer(db_config)

        # 테스트용 심볼
        symbol = "BTC_KRW"

        print(f"테스트 심볼: {symbol}")

        # 과거 데이터 수집 테스트
        print("\n1. 과거 데이터 수집 테스트")
        historical_data = await analyzer.fetch_historical_data(symbol)

        if historical_data:
            print(f"✅ 과거 데이터 수집 성공: {len(historical_data)}개 데이터 포인트")
            print(f"   기간: {historical_data[0]['timestamp']} ~ {historical_data[-1]['timestamp']}")
            print(f"   가격 범위: {min(d['close'] for d in historical_data):,.2f} ~ {max(d['close'] for d in historical_data):,.2f}")
        else:
            print("❌ 과거 데이터 수집 실패")
            return

        # 볼린저 밴드 계산 테스트
        print("\n2. 볼린저 밴드 계산 테스트")
        bb_data = analyzer.calculate_bollinger_bands(historical_data)

        if bb_data:
            print("✅ 볼린저 밴드 계산 성공")
            print(f"   기간: {bb_data['period']}일")
            print(f"   표준편차: {bb_data['stddev_multiplier']}")

            # 최신 값 확인
            import numpy as np
            for i in range(len(bb_data['upper']) - 1, -1, -1):
                if not np.isnan(bb_data['upper'][i]):
                    print(f"   최신 상단밴드: {bb_data['upper'][i]:,.2f}")
                    print(f"   최신 중간밴드: {bb_data['middle'][i]:,.2f}")
                    print(f"   최신 하단밴드: {bb_data['lower'][i]:,.2f}")
                    break
        else:
            print("❌ 볼린저 밴드 계산 실패")
            return

        # 밴드 폭 계산 테스트
        print("\n3. 밴드 폭 계산 테스트")
        band_width = analyzer.calculate_band_width(bb_data)

        if band_width is not None:
            import numpy as np
            valid_bw = band_width[~np.isnan(band_width)]
            if len(valid_bw) > 0:
                print("✅ 밴드 폭 계산 성공")
                print(f"   현재 밴드 폭: {valid_bw[-1]:.2f}%")
                print(f"   평균 밴드 폭: {np.mean(valid_bw):.2f}%")
                print(f"   최대 밴드 폭: {np.max(valid_bw):.2f}%")
                print(f"   최소 밴드 폭: {np.min(valid_bw):.2f}%")
            else:
                print("❌ 유효한 밴드 폭 데이터 없음")
        else:
            print("❌ 밴드 폭 계산 실패")

        # 현재 상태 분석 테스트
        print("\n4. 현재 볼린저 밴드 상태 분석 테스트")
        bb_status = analyzer.get_current_bb_status(bb_data, band_width)

        if bb_status:
            print("✅ 볼린저 밴드 상태 분석 성공")
            print_bb_status(bb_status)
        else:
            print("❌ 볼린저 밴드 상태 분석 실패")

        print(f"\n서비스 통계: {analyzer.get_stats()}")

    except Exception as e:
        logger.error(f"기본 볼린저 밴드 계산 테스트 중 오류: {str(e)}")
        print(f"❌ 테스트 실패: {str(e)}")


async def test_squeeze_detection():
    """스퀴즈 패턴 감지 테스트"""
    print_separator("스퀴즈 패턴 감지 테스트")

    try:
        db_config = DatabaseConfig()
        analyzer = BollingerAnalyzer(db_config)

        symbol = "BTC_KRW"
        print(f"테스트 심볼: {symbol}")

        # 데이터 수집 및 볼린저 밴드 계산
        historical_data = await analyzer.fetch_historical_data(symbol, 40)
        if not historical_data:
            print("❌ 과거 데이터 수집 실패")
            return

        bb_data = analyzer.calculate_bollinger_bands(historical_data)
        if not bb_data:
            print("❌ 볼린저 밴드 계산 실패")
            return

        band_width = analyzer.calculate_band_width(bb_data)
        if band_width is None:
            print("❌ 밴드 폭 계산 실패")
            return

        # 스퀴즈 패턴 감지 테스트
        print("\n1. 스퀴즈 패턴 감지")
        squeeze_info = analyzer.detect_squeeze_pattern(band_width)

        if squeeze_info:
            print("✅ 스퀴즈 패턴 감지 완료")
            print_squeeze_info(squeeze_info)
        else:
            print("❌ 스퀴즈 패턴 감지 실패")

        # 다양한 기간으로 테스트
        print("\n2. 다양한 분석 기간 테스트")
        for period in [10, 15, 20]:
            squeeze_info = analyzer.detect_squeeze_pattern(band_width, period)
            if squeeze_info:
                print(f"\n{period}일 분석:")
                print(f"  스퀴즈 감지: {'예' if squeeze_info['squeeze_detected'] else '아니오'}")
                print(f"  수준: {squeeze_info['squeeze_level']}")
                print(f"  강도: {squeeze_info['squeeze_strength']:.1f}")

    except Exception as e:
        logger.error(f"스퀴즈 패턴 감지 테스트 중 오류: {str(e)}")
        print(f"❌ 테스트 실패: {str(e)}")


async def test_breakout_probability():
    """돌파 확률 계산 테스트"""
    print_separator("돌파 확률 계산 테스트")

    try:
        db_config = DatabaseConfig()
        analyzer = BollingerAnalyzer(db_config)

        symbol = "BTC_KRW"
        print(f"테스트 심볼: {symbol}")

        # 필요한 데이터 준비
        historical_data = await analyzer.fetch_historical_data(symbol, 40)
        if not historical_data:
            print("❌ 과거 데이터 수집 실패")
            return

        bb_data = analyzer.calculate_bollinger_bands(historical_data)
        band_width = analyzer.calculate_band_width(bb_data)
        bb_status = analyzer.get_current_bb_status(bb_data, band_width)
        squeeze_info = analyzer.detect_squeeze_pattern(band_width)

        if not all([bb_data, band_width is not None, bb_status, squeeze_info]):
            print("❌ 필요한 데이터 준비 실패")
            return

        # 돌파 확률 계산 테스트
        print("\n1. 기본 돌파 확률 계산")
        breakout_prob = analyzer.calculate_breakout_probability(bb_status, squeeze_info)

        if breakout_prob:
            print("✅ 돌파 확률 계산 완료")
            print_breakout_probability(breakout_prob)
        else:
            print("❌ 돌파 확률 계산 실패")

        # 거래량 데이터 포함 테스트
        print("\n2. 거래량 데이터 포함 테스트")
        try:
            # 가상의 거래량 데이터 생성 (테스트용)
            mock_volume_data = {
                'surge_score': 6.5,
                'volume_ratios': {
                    'vs_7d_avg': 1.8,
                    'vs_30d_avg': 1.5
                }
            }

            breakout_prob_with_volume = analyzer.calculate_breakout_probability(
                bb_status, squeeze_info, mock_volume_data
            )

            if breakout_prob_with_volume:
                print("✅ 거래량 포함 돌파 확률 계산 완료")
                print(f"확률 (기본): {breakout_prob['probability_percent']:.1f}%")
                print(f"확률 (거래량 포함): {breakout_prob_with_volume['probability_percent']:.1f}%")
                print(f"거래량 점수: {breakout_prob_with_volume['score_breakdown']['volume_score']:.2f}")
            else:
                print("❌ 거래량 포함 돌파 확률 계산 실패")

        except Exception as e:
            logger.warning(f"거래량 포함 테스트 중 오류: {str(e)}")

    except Exception as e:
        logger.error(f"돌파 확률 계산 테스트 중 오류: {str(e)}")
        print(f"❌ 테스트 실패: {str(e)}")


async def test_comprehensive_analysis():
    """종합적인 볼린저 밴드 분석 테스트"""
    print_separator("종합적인 볼린저 밴드 분석 테스트")

    try:
        db_config = DatabaseConfig()

        # 글로벌 인스턴스 사용
        analyzer = await get_bollinger_analyzer(db_config)

        # 다양한 심볼 테스트
        symbols = ["BTC_KRW", "ETH_KRW", "XRP_KRW"]

        for symbol in symbols:
            print(f"\n📊 {symbol} 종합 분석")
            print("-" * 40)

            # 종합 분석 실행
            result = await analyzer.comprehensive_bollinger_analysis(
                symbol=symbol,
                include_volume=True  # 거래량 분석 포함
            )

            if result:
                print("✅ 종합 분석 성공")
                print(f"분석 시점: {result['timestamp']}")
                print(f"사용된 데이터: {result['data_points_used']}개")
                print(f"분석 품질: {result['analysis_quality']['grade']} ({result['analysis_quality']['score']:.1f}/10)")

                # 주요 결과 출력
                print("\n📈 볼린저 밴드 상태:")
                print_bb_status(result['bb_status'])

                print("\n🔍 스퀴즈 분석:")
                print_squeeze_info(result['squeeze_info'])

                print("\n🎯 돌파 확률:")
                print_breakout_probability(result['breakout_probability'])

                print("\n📊 트레이딩 신호:")
                print_trading_signals(result['trading_signals'])

                # 통계 정보
                if result.get('bb_statistics'):
                    stats = result['bb_statistics']
                    print(f"\n📈 볼린저 밴드 통계:")
                    print(f"평균 밴드 폭: {stats.get('avg_band_width', 0):.2f}%")
                    print(f"현재 밴드폭 백분위: {stats.get('current_bw_percentile', 0):.0f}%")
                    print(f"스퀴즈 기간: {stats.get('squeeze_periods', 0)}회")
                    print(f"확장 기간: {stats.get('expansion_periods', 0)}회")
            else:
                print(f"❌ {symbol} 종합 분석 실패")

            # 구분선
            if symbol != symbols[-1]:
                print("\n" + "=" * 60)

        # 서비스 상태 확인
        print("\n🔍 서비스 상태 확인")
        health_status = await analyzer.health_check()
        print(f"서비스명: {health_status['service_name']}")
        print(f"HTTP 클라이언트: {'✅' if health_status['http_client_available'] else '❌'}")
        print(f"데이터베이스: {'✅' if health_status['database_connected'] else '❌'}")
        print(f"TA-Lib: {'✅' if health_status['talib_available'] else '❌'}")
        print(f"통계: {health_status['stats']}")

    except Exception as e:
        logger.error(f"종합 분석 테스트 중 오류: {str(e)}")
        print(f"❌ 테스트 실패: {str(e)}")
    finally:
        # 리소스 정리
        await close_bollinger_analyzer()


async def test_edge_cases():
    """엣지 케이스 테스트"""
    print_separator("엣지 케이스 테스트")

    try:
        db_config = DatabaseConfig()
        analyzer = BollingerAnalyzer(db_config)

        print("1. 빈 데이터 테스트")
        result = analyzer.calculate_bollinger_bands([])
        print(f"빈 데이터 결과: {result is None}")

        print("\n2. 부족한 데이터 테스트")
        minimal_data = [
            {'timestamp': '2023-01-01', 'close': 100, 'open': 99, 'high': 101, 'low': 98, 'volume': 1000}
            for _ in range(5)  # 20일 기간보다 적은 데이터
        ]
        result = analyzer.calculate_bollinger_bands(minimal_data)
        print(f"부족한 데이터 결과: {result is None}")

        print("\n3. 잘못된 매개변수 테스트")
        # 음수 기간
        try:
            result = analyzer.calculate_bollinger_bands(minimal_data, period=-1)
            print(f"음수 기간 결과: {result}")
        except Exception as e:
            print(f"음수 기간 예외 처리: {type(e).__name__}")

        # 0 표준편차
        try:
            result = analyzer.calculate_bollinger_bands(minimal_data, stddev_multiplier=0)
            print(f"0 표준편차 결과: {result}")
        except Exception as e:
            print(f"0 표준편차 예외 처리: {type(e).__name__}")

        print("\n4. 존재하지 않는 심볼 테스트")
        result = await analyzer.comprehensive_bollinger_analysis("INVALID_SYMBOL")
        print(f"존재하지 않는 심볼 결과: {result is None}")

        print("\n✅ 엣지 케이스 테스트 완료")

    except Exception as e:
        logger.error(f"엣지 케이스 테스트 중 오류: {str(e)}")
        print(f"❌ 테스트 실패: {str(e)}")


async def main():
    """메인 테스트 함수"""
    print("🚀 볼린저 밴드 분석기 종합 테스트 시작")
    print(f"프로젝트 루트: {project_root}")

    try:
        # 1. 기본 계산 테스트
        await test_basic_bb_calculation()

        # 2. 스퀴즈 감지 테스트
        await test_squeeze_detection()

        # 3. 돌파 확률 테스트
        await test_breakout_probability()

        # 4. 종합 분석 테스트
        await test_comprehensive_analysis()

        # 5. 엣지 케이스 테스트
        await test_edge_cases()

        print_separator("전체 테스트 완료")
        print("✅ 모든 테스트가 완료되었습니다.")
        print("📋 로그 파일: test_bollinger_analyzer.log")

    except Exception as e:
        logger.error(f"메인 테스트 중 오류: {str(e)}")
        print(f"❌ 전체 테스트 실패: {str(e)}")


if __name__ == "__main__":
    # 비동기 테스트 실행
    asyncio.run(main())