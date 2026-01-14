"""
간단한 STT 테스트 스크립트
Usage: python tests/test_stt.py
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# logs 디렉토리 생성
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# 로그 파일명 (날짜_시분초 포맷)
log_file = log_dir / f"test_stt_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler(log_file, encoding='utf-8')  # 파일 저장
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_file}")

from app.services.stt import STTFactory, MockWhisperSTT


def test_stt_factory():
    """STT Factory 테스트"""
    logger.info("Starting STT Factory test")
    
    # Mock STT 생성
    start_time = time.time()
    stt = STTFactory.create("mock")
    init_time = time.time() - start_time
    
    logger.info(f"Mock STT created: {type(stt).__name__}")
    logger.info(f"Initialization time: {init_time:.4f}s")
    
    # 사용 가능한 제공자
    providers = STTFactory.available_providers()
    logger.info(f"Available providers: {providers}")
    logger.info(f"Total provider count: {len(providers)}")


def test_mock_transcribe():
    """Mock STT 변환 테스트"""
    logger.info("Starting Mock STT transcription test")
    
    start_time = time.time()
    stt = MockWhisperSTT()
    init_time = time.time() - start_time
    logger.info(f"MockWhisperSTT initialized in {init_time:.4f}s")
    
    audio_data = b"fake_audio_data"
    logger.info(f"Processing audio data: {len(audio_data)} bytes")
    
    transcribe_start = time.time()
    result = stt.transcribe(audio_data)
    transcribe_time = time.time() - transcribe_start
    
    logger.info(f"Transcription completed in {transcribe_time:.4f}s")
    logger.info(f"Transcribed text: '{result.text}'")
    logger.info(f"Detected language: {result.language}")
    logger.info(f"Confidence score: {result.confidence:.4f}")
    
    if result.segments:
        logger.info(f"Number of segments: {len(result.segments)}")


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("STT Test Suite Started")
        logger.info("=" * 60)
        
        test_stt_factory()
        test_mock_transcribe()
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
