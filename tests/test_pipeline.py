"""
간단한 파이프라인 테스트 스크립트
Usage: python tests/test_pipeline.py
"""

import sys
import logging
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# logs 디렉토리 생성
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# 로그 파일명 (날짜_시분초 포맷)
log_file = log_dir / f"test_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

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

from app.pipelines import VoiceRAGPipeline


def test_text_pipeline():
    """텍스트 파이프라인 테스트"""
    logger.info("Starting text pipeline test")
    
    start_time = time.time()
    pipeline = VoiceRAGPipeline(
        use_mock_emotion=True,
        use_hybrid_rag=True,
        use_query_builder=False
    )
    init_time = time.time() - start_time
    logger.info(f"Pipeline initialized in {init_time:.2f}s")
    
    query = "청년 복지 정책 알려줘"
    logger.info(f"Processing text input: '{query}'")
    
    result = pipeline.process_text(query)
    
    logger.info(f"Input text: {result.input_text}")
    logger.info(f"Output length: {len(result.output_text)} characters")
    logger.info(f"Processing time: {result.processing_time:.2f}s")
    logger.info(f"Output: {result.output_text[:200]}...")
    
    if result.query_type:
        logger.info(f"Query type: {result.query_type}")
    
    if result.error:
        logger.error(f"Pipeline error: {result.error}")


def test_stats():
    """파이프라인 통계 확인"""
    logger.info("Starting pipeline stats test")
    
    start_time = time.time()
    pipeline = VoiceRAGPipeline()
    init_time = time.time() - start_time
    logger.info(f"Pipeline initialized in {init_time:.2f}s")
    
    stats = pipeline.get_stats()
    
    logger.info(f"STT provider: {stats['stt_provider']}")
    logger.info(f"TTS provider: {stats['tts_provider']}")
    logger.info(f"Hybrid RAG enabled: {stats['use_hybrid_rag']}")
    logger.info(f"Query Builder enabled: {stats['use_query_builder']}")
    logger.debug(f"Full stats: {stats}")


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("Pipeline Test Suite Started")
        logger.info("=" * 60)
        
        test_stats()
        test_text_pipeline()
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
