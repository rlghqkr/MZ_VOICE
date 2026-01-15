"""
간단한 RAG 테스트 스크립트
Usage: python tests/test_rag.py
"""

import sys
import logging
import time
import os
from pathlib import Path
from datetime import datetime

# 프롬프트 로깅 활성화 (import 전에 설정 필수!)
os.environ["ENABLE_PROMPT_LOGGING"] = "true"

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# logs 디렉토리 생성
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# 로그 파일명 (날짜_시분초 포맷)
log_file = log_dir / f"test_rag_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

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
logger.info(f"Prompt logging enabled: {os.getenv('ENABLE_PROMPT_LOGGING')}")

from app.services.rag import RAGChain, HybridRAGService
from app.services.emotion import Emotion


def _log_source_documents(documents, query_type: str = ""):
    """Source documents 상세 출력 (원본 vs contextualized)"""
    logger.info("-" * 40)
    logger.info(f"Source Documents Detail ({query_type}):")
    for i, doc in enumerate(documents):
        logger.info(f"\n[Doc {i+1}]")
        logger.info(f"  Metadata: {doc.metadata.get('source_file', 'N/A')} | {doc.metadata.get('chunk_type', 'N/A')}")
        logger.info(f"  Has Context: {doc.metadata.get('has_context', False)}")

        # Contextualized content (실제 검색에 사용된 내용)
        contextualized = doc.page_content[:300]
        logger.info(f"  [Contextualized]: {contextualized}...")

        # Original content (원본 내용)
        original = doc.metadata.get('original_content', '')
        if original:
            logger.info(f"  [Original]: {original[:300]}...")
        else:
            logger.info(f"  [Original]: (not available - same as contextualized)")
    logger.info("-" * 40)


def test_ragchain():
    """RAGChain 기본 테스트"""
    logger.info("Starting RAGChain basic test")
    
    start_time = time.time()
    rag = RAGChain(search_type="hybrid", use_reranker=False)
    init_time = time.time() - start_time
    logger.info(f"RAGChain initialized in {init_time:.2f}s")
    
    query = "청년 취업 지원 프로그램 알려줘"
    logger.info(f"Processing query: '{query}'")
    
    query_start = time.time()
    response = rag.query(query, emotion=Emotion.NEUTRAL)
    query_time = time.time() - query_start
    
    logger.info(f"Query completed in {query_time:.2f}s")
    logger.info(f"Response length: {len(response.answer)} characters")
    logger.info(f"Source documents retrieved: {len(response.source_documents)}")
    logger.info(f"Response: {response.answer[:200]}...")

    # Source documents 상세 출력
    _log_source_documents(response.source_documents, "RAGChain")


def test_hybrid_rag():
    """HybridRAGService 테스트"""
    logger.info("Starting HybridRAGService test")
    
    start_time = time.time()
    service = HybridRAGService(use_reranker=False, use_quick_check=True)
    init_time = time.time() - start_time
    logger.info(f"HybridRAGService initialized in {init_time:.2f}s")
    
    # 일반 질문
    query1 = "청년 창업 지원 알려줘"
    logger.info(f"Processing general query: '{query1}'")
    
    query_start = time.time()
    response = service.query(query1, emotion=Emotion.NEUTRAL)
    query_time = time.time() - query_start
    
    logger.info(f"General query completed in {query_time:.2f}s")
    logger.info(f"Query type: {response.query_type}")
    logger.info(f"Response length: {len(response.answer)} characters")
    logger.info(f"Response: {response.answer[:200]}...")

    # Source documents 상세 출력
    _log_source_documents(response.source_documents, "GENERAL")

    # 법령 질문
    query2 = "청년기본법이 뭐야?"
    logger.info(f"Processing law query: '{query2}'")
    
    query_start = time.time()
    response = service.query(query2, emotion=Emotion.NEUTRAL)
    query_time = time.time() - query_start
    
    logger.info(f"Law query completed in {query_time:.2f}s")
    logger.info(f"Query type: {response.query_type}")
    logger.info(f"Response length: {len(response.answer)} characters")
    logger.info(f"Response: {response.answer[:200]}...")

    # Source documents 상세 출력
    _log_source_documents(response.source_documents, "LAW")


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("RAG Test Suite Started")
        logger.info("=" * 60)
        
        test_ragchain()
        test_hybrid_rag()
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
