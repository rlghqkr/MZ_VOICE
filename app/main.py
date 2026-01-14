"""
Voice RAG Chatbot - Main Entry Point

음성 기반 RAG 고객 상담 챗봇 애플리케이션입니다.

Usage:
    # Gradio 모드 (기본)
    python -m app.main
    python -m app.main --mode gradio

    # FastAPI 모드
    python -m app.main --mode fastapi

    # 또는 직접 실행
    python app/main.py
"""

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.config import settings

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_gradio():
    """Gradio 앱 실행"""
    from app.interfaces import launch_app

    logger.info("Starting Gradio interface...")
    launch_app(
        share=settings.gradio_share,
        port=settings.gradio_server_port
    )


def run_fastapi():
    """FastAPI 앱 실행"""
    import uvicorn

    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="MZ-VOICE Chatbot")
    parser.add_argument(
        "--mode",
        choices=["gradio", "fastapi"],
        default="gradio",
        help="실행 모드 선택 (기본값: gradio)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="서버 포트 (기본값: gradio=7860, fastapi=8000)"
    )

    args = parser.parse_args()

    logger.info("="*50)
    logger.info("MZ-VOICE Chatbot Starting...")
    logger.info("="*50)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"STT Provider: {settings.stt_provider}")
    logger.info(f"TTS Provider: {settings.tts_provider}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info("="*50)

    if args.mode == "fastapi":
        run_fastapi()
    else:
        run_gradio()


if __name__ == "__main__":
    main()
