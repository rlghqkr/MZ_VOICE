"""
FastAPI Dependencies

Dependency injection for pipeline, session manager, etc.
"""

import os
import uuid
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..pipelines import VoiceRAGPipeline
from ..services.session import SessionManager, SessionStorage
from ..services.summary import LLMSummarizer
from ..services.mail import SMTPMailSender
from ..config import settings

logger = logging.getLogger(__name__)

# Audio temp directory
AUDIO_TEMP_DIR = Path("./data/audio_temp")
AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_pipeline() -> VoiceRAGPipeline:
    """
    Get singleton VoiceRAGPipeline instance

    Documents are loaded on first access.
    """
    logger.info("Initializing VoiceRAGPipeline...")
    pipeline = VoiceRAGPipeline(use_mock_emotion=True)

    # Load documents from JSON directory
    json_dir = Path("./data/documents")
    if json_dir.exists():
        pipeline.load_documents_from_json_directory(str(json_dir))
        logger.info(f"Loaded documents from {json_dir}")
    else:
        logger.warning(f"Document directory not found: {json_dir}")

    return pipeline


@lru_cache()
def get_session_manager() -> SessionManager:
    """
    Get singleton SessionManager instance
    """
    logger.info("Initializing SessionManager...")

    storage = SessionStorage()

    # Optional: Initialize summarizer and mail sender
    summarizer = None
    mail_sender = None

    try:
        if settings.openai_api_key:
            summarizer = LLMSummarizer()
    except Exception as e:
        logger.warning(f"Failed to initialize summarizer: {e}")

    try:
        if settings.smtp_username and settings.smtp_password:
            mail_sender = SMTPMailSender()
    except Exception as e:
        logger.warning(f"Failed to initialize mail sender: {e}")

    return SessionManager(
        storage=storage,
        summarizer=summarizer,
        mail_sender=mail_sender
    )


def save_audio_temp(audio_bytes: bytes, extension: str = "mp3") -> str:
    """
    Save audio to temporary file and return the audio ID

    Args:
        audio_bytes: Audio data
        extension: File extension (mp3, wav)

    Returns:
        str: Audio ID for retrieval
    """
    audio_id = uuid.uuid4().hex[:12]
    file_path = AUDIO_TEMP_DIR / f"{audio_id}.{extension}"

    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    logger.debug(f"Saved audio temp file: {file_path}")
    return audio_id


def get_audio_temp(audio_id: str) -> Optional[tuple[bytes, str]]:
    """
    Get audio from temporary storage

    Args:
        audio_id: Audio ID

    Returns:
        tuple: (audio_bytes, content_type) or None if not found
    """
    # Check for common extensions
    for ext in ["mp3", "wav"]:
        file_path = AUDIO_TEMP_DIR / f"{audio_id}.{ext}"
        if file_path.exists():
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            content_type = "audio/mpeg" if ext == "mp3" else "audio/wav"
            return audio_bytes, content_type

    return None


def cleanup_old_audio_files(max_age_hours: int = 1):
    """
    Clean up old temporary audio files

    Args:
        max_age_hours: Maximum age in hours before deletion
    """
    import time

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for file_path in AUDIO_TEMP_DIR.glob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                logger.debug(f"Cleaned up old audio file: {file_path}")
