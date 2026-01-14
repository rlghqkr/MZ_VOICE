"""
STT (Speech-to-Text) Service Module

음성을 텍스트로 변환하는 서비스 모듈입니다.
Strategy Pattern을 사용하여 다양한 STT 제공자를 지원합니다.

Example:
    from app.services.stt import STTFactory
    
    # 기본 STT 생성
    stt = STTFactory.create()
    result = stt.transcribe(audio_bytes)
    
    # 특정 제공자 사용
    stt = STTFactory.create("whisper", model_size="large-v3")
"""

from .base import STTBase, TranscriptionResult
from .whisper_stt import WhisperSTT, MockWhisperSTT
from .factory import STTFactory

__all__ = [
    "STTBase",
    "TranscriptionResult",
    "WhisperSTT",
    "MockWhisperSTT",
    "STTFactory",
]
