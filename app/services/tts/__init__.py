"""
TTS (Text-to-Speech) Service Module

텍스트를 음성으로 변환하는 서비스 모듈입니다.
"""

from .base import TTSBase, SynthesisResult
from .gtts_tts import gTTSTTS, MockTTS
from .factory import TTSFactory

__all__ = [
    "TTSBase",
    "SynthesisResult",
    "gTTSTTS",
    "MockTTS",
    "TTSFactory",
]
