"""
TTS Factory - Factory Pattern Implementation
"""

from typing import Literal
from .base import TTSBase
from .gtts_tts import gTTSTTS, MockTTS
from ...config import settings


TTSProvider = Literal["gtts", "edge", "clova", "mock"]


class TTSFactory:
    """TTS 객체 생성 팩토리"""
    
    _providers = {
        "gtts": gTTSTTS,
        "mock": MockTTS,
        # "edge": EdgeTTS,
        # "clova": ClovaTTS,
    }
    
    @classmethod
    def create(cls, provider: TTSProvider = None, **kwargs) -> TTSBase:
        """TTS 인스턴스 생성"""
        provider = provider or settings.tts_provider
        
        if provider not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown TTS provider: {provider}. "
                f"Available providers: {available}"
            )
        
        tts_class = cls._providers[provider]
        return tts_class(**kwargs)
    
    @classmethod
    def register(cls, name: str, tts_class: type):
        """새로운 TTS 제공자 등록"""
        if not issubclass(tts_class, TTSBase):
            raise TypeError(f"{tts_class} must be a subclass of TTSBase")
        cls._providers[name] = tts_class
    
    @classmethod
    def available_providers(cls) -> list:
        """사용 가능한 TTS 제공자 목록"""
        return list(cls._providers.keys())
