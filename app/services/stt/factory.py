"""
STT Factory - Factory Pattern Implementation

STT 제공자에 따라 적절한 STT 구현체를 생성합니다.
새로운 STT 서비스를 추가할 때 이 팩토리에 등록하면 됩니다.
"""

from typing import Literal
from .base import STTBase
from .whisper_stt import WhisperSTT, MockWhisperSTT
from ...config import settings


STTProvider = Literal["whisper", "clova", "returnzero", "mock"]


class STTFactory:
    """
    STT 객체 생성 팩토리
    
    설정이나 인자에 따라 적절한 STT 구현체를 생성합니다.
    
    Example:
        # 기본 설정 사용
        stt = STTFactory.create()
        
        # 특정 제공자 지정
        stt = STTFactory.create("whisper")
        
        # Mock 테스트용
        stt = STTFactory.create("mock")
    """
    
    # 등록된 STT 제공자 매핑
    _providers = {
        "whisper": WhisperSTT,
        "mock": MockWhisperSTT,
        # 추후 추가 가능:
        # "clova": ClovaSTT,
        # "returnzero": ReturnZeroSTT,
    }
    
    @classmethod
    def create(cls, provider: STTProvider = None, **kwargs) -> STTBase:
        """
        STT 인스턴스 생성
        
        Args:
            provider: STT 제공자 이름 (기본값: config에서 로드)
            **kwargs: STT 생성자에 전달할 추가 인자
            
        Returns:
            STTBase: STT 인스턴스
            
        Raises:
            ValueError: 알 수 없는 제공자인 경우
        """
        provider = provider or settings.stt_provider
        
        if provider not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown STT provider: {provider}. "
                f"Available providers: {available}"
            )
        
        stt_class = cls._providers[provider]
        return stt_class(**kwargs)
    
    @classmethod
    def register(cls, name: str, stt_class: type):
        """
        새로운 STT 제공자 등록
        
        Args:
            name: 제공자 이름
            stt_class: STTBase를 상속받은 클래스
            
        Example:
            STTFactory.register("custom", CustomSTT)
        """
        if not issubclass(stt_class, STTBase):
            raise TypeError(f"{stt_class} must be a subclass of STTBase")
        
        cls._providers[name] = stt_class
    
    @classmethod
    def available_providers(cls) -> list:
        """사용 가능한 STT 제공자 목록 반환"""
        return list(cls._providers.keys())
