"""
TTS Base Class - Strategy Pattern Interface

모든 TTS 구현체는 이 추상 클래스를 상속받아야 합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class SynthesisResult:
    """TTS 결과를 담는 데이터 클래스"""
    audio: bytes            # 생성된 오디오 데이터
    sample_rate: int        # 샘플레이트 (Hz)
    duration: float         # 오디오 길이 (초)
    format: str = "wav"     # 오디오 포맷


class TTSBase(ABC):
    """
    Text-to-Speech 추상 베이스 클래스
    
    Strategy Pattern을 적용하여 다양한 TTS 서비스를 
    동일한 인터페이스로 사용할 수 있게 합니다.
    
    Example:
        tts = gTTSTTS()  # 또는 EdgeTTS(), ClovaTTS()
        result = tts.synthesize("안녕하세요")
        # result.audio를 재생
    """
    
    @abstractmethod
    def synthesize(self, text: str, language: str = "ko") -> SynthesisResult:
        """
        텍스트를 음성으로 변환합니다.
        
        Args:
            text: 변환할 텍스트
            language: 언어 코드 (기본값: "ko")
            
        Returns:
            SynthesisResult: 생성된 오디오 데이터
        """
        pass
    
    def synthesize_stream(
        self, 
        text: str, 
        language: str = "ko"
    ) -> Generator[bytes, None, None]:
        """
        텍스트를 스트리밍 방식으로 음성 변환합니다.
        
        기본 구현은 전체 합성 후 청크 단위로 반환합니다.
        필요시 서브클래스에서 오버라이드하세요.
        
        Args:
            text: 변환할 텍스트
            language: 언어 코드
            
        Yields:
            bytes: 오디오 청크
        """
        result = self.synthesize(text, language)
        
        # 청크 단위로 반환 (4KB씩)
        chunk_size = 4096
        for i in range(0, len(result.audio), chunk_size):
            yield result.audio[i:i + chunk_size]
    
    def save_to_file(self, text: str, file_path: str, language: str = "ko") -> str:
        """
        텍스트를 음성 파일로 저장합니다.
        
        Args:
            text: 변환할 텍스트
            file_path: 저장할 파일 경로
            language: 언어 코드
            
        Returns:
            str: 저장된 파일 경로
        """
        result = self.synthesize(text, language)
        
        with open(file_path, "wb") as f:
            f.write(result.audio)
        
        return file_path
