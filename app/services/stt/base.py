"""
STT Base Class - Strategy Pattern Interface

모든 STT 구현체는 이 추상 클래스를 상속받아야 합니다.
새로운 STT 서비스(예: Clova, ReturnZero)를 추가할 때
이 인터페이스를 구현하면 됩니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Generator
import numpy as np


@dataclass
class TranscriptionResult:
    """STT 결과를 담는 데이터 클래스"""
    text: str                          # 변환된 텍스트
    language: str = "ko"               # 감지된 언어
    confidence: float = 0.0            # 신뢰도 (0.0 ~ 1.0)
    duration: float = 0.0              # 오디오 길이 (초)
    segments: Optional[list] = None    # 타임스탬프가 있는 세그먼트


class STTBase(ABC):
    """
    Speech-to-Text 추상 베이스 클래스
    
    Strategy Pattern을 적용하여 다양한 STT 서비스를 
    동일한 인터페이스로 사용할 수 있게 합니다.
    
    Example:
        stt = WhisperSTT()  # 또는 ClovaSTT(), ReturnZeroSTT()
        result = stt.transcribe(audio_bytes)
        print(result.text)
    """
    
    @abstractmethod
    def transcribe(self, audio: bytes, language: str = "ko") -> TranscriptionResult:
        """
        오디오를 텍스트로 변환합니다.
        
        Args:
            audio: 오디오 데이터 (WAV 형식 권장)
            language: 언어 코드 (기본값: "ko")
            
        Returns:
            TranscriptionResult: 변환 결과
        """
        pass
    
    @abstractmethod
    def transcribe_stream(
        self, 
        audio_stream: Generator[bytes, None, None],
        language: str = "ko"
    ) -> Generator[str, None, None]:
        """
        스트리밍 오디오를 실시간으로 변환합니다.
        
        Args:
            audio_stream: 오디오 청크 제너레이터
            language: 언어 코드
            
        Yields:
            str: 부분 변환 텍스트
        """
        pass
    
    def transcribe_file(self, file_path: str, language: str = "ko") -> TranscriptionResult:
        """
        오디오 파일을 텍스트로 변환합니다.
        
        Args:
            file_path: 오디오 파일 경로
            language: 언어 코드
            
        Returns:
            TranscriptionResult: 변환 결과
        """
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        return self.transcribe(audio_bytes, language)
    
    @staticmethod
    def audio_to_numpy(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
        """오디오 바이트를 numpy 배열로 변환하는 유틸리티 메서드"""
        import soundfile as sf
        import io
        
        audio_io = io.BytesIO(audio_bytes)
        audio_array, sr = sf.read(audio_io)
        
        # 모노로 변환
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        return audio_array.astype(np.float32)
