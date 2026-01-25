"""
TTS Base Class - Strategy Pattern Interface

모든 TTS 구현체는 이 추상 클래스를 상속받아야 합니다.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional, List, Tuple


@dataclass
class SynthesisResult:
    """TTS 결과를 담는 데이터 클래스"""
    audio: bytes            # 생성된 오디오 데이터
    sample_rate: int        # 샘플레이트 (Hz)
    duration: float         # 오디오 길이 (초)
    format: str = "wav"     # 오디오 포맷


def split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분할

    한국어와 영어 문장 부호를 모두 처리합니다.
    """
    # 문장 종결 패턴 (한국어/영어)
    # 마침표, 물음표, 느낌표 뒤에 공백이나 문자열 끝이 오는 경우
    sentence_pattern = r'(?<=[.!?。？！])\s+'

    sentences = re.split(sentence_pattern, text.strip())

    # 빈 문장 제거 및 정리
    return [s.strip() for s in sentences if s.strip()]


def is_sentence_complete(text: str) -> bool:
    """
    문장이 완성되었는지 판단

    한국어 종결어미와 문장 부호를 확인합니다.
    """
    if not text or not text.strip():
        return False

    stripped = text.strip()

    # 문장 부호로 끝나는 경우
    if re.search(r'[.!?。？！]$', stripped):
        return True

    # 한국어 종결어미 패턴 (다, 요, 죠, 니다, 세요 등)
    korean_endings = r'(다|요|죠|니다|세요|습니다|됩니다|합니다|입니다|겠습니다|드립니다|주세요|하세요|됩니까|합니까|입니까)$'
    if re.search(korean_endings, stripped):
        return True

    return False


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

    def synthesize_sentence_stream(
        self,
        text: str,
        language: str = "ko"
    ) -> Generator[Tuple[str, bytes], None, None]:
        """
        텍스트를 문장 단위로 나눠서 TTS 합성 후 스트리밍

        실시간 전화 응대처럼 문장이 완성되면 바로 음성을 출력합니다.

        Args:
            text: 변환할 전체 텍스트
            language: 언어 코드

        Yields:
            Tuple[str, bytes]: (문장 텍스트, 오디오 바이트)
        """
        sentences = split_into_sentences(text)

        for sentence in sentences:
            if sentence:
                result = self.synthesize(sentence, language)
                yield sentence, result.audio
