"""
Whisper STT Implementation

OpenAI Whisper를 사용한 STT 구현체입니다.
faster-whisper 라이브러리를 사용하여 빠른 추론을 지원합니다.
"""

from typing import Generator, Optional
import logging

from .base import STTBase, TranscriptionResult
from ...config import settings

logger = logging.getLogger(__name__)


class WhisperSTT(STTBase):
    """
    Whisper 기반 Speech-to-Text 구현
    
    faster-whisper를 사용하여 로컬에서 빠른 추론을 수행합니다.
    
    Attributes:
        model_size: 모델 크기 (tiny, base, small, medium, large-v3)
        device: 실행 디바이스 (cuda, cpu)
        
    Example:
        stt = WhisperSTT(model_size="base", device="cpu")
        result = stt.transcribe(audio_bytes)
        print(result.text)
    """
    
    def __init__(
        self, 
        model_size: str = None, 
        device: str = None,
        compute_type: str = "float16"
    ):
        """
        Args:
            model_size: 모델 크기 (기본값: config에서 로드)
            device: cuda 또는 cpu (기본값: config에서 로드)
            compute_type: 연산 타입 (float16, int8 등)
        """
        self.model_size = model_size or settings.whisper_model_size
        self.device = device or settings.whisper_device
        self.compute_type = compute_type if device == "cuda" else "int8"
        
        self._model = None
        
    @property
    def model(self):
        """Lazy loading: 첫 사용 시에만 모델 로드"""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Whisper 모델 로드"""
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            
            logger.info("Whisper model loaded successfully")
            
        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio: bytes, language: str = "ko") -> TranscriptionResult:
        """
        오디오를 텍스트로 변환

        Args:
            audio: WAV 형식의 오디오 바이트
            language: 언어 코드 (ko, en 등)

        Returns:
            TranscriptionResult: 변환 결과
        """
        import tempfile
        import os

        # 임시 파일로 저장 (faster-whisper는 파일 경로 필요)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio)
            tmp_path = tmp_file.name

        try:
            segments, info = self.model.transcribe(
                tmp_path,
                language=language,
                beam_size=5,
                best_of=5,
                vad_filter=True,
                initial_prompt=self._get_initial_prompt(language)
            )

            # 세그먼트 수집
            segment_list = []
            full_text_parts = []

            for segment in segments:
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
                full_text_parts.append(segment.text.strip())

            full_text = " ".join(full_text_parts)

            return TranscriptionResult(
                text=full_text,
                language=info.language,
                confidence=info.language_probability,
                duration=info.duration,
                segments=segment_list
            )

        finally:
            os.unlink(tmp_path)
    
    def transcribe_stream(
        self, 
        audio_stream: Generator[bytes, None, None],
        language: str = "ko"
    ) -> Generator[str, None, None]:
        """
        스트리밍 오디오 변환 (간소화 버전)
        
        Note: 실제 실시간 스트리밍을 위해서는 VAD(Voice Activity Detection)와
              청크 단위 처리가 필요합니다. 이 구현은 청크를 모아서 처리합니다.
        """
        buffer = b""
        
        for chunk in audio_stream:
            buffer += chunk
            
            # 일정 크기 이상 모이면 처리
            if len(buffer) > 32000 * 2:  # 약 1초 분량 (16kHz, 16bit)
                result = self.transcribe(buffer, language)
                if result.text.strip():
                    yield result.text
                buffer = b""
        
        # 남은 버퍼 처리
        if buffer:
            result = self.transcribe(buffer, language)
            if result.text.strip():
                yield result.text
    
    def _get_initial_prompt(self, language: str) -> str:
        """언어별 초기 프롬프트 반환 (인식률 향상)"""
        prompts = {
            "ko": "이것은 한국어 대화입니다. 고객 상담, 문의, 주문 관련 내용입니다.",
            "en": "This is a customer service conversation in English.",
        }
        return prompts.get(language, "")


class MockWhisperSTT(STTBase):
    """
    테스트용 Mock Whisper STT
    
    실제 모델 로드 없이 테스트할 때 사용합니다.
    """
    
    def transcribe(self, audio: bytes, language: str = "ko") -> TranscriptionResult:
        """Mock 변환 - 항상 고정된 텍스트 반환"""
        return TranscriptionResult(
            text="[Mock] 안녕하세요, 주문 관련해서 문의드립니다.",
            language=language,
            confidence=0.95,
            duration=2.5,
            segments=[
                {"start": 0.0, "end": 2.5, "text": "[Mock] 안녕하세요, 주문 관련해서 문의드립니다."}
            ]
        )
    
    def transcribe_stream(
        self, 
        audio_stream: Generator[bytes, None, None],
        language: str = "ko"
    ) -> Generator[str, None, None]:
        """Mock 스트리밍 - 고정된 텍스트 yield"""
        yield "[Mock] 안녕하세요,"
        yield " 주문 관련해서"
        yield " 문의드립니다."
