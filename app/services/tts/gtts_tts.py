"""
gTTS (Google Text-to-Speech) Implementation

Google TTS를 사용한 무료 TTS 구현체입니다.
"""

from typing import Generator
import logging
import io

from .base import TTSBase, SynthesisResult

logger = logging.getLogger(__name__)


class gTTSTTS(TTSBase):
    """
    Google TTS 기반 Text-to-Speech 구현
    
    무료로 사용할 수 있으며, 다양한 언어를 지원합니다.
    음질은 기본적이지만 프로토타입에 적합합니다.
    
    Example:
        tts = gTTSTTS()
        result = tts.synthesize("안녕하세요")
    """
    
    def __init__(self, slow: bool = False):
        """
        Args:
            slow: True면 천천히 읽음
        """
        self.slow = slow
    
    def synthesize(self, text: str, language: str = "ko") -> SynthesisResult:
        """
        텍스트를 음성으로 변환

        Args:
            text: 변환할 텍스트
            language: 언어 코드 (ko, en 등)

        Returns:
            SynthesisResult: MP3 형식의 오디오
        """
        try:
            from gtts import gTTS

            # gTTS로 음성 생성
            tts = gTTS(text=text, lang=language, slow=self.slow)

            # 메모리에 저장
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_bytes = audio_buffer.read()

            return SynthesisResult(
                audio=audio_bytes,
                sample_rate=24000,
                duration=self._estimate_duration(text),
                format="mp3"
            )

        except ImportError:
            logger.error("gTTS not installed. Run: pip install gTTS")
            raise
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def _estimate_duration(self, text: str) -> float:
        """텍스트 길이로 대략적인 오디오 길이 추정"""
        # 한국어 기준 약 5글자/초
        chars_per_second = 5 if not self.slow else 3
        return len(text) / chars_per_second
    
    def _mp3_to_wav(self, mp3_bytes: bytes) -> bytes:
        """MP3를 WAV로 변환 (pydub 필요)"""
        try:
            from pydub import AudioSegment
            
            mp3_io = io.BytesIO(mp3_bytes)
            audio = AudioSegment.from_mp3(mp3_io)
            
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            
            return wav_io.read()
        except ImportError:
            logger.warning("pydub not installed, returning MP3")
            return mp3_bytes


class MockTTS(TTSBase):
    """
    테스트용 Mock TTS
    
    실제 음성 생성 없이 테스트할 때 사용합니다.
    """
    
    def synthesize(self, text: str, language: str = "ko") -> SynthesisResult:
        """Mock 합성 - 빈 오디오 반환"""
        # 1초 길이의 무음 WAV 생성
        import struct
        
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)
        
        # WAV 헤더 생성
        wav_header = self._create_wav_header(num_samples, sample_rate)
        audio_data = struct.pack('<' + 'h' * num_samples, *([0] * num_samples))
        
        return SynthesisResult(
            audio=wav_header + audio_data,
            sample_rate=sample_rate,
            duration=duration,
            format="wav"
        )
    
    def _create_wav_header(self, num_samples: int, sample_rate: int) -> bytes:
        """WAV 파일 헤더 생성"""
        import struct
        
        bits_per_sample = 16
        num_channels = 1
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = num_samples * block_align
        
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,  # Subchunk1Size
            1,   # AudioFormat (PCM)
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size
        )
        
        return header
