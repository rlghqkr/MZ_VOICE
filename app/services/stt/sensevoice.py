"""
SenseVoice STT Implementation

Calls a remote ASR API with multipart audio upload.
"""

from typing import Generator, Optional, Tuple, Dict, Any
import logging

from .base import STTBase, TranscriptionResult
from .llm_correction import correct_with_llm
from ...config import settings

logger = logging.getLogger(__name__)


class SenseVoiceSTT(STTBase):
    """
    Remote ASR-based Speech-to-Text implementation.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        include_emotion: Optional[bool] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.asr_api_base_url).rstrip("/")
        if include_emotion is None:
            include_emotion = settings.asr_include_emotion
        self.include_emotion = include_emotion
        self.timeout_seconds = timeout_seconds or settings.asr_timeout_seconds

    def transcribe(self, audio: bytes, language: str = "ko") -> TranscriptionResult:
        import requests

        url = f"{self.base_url}/api/v1/asr"
        params = {"include_emotion": str(self.include_emotion).lower()}
        files = {"audio": ("audio.wav", audio, "audio/wav")}

        try:
            response = requests.post(
                url,
                params=params,
                files=files,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"ASR request failed: {exc}") from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"ASR request failed: {response.status_code} {response.text}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("ASR response is not valid JSON") from exc

        text = self._extract_text(payload)
        if not text:
            keys = list(payload.keys()) if isinstance(payload, dict) else []
            raise RuntimeError(f"ASR response missing text field: keys={keys}")

        if settings.stt_llm_correction:
            text = correct_with_llm(text)

        language_out = payload.get("language") or language
        confidence = self._to_float(payload.get("confidence"), default=0.0)
        duration = self._to_float(payload.get("duration"), default=0.0)
        segments = payload.get("segments")

        emotion_label, emotion_scores = self._extract_emotion(payload)

        return TranscriptionResult(
            text=text,
            language=language_out,
            confidence=confidence,
            duration=duration,
            segments=segments,
            emotion_label=emotion_label,
            emotion_scores=emotion_scores,
            extra=payload,
        )

    def transcribe_stream(
        self,
        audio_stream: Generator[bytes, None, None],
        language: str = "ko",
    ) -> Generator[str, None, None]:
        buffer = b""

        for chunk in audio_stream:
            buffer += chunk
            if len(buffer) > 32000 * 2:
                result = self.transcribe(buffer, language)
                if result.text.strip():
                    yield result.text
                buffer = b""

        if buffer:
            result = self.transcribe(buffer, language)
            if result.text.strip():
                yield result.text

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""

        for key in ("text", "transcription", "result"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        nested = payload.get("data")
        if isinstance(nested, dict):
            for key in ("text", "transcription", "result"):
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""

    @staticmethod
    def _extract_emotion(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[dict]]:
        if not isinstance(payload, dict):
            return None, None

        emotion = payload.get("emotion")
        if isinstance(emotion, dict):
            label = emotion.get("label") or emotion.get("emotion") or emotion.get("name")
            scores = emotion.get("scores") or emotion.get("emotion_scores")
            if isinstance(scores, dict):
                return label, scores
            return label, None
        if isinstance(emotion, str):
            return emotion, None

        label = payload.get("emotion_label") or payload.get("emotion_name")
        scores = payload.get("emotion_scores")
        if isinstance(scores, dict):
            return label, scores
        return label, None

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
