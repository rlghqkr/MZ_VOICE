"""
Voice Processing API Schemas
"""

from pydantic import BaseModel
from typing import Optional, Literal
from .common import EmotionInfo


class VoiceProcessResponse(BaseModel):
    """Voice processing response"""
    transcription: str
    response_text: str
    emotion: EmotionInfo
    audio_url: Optional[str] = None
    processing_time: float
    needs_more_info: bool = False  # QueryBuilder가 추가 정보 수집 필요 여부
    conversation_phase: Optional[str] = None  # 대화 단계 (GREETING, COLLECTING, READY 등)


class VoiceStreamEvent(BaseModel):
    """SSE 스트리밍 이벤트"""
    event: Literal["transcription", "emotion", "text_chunk", "audio_chunk", "done", "error"]
    data: dict

    class Config:
        # JSON 직렬화 시 한글 유지
        json_encoders = {
            str: lambda v: v
        }
