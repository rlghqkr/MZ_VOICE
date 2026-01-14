"""
Voice Processing API Schemas
"""

from pydantic import BaseModel
from typing import Optional
from .common import EmotionInfo


class VoiceProcessResponse(BaseModel):
    """Voice processing response"""
    transcription: str
    response_text: str
    emotion: EmotionInfo
    audio_url: Optional[str] = None
    processing_time: float
