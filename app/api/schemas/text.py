"""
Text Processing API Schemas
"""

from pydantic import BaseModel
from typing import Optional


class TextProcessRequest(BaseModel):
    """Text processing request"""
    text: str
    emotion: Optional[str] = "neutral"
    session_id: Optional[str] = None
    return_audio: bool = False


class TextProcessResponse(BaseModel):
    """Text processing response"""
    response_text: str
    audio_url: Optional[str] = None
    processing_time: float
