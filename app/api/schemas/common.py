"""
Common API Schemas
"""

from pydantic import BaseModel
from typing import Optional, Any


class APIResponse(BaseModel):
    """Generic API Response"""
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None


class EmotionInfo(BaseModel):
    """Emotion information"""
    emotion: str  # "angry", "happy", "sad", "neutral", "fearful", "surprised"
    korean_label: str  # "화남", "기쁨", etc.
    confidence: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"


class ConfigResponse(BaseModel):
    """Configuration response"""
    stt_provider: str
    tts_provider: str
    llm_model: str
    embedding_model: str
