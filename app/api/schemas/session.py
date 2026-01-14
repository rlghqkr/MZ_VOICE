"""
Session API Schemas
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime


class SessionCreateRequest(BaseModel):
    """Session creation request"""
    customer_email: Optional[str] = None


class SessionResponse(BaseModel):
    """Session information response"""
    session_id: str
    customer_email: Optional[str] = None
    created_at: datetime
    ended_at: Optional[datetime] = None
    message_count: int


class SessionEndResponse(BaseModel):
    """Session end response"""
    session_id: str
    success: bool
    summary: Optional[str] = None
    mail_sent: bool = False
    error: Optional[str] = None


class MessageResponse(BaseModel):
    """Message in session"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    emotion: Optional[str] = None


class SessionMessagesResponse(BaseModel):
    """Session messages response"""
    session_id: str
    messages: List[MessageResponse]


class EmailUpdateRequest(BaseModel):
    """Email update request"""
    email: str
