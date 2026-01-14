"""
Session Management Service Module

세션 생성/종료 및 대화 내역 관리 모듈.
"""

from .storage import (
    SessionStorage,
    SessionData,
    Message,
    MessageRole
)
from .manager import (
    SessionManager,
    SessionEndResult,
    MockSessionManager
)

__all__ = [
    "SessionStorage",
    "SessionData",
    "Message",
    "MessageRole",
    "SessionManager",
    "SessionEndResult",
    "MockSessionManager",
]
