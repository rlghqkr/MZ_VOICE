"""
Session Storage

세션별 대화 내역을 저장하고 조회하는 모듈.
현재는 메모리 기반으로 구현, 추후 Redis 등으로 교체 가능.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """메시지 발신자 역할"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """대화 메시지"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict] = None  # 감정, 음성파일 경로 등 추가 정보
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class SessionData:
    """세션 데이터"""
    session_id: str
    customer_email: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    
    def add_message(self, role: MessageRole, content: str, metadata: Dict = None):
        """메시지 추가"""
        self.messages.append(Message(
            role=role,
            content=content,
            metadata=metadata
        ))
    
    def get_conversation_text(self) -> str:
        """대화 내용을 텍스트로 반환 (요약용)"""
        lines = []
        for msg in self.messages:
            if msg.role == MessageRole.USER:
                role_label = "고객"
            elif msg.role == MessageRole.ASSISTANT:
                role_label = "상담사"
            else:
                role_label = "시스템"
            lines.append(f"[{role_label}] {msg.content}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "customer_email": self.customer_email,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None
        }


class SessionStorage:
    """
    세션 저장소 (메모리 기반)
    
    세션별 대화 내역을 저장하고 조회한다.
    프로덕션에서는 Redis나 DB로 교체 필요.
    
    Example:
        storage = SessionStorage()
        storage.create_session("session-123", "user@example.com")
        storage.add_message("session-123", MessageRole.USER, "환불하고 싶어요")
        storage.add_message("session-123", MessageRole.ASSISTANT, "네, 도와드리겠습니다.")
        
        session = storage.get_session("session-123")
        print(session.get_conversation_text())
    """
    
    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
    
    def create_session(self, session_id: str, customer_email: str = None) -> SessionData:
        """새 세션 생성"""
        if session_id in self._sessions:
            logger.warning(f"Session {session_id} already exists, returning existing")
            return self._sessions[session_id]
        
        session = SessionData(
            session_id=session_id,
            customer_email=customer_email
        )
        self._sessions[session_id] = session
        logger.info(f"Session created: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """세션 조회"""
        return self._sessions.get(session_id)
    
    def add_message(
        self, 
        session_id: str, 
        role: MessageRole, 
        content: str,
        metadata: Dict = None
    ) -> bool:
        """세션에 메시지 추가"""
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            return False
        
        session.add_message(role, content, metadata)
        return True
    
    def end_session(self, session_id: str) -> Optional[SessionData]:
        """세션 종료 (삭제하지 않고 종료 시간만 기록)"""
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            return None
        
        session.ended_at = datetime.now()
        logger.info(f"Session ended: {session_id}")
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
    
    def set_customer_email(self, session_id: str, email: str) -> bool:
        """고객 이메일 설정"""
        session = self.get_session(session_id)
        if not session:
            return False
        session.customer_email = email
        return True
    
    def list_sessions(self) -> List[str]:
        """모든 세션 ID 목록"""
        return list(self._sessions.keys())
    
    def get_active_sessions(self) -> List[SessionData]:
        """종료되지 않은 세션 목록"""
        return [s for s in self._sessions.values() if s.ended_at is None]
