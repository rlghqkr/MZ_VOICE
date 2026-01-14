"""
Session Manager

세션 생성/종료를 관리하고, 종료 시 요약 및 메일 발송을 조율한다.
"""

import uuid
import logging
from typing import Optional, Callable
from dataclasses import dataclass

from .storage import SessionStorage, SessionData, MessageRole

logger = logging.getLogger(__name__)


@dataclass
class SessionEndResult:
    """세션 종료 결과"""
    session_id: str
    success: bool
    summary: Optional[str] = None
    mail_sent: bool = False
    error: Optional[str] = None


class SessionManager:
    """
    세션 관리자
    
    세션의 생명주기를 관리한다:
    - 세션 생성 (고유 ID 발급)
    - 메시지 추가
    - 세션 종료 (요약 생성 → 메일 발송)
    
    Example:
        manager = SessionManager(
            storage=SessionStorage(),
            summarizer=LLMSummarizer(),
            mail_sender=SMTPMailSender()
        )
        
        # 세션 시작
        session_id = manager.start_session(customer_email="user@example.com")
        
        # 대화 진행
        manager.add_user_message(session_id, "환불하고 싶어요")
        manager.add_assistant_message(session_id, "네, 도와드리겠습니다.")
        
        # 세션 종료 (자동으로 요약 생성 및 메일 발송)
        result = manager.end_session(session_id)
    """
    
    def __init__(
        self,
        storage: SessionStorage = None,
        summarizer = None,  # LLMSummarizer
        mail_sender = None,  # MailSenderBase
    ):
        self.storage = storage or SessionStorage()
        self.summarizer = summarizer
        self.mail_sender = mail_sender
        
        # 세션 종료 시 호출할 콜백 (옵션)
        self._on_session_end_callbacks: list[Callable] = []
    
    def start_session(self, customer_email: str = None) -> str:
        """
        새 세션 시작
        
        Args:
            customer_email: 고객 이메일 (나중에 설정 가능)
            
        Returns:
            str: 생성된 세션 ID
        """
        session_id = self._generate_session_id()
        self.storage.create_session(session_id, customer_email)
        logger.info(f"Session started: {session_id}")
        return session_id
    
    def _generate_session_id(self) -> str:
        """세션 ID 생성"""
        return f"session-{uuid.uuid4().hex[:12]}"
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """세션 조회"""
        return self.storage.get_session(session_id)
    
    def add_user_message(
        self, 
        session_id: str, 
        content: str,
        emotion: str = None
    ) -> bool:
        """사용자 메시지 추가"""
        metadata = {"emotion": emotion} if emotion else None
        return self.storage.add_message(
            session_id, 
            MessageRole.USER, 
            content,
            metadata
        )
    
    def add_assistant_message(
        self, 
        session_id: str, 
        content: str
    ) -> bool:
        """어시스턴트 응답 추가"""
        return self.storage.add_message(
            session_id,
            MessageRole.ASSISTANT,
            content
        )
    
    def set_customer_email(self, session_id: str, email: str) -> bool:
        """고객 이메일 설정"""
        return self.storage.set_customer_email(session_id, email)
    
    def end_session(self, session_id: str) -> SessionEndResult:
        """
        세션 종료 및 후처리
        
        1. 세션 종료 처리
        2. 대화 내용 요약 생성
        3. 고객에게 요약 메일 발송
        
        Args:
            session_id: 종료할 세션 ID
            
        Returns:
            SessionEndResult: 종료 결과
        """
        session = self.storage.get_session(session_id)
        
        if not session:
            return SessionEndResult(
                session_id=session_id,
                success=False,
                error="Session not found"
            )
        
        # 1. 세션 종료
        self.storage.end_session(session_id)
        
        result = SessionEndResult(
            session_id=session_id,
            success=True
        )
        
        # 2. 대화 요약 생성
        if self.summarizer and session.messages:
            try:
                conversation = session.get_conversation_text()
                result.summary = self.summarizer.summarize(conversation)
                logger.info(f"Summary generated for session {session_id}")
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")
                result.error = f"Summary failed: {e}"
        
        # 3. 메일 발송
        if self.mail_sender and session.customer_email and result.summary:
            try:
                self.mail_sender.send_summary_mail(
                    to_email=session.customer_email,
                    summary=result.summary,
                    session_data=session
                )
                result.mail_sent = True
                logger.info(f"Summary mail sent to {session.customer_email}")
            except Exception as e:
                logger.error(f"Mail sending failed: {e}")
                result.error = f"Mail failed: {e}"
        
        # 콜백 실행
        for callback in self._on_session_end_callbacks:
            try:
                callback(session, result)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        return result
    
    def on_session_end(self, callback: Callable):
        """세션 종료 콜백 등록"""
        self._on_session_end_callbacks.append(callback)
    
    def cleanup_session(self, session_id: str):
        """세션 데이터 정리 (메모리 해제)"""
        self.storage.delete_session(session_id)


class MockSessionManager(SessionManager):
    """테스트용 Mock 세션 매니저"""
    
    def end_session(self, session_id: str) -> SessionEndResult:
        """Mock 종료 - 요약/메일 없이 바로 종료"""
        session = self.storage.get_session(session_id)
        
        if not session:
            return SessionEndResult(
                session_id=session_id,
                success=False,
                error="Session not found"
            )
        
        self.storage.end_session(session_id)
        
        return SessionEndResult(
            session_id=session_id,
            success=True,
            summary="[Mock Summary] 테스트 요약입니다.",
            mail_sent=False
        )
