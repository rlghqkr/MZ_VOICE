"""
Mail Sender Base Class

메일 발송 서비스의 추상 베이스 클래스.
Strategy Pattern 적용으로 SMTP, SendGrid, AWS SES 등 교체 가능.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime


@dataclass
class MailMessage:
    """메일 메시지"""
    to: str                              # 수신자 이메일
    subject: str                         # 제목
    body: str                            # 본문 (plain text)
    html_body: Optional[str] = None      # HTML 본문 (옵션)
    cc: Optional[List[str]] = None       # 참조
    bcc: Optional[List[str]] = None      # 숨은 참조


@dataclass 
class SendResult:
    """발송 결과"""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    sent_at: datetime = None
    
    def __post_init__(self):
        if self.sent_at is None and self.success:
            self.sent_at = datetime.now()


class MailSenderBase(ABC):
    """
    메일 발송 추상 베이스 클래스
    
    Strategy Pattern을 적용하여 다양한 메일 발송 방식 지원:
    - SMTP 직접 발송
    - SendGrid API
    - AWS SES
    - 기타
    
    Example:
        sender = SMTPMailSender(host="smtp.gmail.com", port=587, ...)
        result = sender.send(MailMessage(
            to="user@example.com",
            subject="상담 요약",
            body="..."
        ))
    """
    
    @abstractmethod
    def send(self, message: MailMessage) -> SendResult:
        """
        메일 발송
        
        Args:
            message: 발송할 메일 메시지
            
        Returns:
            SendResult: 발송 결과
        """
        pass
    
    def send_summary_mail(
        self,
        to_email: str,
        summary: str,
        session_data = None  # SessionData
    ) -> SendResult:
        """
        상담 요약 메일 발송 (헬퍼 메서드)
        
        Args:
            to_email: 수신자 이메일
            summary: 요약 텍스트
            session_data: 세션 데이터 (옵션, 추가 정보용)
            
        Returns:
            SendResult: 발송 결과
        """
        # 날짜 포맷
        session_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        if session_data and session_data.created_at:
            session_date = session_data.created_at.strftime("%Y-%m-%d %H:%M")
        
        subject = f"[상담 요약] {session_date} 상담 내역"
        
        body = self._build_summary_body(summary, session_date)
        html_body = self._build_summary_html(summary, session_date)
        
        message = MailMessage(
            to=to_email,
            subject=subject,
            body=body,
            html_body=html_body
        )
        
        return self.send(message)
    
    def _build_summary_body(self, summary: str, session_date: str) -> str:
        """요약 메일 본문 (plain text)"""
        return f"""안녕하세요, 고객님.

{session_date}에 진행된 상담 내용을 요약하여 보내드립니다.

{'='*50}

{summary}

{'='*50}

추가 문의 사항이 있으시면 언제든 연락 주세요.
감사합니다.

---
본 메일은 자동 발송되었습니다.
"""
    
    def _build_summary_html(self, summary: str, session_date: str) -> str:
        """요약 메일 본문 (HTML)"""
        # 줄바꿈을 <br>로 변환
        summary_html = summary.replace('\n', '<br>')
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #4A90D9; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; background: #f9f9f9; }}
        .summary-box {{ background: white; padding: 20px; border-left: 4px solid #4A90D9; margin: 20px 0; }}
        .footer {{ text-align: center; color: #888; font-size: 12px; padding: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>상담 요약</h2>
            <p>{session_date}</p>
        </div>
        <div class="content">
            <p>안녕하세요, 고객님.</p>
            <p>진행된 상담 내용을 요약하여 보내드립니다.</p>
            
            <div class="summary-box">
                {summary_html}
            </div>
            
            <p>추가 문의 사항이 있으시면 언제든 연락 주세요.</p>
            <p>감사합니다.</p>
        </div>
        <div class="footer">
            본 메일은 자동 발송되었습니다.
        </div>
    </div>
</body>
</html>
"""
