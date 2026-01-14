"""
SMTP Mail Sender

SMTP 프로토콜을 사용한 메일 발송 구현체.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from .base import MailSenderBase, MailMessage, SendResult

logger = logging.getLogger(__name__)


class SMTPMailSender(MailSenderBase):
    """
    SMTP 메일 발송기
    
    SMTP 서버를 통해 메일을 발송한다.
    Gmail, Naver, 회사 메일 서버 등에서 사용 가능.
    
    Gmail 사용 시:
        - host: smtp.gmail.com
        - port: 587
        - use_tls: True
        - 앱 비밀번호 사용 필요 (2FA 활성화 시)
    
    Example:
        sender = SMTPMailSender(
            host="smtp.gmail.com",
            port=587,
            username="your-email@gmail.com",
            password="your-app-password",
            from_email="your-email@gmail.com",
            from_name="고객상담센터"
        )
        
        result = sender.send(MailMessage(
            to="customer@example.com",
            subject="상담 요약",
            body="..."
        ))
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        from_email: str,
        from_name: str = None,
        use_tls: bool = True,
        use_ssl: bool = False,
        timeout: int = 30
    ):
        """
        Args:
            host: SMTP 서버 호스트
            port: SMTP 서버 포트 (TLS: 587, SSL: 465)
            username: SMTP 인증 사용자명
            password: SMTP 인증 비밀번호
            from_email: 발신자 이메일
            from_name: 발신자 이름 (옵션)
            use_tls: TLS 사용 여부 (기본값: True)
            use_ssl: SSL 사용 여부 (기본값: False)
            timeout: 연결 타임아웃 (초)
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.from_name = from_name or from_email
        self.use_tls = use_tls
        self.use_ssl = use_ssl
        self.timeout = timeout
    
    def send(self, message: MailMessage) -> SendResult:
        """
        메일 발송

        Args:
            message: 발송할 메일 메시지

        Returns:
            SendResult: 발송 결과
        """
        try:
            # MIME 메시지 생성
            msg = self._create_mime_message(message)

            # SMTP 연결
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout)
            else:
                server = smtplib.SMTP(self.host, self.port, timeout=self.timeout)

            try:
                if self.use_tls and not self.use_ssl:
                    server.starttls()

                server.login(self.username, self.password)

                # 수신자 목록 구성
                recipients = [message.to]
                if message.cc:
                    recipients.extend(message.cc)
                if message.bcc:
                    recipients.extend(message.bcc)

                server.sendmail(
                    self.from_email,
                    recipients,
                    msg.as_string()
                )

                logger.info(f"Mail sent successfully to {message.to}")

                return SendResult(
                    success=True,
                    message_id=msg["Message-ID"]
                )

            finally:
                server.quit()

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return SendResult(success=False, error=f"Authentication failed: {e}")
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return SendResult(success=False, error=f"SMTP error: {e}")
        except Exception as e:
            logger.error(f"Mail sending failed: {e}")
            return SendResult(success=False, error=str(e))
    
    def _create_mime_message(self, message: MailMessage) -> MIMEMultipart:
        """MIME 메시지 생성"""
        msg = MIMEMultipart("alternative")
        
        # 헤더 설정
        msg["Subject"] = message.subject
        msg["From"] = f"{self.from_name} <{self.from_email}>"
        msg["To"] = message.to
        
        if message.cc:
            msg["Cc"] = ", ".join(message.cc)
        
        # Plain text 본문
        msg.attach(MIMEText(message.body, "plain", "utf-8"))
        
        # HTML 본문 (있는 경우)
        if message.html_body:
            msg.attach(MIMEText(message.html_body, "html", "utf-8"))
        
        return msg
    
    def test_connection(self) -> bool:
        """SMTP 연결 테스트"""
        try:
            if self.use_ssl:
                server = smtplib.SMTP_SSL(
                    self.host, 
                    self.port, 
                    timeout=self.timeout
                )
            else:
                server = smtplib.SMTP(
                    self.host, 
                    self.port, 
                    timeout=self.timeout
                )
            
            try:
                if self.use_tls and not self.use_ssl:
                    server.starttls()
                server.login(self.username, self.password)
                logger.info("SMTP connection test successful")
                return True
            finally:
                server.quit()
                
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False


class MockMailSender(MailSenderBase):
    """테스트용 Mock 메일 발송기"""
    
    def __init__(self):
        self.sent_messages: list[MailMessage] = []
    
    def send(self, message: MailMessage) -> SendResult:
        """Mock 발송 - 실제 발송 없이 기록만"""
        self.sent_messages.append(message)
        logger.info(f"[Mock] Mail sent to {message.to}: {message.subject}")
        
        return SendResult(
            success=True,
            message_id=f"mock-{len(self.sent_messages)}"
        )
    
    def get_last_message(self) -> Optional[MailMessage]:
        """마지막 발송된 메시지 조회"""
        return self.sent_messages[-1] if self.sent_messages else None
    
    def clear(self):
        """발송 기록 초기화"""
        self.sent_messages.clear()
