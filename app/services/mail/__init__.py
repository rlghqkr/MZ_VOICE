"""
Mail Sending Service Module

메일 발송 서비스 모듈.
Strategy Pattern 적용으로 SMTP, SendGrid 등 교체 가능.
"""

from .base import (
    MailSenderBase,
    MailMessage,
    SendResult
)
from .smtp_sender import (
    SMTPMailSender,
    MockMailSender
)

__all__ = [
    "MailSenderBase",
    "MailMessage",
    "SendResult",
    "SMTPMailSender",
    "MockMailSender",
]
