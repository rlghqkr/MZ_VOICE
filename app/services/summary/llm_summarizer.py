"""
LLM Summarizer

LLM을 사용하여 대화 내용을 요약한다.
"""

import logging
import os
from typing import Optional
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from ...config import settings
from ...utils.logging_utils import log_prompt, log_llm_response

logger = logging.getLogger(__name__)

# 환경변수로 프롬프트 로깅 활성화 여부 제어
ENABLE_PROMPT_LOGGING = os.getenv("ENABLE_PROMPT_LOGGING", "false").lower() == "true"


# 요약 프롬프트 템플릿
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 고객 상담 대화를 요약하는 전문가입니다.
다음 상담 대화를 읽고 핵심 내용을 요약해주세요. 해당 날짜는 꼭 포함해야 합니다.

요약 형식:
**[1월 20일 상담 요약]**

 청년 전세자금 지원 안내

안녕하세요!

청년 전세자금 지원 관련 상담 내용을

요약해 안내드립니다 

**안내 결과**

- 버팀목전세자금대출: 기금e든든 또는 수탁은행 방문 신청
- 신청 절차: 수탁은행 접수 → 대상자 심사 → 지원 여부 확정

**신청·문의**

- 홈페이지: 기금e든든 ([https://enhuf.molit.go.kr](https://enhuf.molit.go.kr/))
- 전화: 주택도시기금 1566-9009
- 방문: 우리·국민·신한·하나·농협 등 수탁은행

추가로 궁금한 사항이 있으면 언제든지 문의해 주세요!

청년님의 주거 안정과 취업 준비를 응원합니다

간결하고 명확하게 작성하되, 중요한 정보(주문번호, 날짜, 금액 등)는 반드시 포함하세요."""),
    ("human", """다음 상담 대화를 요약해주세요:

{conversation}

요약:""")
])


class SummarizerBase(ABC):
    """요약기 추상 베이스 클래스"""
    
    @abstractmethod
    def summarize(self, conversation: str) -> str:
        """대화 내용 요약"""
        pass


class LLMSummarizer(SummarizerBase):
    """
    LLM 기반 대화 요약기
    
    OpenAI GPT 또는 다른 LLM을 사용하여 대화 내용을 요약한다.
    
    Example:
        summarizer = LLMSummarizer()
        
        conversation = '''
        [고객] 주문한 물건이 아직 안 왔어요
        [상담사] 주문번호 알려주시겠어요?
        [고객] 12345입니다
        [상담사] 확인해보니 배송 중입니다. 내일 도착 예정입니다.
        '''
        
        summary = summarizer.summarize(conversation)
    """
    
    def __init__(
        self, 
        model: str = None,
        temperature: float = 0.3
    ):
        """
        Args:
            model: LLM 모델명 (기본값: config에서 로드)
            temperature: 생성 온도 (낮을수록 일관된 출력)
        """
        self.model = model or settings.llm_model
        self.temperature = temperature
        self._llm = None
    
    @property
    def llm(self) -> ChatOpenAI:
        """LLM 인스턴스 (Lazy Loading)"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=self.model,
                temperature=self.temperature
            )
        return self._llm
    
    def summarize(self, conversation: str) -> str:
        """
        대화 내용 요약

        Args:
            conversation: 대화 텍스트 (역할 라벨 포함)

        Returns:
            str: 요약된 텍스트
        """
        if not conversation.strip():
            return "대화 내용이 없습니다."

        try:
            # 프롬프트 로깅
            if ENABLE_PROMPT_LOGGING:
                try:
                    formatted_prompt = SUMMARY_PROMPT.format(conversation=conversation)
                    log_prompt(
                        logger=logger,
                        prompt_name="Conversation Summarizer",
                        prompt_text=formatted_prompt,
                        user_input=conversation[:200] + "..." if len(conversation) > 200 else conversation
                    )
                except Exception as e:
                    logger.debug(f"Failed to log prompt: {e}")
            
            chain = SUMMARY_PROMPT | self.llm
            response = chain.invoke({"conversation": conversation})
            summary = response.content.strip()
            
            # 응답 로깅
            if ENABLE_PROMPT_LOGGING:
                log_llm_response(logger, summary, "Summary")
            
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise
    
    def summarize_with_metadata(
        self, 
        conversation: str,
        customer_name: str = None,
        session_date: str = None
    ) -> dict:
        """
        메타데이터 포함 요약
        
        Returns:
            dict: {summary, customer_name, session_date, ...}
        """
        summary = self.summarize(conversation)
        
        return {
            "summary": summary,
            "customer_name": customer_name,
            "session_date": session_date,
            "message_count": len(conversation.split('\n'))
        }


class MockSummarizer(SummarizerBase):
    """테스트용 Mock 요약기"""
    
    def summarize(self, conversation: str) -> str:
        """Mock 요약 - 고정된 텍스트 반환"""
        line_count = len(conversation.strip().split('\n'))
        
        return f"""[문의 내용]
- 테스트 문의 (총 {line_count}개 메시지)

[처리 결과]
- Mock 요약입니다. 실제 LLM 요약으로 교체 필요.

[추가 안내]
- 추가 문의 시 고객센터(1588-0000)로 연락 바랍니다."""
