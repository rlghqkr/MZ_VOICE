"""
Query Router - LLM 기반 쿼리 분류기

질문 유형에 따라 적절한 RAG 시스템으로 라우팅합니다.
- 법령 관련 질문 → GraphRAG
- 일반 질문 → Hybrid RAG
"""

import logging
from typing import Literal, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...config import settings

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """쿼리 유형"""
    LAW = "law"          # 법령/법률 관련
    GENERAL = "general"  # 일반 청년 정책


@dataclass
class RouterResult:
    """라우터 결과"""
    query_type: QueryType
    confidence: str  # high, medium, low
    reason: str


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 청년 복지 업무를 담당하는 변호사입니다. 사용자의 입력에 대해서 법령 검색이 필요할지 판단해주세요.

## 분류 기준

### LAW (법령/법률 관련)
- 특정 법률, 법령, 시행령, 규정에 대한 질문
- "~법", "제~조", "~령", "법적 근거", "법률상" 등의 표현 포함
- 법적 요건, 자격 조건의 법적 근거를 묻는 질문
- **법령만을 근거로 요구하거나 사용자의 질문에 답할 때 법령 근거가 포함되어 있다면 더 신뢰도가 높은 정답이 될 수 있는 질문**
- 예시: "저희는 결혼한 지 1년 된 신혼부부이고 여주시에 주민등록이 되어 있는데요, 전세로 살면서 은행에서 전세자금 대출을 받아 이자를 내고 있습니다. 이런 경우에 대출이자 지원을 받으려면 어떤 조건을 충족해야 하나요?", "저 지금 군포시에 살고 있고 전월세 보증금 대출 이자 같은거 지원 신청하려고 하는데요, 이거 신청은 어디로 내야 돼요? 서류도 뭐 같이 내야 하는지, 꼭 제가 직접 가야 되는지도 궁금해요.", "강남구에 주민등록 돼 있는 결혼 1년차 무주택 부부인데요, 전세 보증금 대출 이자 지원 받으려면 소득이랑 집 면적·보증금 기준이 어떻게 되나요?"

### GENERAL (일반 청년 정책)
- 청년 정책, 지원 프로그램, 서비스에 대한 질문
- 신청 방법, 지원 내용, 자격 조건 등 실용적 정보
- 법령 참조 없이 답변 가능한 일반적인 질문
- 예시: "청년 취업 지원 프로그램 알려줘", "주거 지원 받으려면 어떻게 해?", "창업 관련 정책 있어?"

## 출력 형식
반드시 아래 형식으로만 답변하세요:
TYPE: [LAW 또는 GENERAL]
CONFIDENCE: [high, medium, low]
REASON: [분류 이유를 한 줄로]"""),
    ("human", "{query}")
])


class QueryRouter:
    """
    LLM 기반 쿼리 라우터

    질문을 분석하여 법령 관련인지 일반 정책 관련인지 분류합니다.

    Example:
        router = QueryRouter()
        result = router.route("청년기본법 제3조가 뭐야?")
        # result.query_type == QueryType.LAW

        result = router.route("취업 지원 프로그램 알려줘")
        # result.query_type == QueryType.GENERAL
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Args:
            model: 분류에 사용할 LLM 모델 (빠르고 저렴한 모델 권장)
        """
        self.model = model
        self._llm = None
        self._chain = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=self.model,
                temperature=0
            )
        return self._llm

    @property
    def chain(self):
        if self._chain is None:
            self._chain = ROUTER_PROMPT | self.llm | StrOutputParser()
        return self._chain

    def route(self, query: str) -> RouterResult:
        """
        질문을 분류합니다.

        Args:
            query: 사용자 질문

        Returns:
            RouterResult: 분류 결과
        """
        try:
            response = self.chain.invoke({"query": query})
            return self._parse_response(response)
        except Exception as e:
            logger.error(f"Router error: {e}")
            # 에러 시 기본값으로 GENERAL 반환
            return RouterResult(
                query_type=QueryType.GENERAL,
                confidence="low",
                reason=f"분류 실패: {str(e)}"
            )

    def _parse_response(self, response: str) -> RouterResult:
        """LLM 응답 파싱"""
        lines = response.strip().split("\n")

        query_type = QueryType.GENERAL
        confidence = "medium"
        reason = ""

        for line in lines:
            line = line.strip()
            if line.startswith("TYPE:"):
                type_str = line.replace("TYPE:", "").strip().upper()
                if type_str == "LAW":
                    query_type = QueryType.LAW
                else:
                    query_type = QueryType.GENERAL
            elif line.startswith("CONFIDENCE:"):
                confidence = line.replace("CONFIDENCE:", "").strip().lower()
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        return RouterResult(
            query_type=query_type,
            confidence=confidence,
            reason=reason
        )

    def is_law_query(self, query: str) -> bool:
        """법령 관련 질문인지 빠르게 확인"""
        result = self.route(query)
        return result.query_type == QueryType.LAW


# 키워드 기반 빠른 분류 (LLM 호출 전 사전 필터링용)
LAW_KEYWORDS = [
    "법령", "법률", "시행령", "시행규칙", "규정", "조례",
    "제1조", "제2조", "제3조", "제4조", "제5조",
    "법적 근거", "법적 요건", "법률상", "법에 따라",
    "기본법", "특별법", "청년기본법", "고용촉진법"
]


def quick_law_check(query: str) -> bool:
    """키워드 기반 빠른 법령 질문 체크"""
    query_lower = query.lower()
    return any(kw in query_lower for kw in LAW_KEYWORDS)
