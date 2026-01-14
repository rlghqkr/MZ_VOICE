"""
Query Builder Graph - LangGraph 기반 복지 정보 수집 에이전트

LangGraph를 사용하여 사용자와의 대화를 통해 필요한 정보를 수집하고,
RAG 시스템에 보낼 최적의 쿼리를 생성합니다.

Workflow:
    사용자 입력 → [정보 추출] → [완료도 체크] → [후속 질문 생성] 또는 [쿼리 생성]
                                    ↓
                            정보 충분? ─────→ [RAG 쿼리 생성] → 종료
                                    ↓
                            정보 부족 ────→ [후속 질문] → 대기
"""

import logging
import json
from typing import TypedDict, Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ...config import settings

logger = logging.getLogger(__name__)


# ============ 상태 및 데이터 클래스 정의 ============

class NodeType(str, Enum):
    """노드 타입"""
    EXTRACT_INFO = "extract_info"
    CHECK_COMPLETENESS = "check_completeness"
    GENERATE_FOLLOW_UP = "generate_follow_up"
    BUILD_QUERY = "build_query"


class ConversationPhase(str, Enum):
    """대화 단계"""
    GREETING = "greeting"
    COLLECTING = "collecting"
    CLARIFYING = "clarifying"
    READY = "ready"
    COMPLETED = "completed"


@dataclass
class UserProfile:
    """수집된 사용자 정보"""
    age: Optional[int] = None
    age_group: Optional[str] = None
    region: Optional[str] = None
    region_detail: Optional[str] = None
    income_level: Optional[str] = None
    employment_status: Optional[str] = None
    family_status: Optional[str] = None
    housing_status: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    specific_needs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "age": self.age,
            "age_group": self.age_group,
            "region": self.region,
            "region_detail": self.region_detail,
            "income_level": self.income_level,
            "employment_status": self.employment_status,
            "family_status": self.family_status,
            "housing_status": self.housing_status,
            "interests": self.interests,
            "specific_needs": self.specific_needs
        }

    def get_missing_essential_fields(self) -> List[str]:
        """필수 정보 중 누락된 필드 반환"""
        missing = []
        if not self.interests and not self.specific_needs:
            missing.append("interest")
        if not self.region:
            missing.append("region")
        return missing

    def is_sufficient(self) -> bool:
        """RAG 쿼리를 위한 최소 정보가 충분한지 확인"""
        has_interest = bool(self.interests or self.specific_needs)
        has_region = bool(self.region)
        return has_interest and has_region

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """딕셔너리에서 UserProfile 생성"""
        return cls(
            age=data.get("age"),
            age_group=data.get("age_group"),
            region=data.get("region"),
            region_detail=data.get("region_detail"),
            income_level=data.get("income_level"),
            employment_status=data.get("employment_status"),
            family_status=data.get("family_status"),
            housing_status=data.get("housing_status"),
            interests=data.get("interests", []),
            specific_needs=data.get("specific_needs", [])
        )


class QueryBuilderState(TypedDict):
    """LangGraph 상태 정의"""
    # 입력
    message: str                              # 현재 사용자 메시지
    session_id: str                           # 세션 ID
    emotion: Optional[str]                    # 감지된 감정

    # 대화 컨텍스트
    conversation_history: List[Dict[str, str]]  # 대화 히스토리
    original_question: Optional[str]           # 첫 질문

    # 사용자 프로필 (딕셔너리로 저장)
    user_profile: Dict[str, Any]

    # 처리 상태
    phase: str                                 # 현재 대화 단계
    extracted_info: Dict[str, Any]             # 이번 턴에서 추출된 정보
    is_sufficient: bool                        # 정보 충분 여부
    confidence: float                          # 정보 수집 완료도

    # 출력
    response_message: Optional[str]            # 사용자에게 보낼 메시지
    rag_query: Optional[str]                   # 생성된 RAG 쿼리
    follow_up_questions: List[str]             # 후속 질문들


@dataclass
class GraphAgentResponse:
    """그래프 에이전트 응답"""
    message: str
    phase: ConversationPhase
    user_profile: UserProfile
    rag_query: Optional[str] = None
    follow_up_questions: List[str] = field(default_factory=list)
    confidence: float = 0.0


# ============ 프롬프트 정의 ============

EXTRACT_INFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 청년 복지 상담 전문가입니다. 사용자의 메시지에서 복지 서비스 추천에 필요한 정보를 추출하세요.

## 추출 대상 정보
1. 나이/연령대: 구체적 나이 또는 청년(18-34), 중장년(35-64), 노년(65+)
2. 거주 지역: 시/도, 구/군/시
3. 소득 수준: 저소득, 차상위, 중위소득 X% 이하/이상
4. 취업 상태: 취업, 미취업, 구직중, 창업준비, 프리랜서, 학생
5. 가족 상황: 1인가구, 신혼부부, 한부모, 다자녀, 임신/출산
6. 주거 상태: 무주택, 전세, 월세, 자가
7. 관심 분야: 취업, 창업, 주거, 교육/훈련, 금융/대출, 건강/의료, 문화/여가, 상담
8. 구체적 요구: 사용자가 명시적으로 원하는 것

## 출력 형식 (JSON)
{{
    "age": null 또는 숫자,
    "age_group": null 또는 "청년"/"중장년"/"노년",
    "region": null 또는 "시/도명",
    "region_detail": null 또는 "구/군/시명",
    "income_level": null 또는 "저소득"/"차상위"/"중위소득 X%",
    "employment_status": null 또는 상태,
    "family_status": null 또는 상태,
    "housing_status": null 또는 상태,
    "interests": ["관심분야1", "관심분야2"],
    "specific_needs": ["구체적 요구1"]
}}

## 주의사항
- 명확하게 언급된 정보만 추출
- 추측하지 말고, 언급되지 않은 것은 null로 처리
- JSON 형식으로만 응답"""),
    ("human", "사용자 메시지: {message}\n\n이전 대화 컨텍스트: {context}")
])


FOLLOW_UP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 친근하고 전문적인 청년 복지 상담사입니다.
사용자가 적절한 복지 서비스를 찾을 수 있도록 자연스럽게 정보를 수집하는 질문을 생성하세요.

## 수집된 정보
{collected_info}

## 아직 수집되지 않은 정보
{missing_info}

## 사용자의 원래 질문
{original_question}

## 응답 지침
1. 자연스럽고 대화체로 질문하세요
2. 한 번에 2개 이하의 질문을 하세요
3. 사용자의 상황에 공감하는 톤을 유지하세요
4. 필수가 아닌 정보는 "혹시~"로 부드럽게 물어보세요

## 중요: 첫 질문 전략
- 관심 분야와 거주 지역이 모두 없으면, 두 가지를 한 번에 자연스럽게 물어보세요
- 예시: "어떤 분야의 복지 서비스가 필요하신가요? (취업, 주거, 창업, 교육 등) 그리고 현재 어느 지역에 거주하고 계신가요?"

## 우선순위
1. 관심 분야 + 거주 지역 (첫 질문에서 함께 수집)
2. 나이/연령대는 청년 정책 자격 확인에 필요
3. 소득/취업상태는 선택적 (특정 서비스에만 필요)"""),
    ("human", "위 정보를 바탕으로 자연스러운 후속 질문을 생성해주세요.")
])


BUILD_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 복지 정보 검색 전문가입니다.
수집된 사용자 정보를 바탕으로 RAG 시스템에 보낼 최적의 검색 쿼리를 생성하세요.

## 사용자 프로필
{user_profile}

## 원본 질문
{original_question}

## 쿼리 생성 지침
1. 사용자의 상황과 요구를 명확히 반영
2. 검색에 효과적인 키워드 포함
3. 자격 조건 관련 정보(나이, 소득, 지역) 포함
4. 구체적이고 명확한 문장으로 작성

## 출력 형식
검색에 최적화된 단일 쿼리 문장만 출력하세요."""),
    ("human", "위 정보를 바탕으로 RAG 검색 쿼리를 생성해주세요.")
])


# ============ QueryBuilderGraph 클래스 ============

class QueryBuilderGraph:
    """
    LangGraph 기반 Query Builder

    사용자와의 대화를 통해 필요한 정보를 수집하고,
    RAG 시스템에 보낼 최적의 쿼리를 생성합니다.

    Example:
        graph = QueryBuilderGraph()

        # 첫 메시지 처리
        response = graph.process("청년 지원 정책 알려줘", session_id="user123")
        # response.phase == COLLECTING
        # response.message == "어떤 분야의 지원이 필요하신가요?..."

        # 후속 대화
        response = graph.process("서울에서 취업 지원이요", session_id="user123")
        # response.phase == READY
        # response.rag_query == "서울 청년 취업 지원 정책..."
    """

    def __init__(self, model: str = None):
        """
        Args:
            model: 사용할 LLM 모델 (기본값: settings.llm_model)
        """
        self.model = model or settings.llm_model
        self._llm = None
        self._graph = None
        self._sessions: Dict[str, QueryBuilderState] = {}

    @property
    def llm(self) -> ChatOpenAI:
        """LLM 인스턴스 (Lazy Loading)"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=self.model,
                temperature=0.3
            )
        return self._llm

    @property
    def graph(self) -> StateGraph:
        """컴파일된 그래프 (Lazy Loading)"""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    def _build_graph(self) -> StateGraph:
        """워크플로우 그래프 구성"""
        workflow = StateGraph(QueryBuilderState)

        # 노드 추가
        workflow.add_node("extract_info", self._extract_info_node)
        workflow.add_node("check_completeness", self._check_completeness_node)
        workflow.add_node("generate_follow_up", self._generate_follow_up_node)
        workflow.add_node("build_query", self._build_query_node)

        # 엣지 정의
        workflow.set_entry_point("extract_info")
        workflow.add_edge("extract_info", "check_completeness")

        # 완료도 체크 후 분기
        workflow.add_conditional_edges(
            "check_completeness",
            self._route_after_check,
            {
                "generate_follow_up": "generate_follow_up",
                "build_query": "build_query"
            }
        )

        workflow.add_edge("generate_follow_up", END)
        workflow.add_edge("build_query", END)

        return workflow.compile()

    # ============ 노드 함수들 ============

    def _extract_info_node(self, state: QueryBuilderState) -> QueryBuilderState:
        """정보 추출 노드"""
        message = state["message"]
        history = state.get("conversation_history", [])

        # 대화 컨텍스트 구성
        context = "\n".join([
            f"{h['role']}: {h['content']}"
            for h in history[-5:]
        ])

        chain = EXTRACT_INFO_PROMPT | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "message": message,
                "context": context
            })

            # JSON 파싱
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]

            extracted = json.loads(result.strip())
            logger.debug(f"Extracted info: {extracted}")

        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}")
            extracted = {}
        except Exception as e:
            logger.error(f"정보 추출 실패: {e}")
            extracted = {}

        # 프로필 업데이트
        current_profile = state.get("user_profile", {})
        updated_profile = self._merge_profile(current_profile, extracted)

        # 대화 히스토리 업데이트
        updated_history = list(history)
        updated_history.append({"role": "user", "content": message})

        return {
            **state,
            "extracted_info": extracted,
            "user_profile": updated_profile,
            "conversation_history": updated_history
        }

    def _check_completeness_node(self, state: QueryBuilderState) -> QueryBuilderState:
        """완료도 체크 노드"""
        profile_dict = state.get("user_profile", {})
        profile = UserProfile.from_dict(profile_dict)
        message = state.get("message", "")

        # 충분성 체크
        is_sufficient = profile.is_sufficient()

        # 명시적 액션 요청 체크
        action_keywords = ["알려줘", "찾아줘", "검색해줘", "어떤 거 있어", "뭐 있어", "추천해줘"]
        has_action_request = any(kw in message for kw in action_keywords)

        # 정보가 충분하고 액션 요청이 있으면 진행
        should_proceed = is_sufficient and (has_action_request or len(profile.interests) > 0)

        # 신뢰도 계산
        confidence = self._calculate_confidence(profile)

        # 단계 결정
        if should_proceed:
            phase = ConversationPhase.READY.value
        else:
            phase = ConversationPhase.COLLECTING.value

        logger.debug(f"Completeness check: sufficient={is_sufficient}, proceed={should_proceed}")

        return {
            **state,
            "is_sufficient": should_proceed,
            "confidence": confidence,
            "phase": phase
        }

    def _generate_follow_up_node(self, state: QueryBuilderState) -> QueryBuilderState:
        """후속 질문 생성 노드"""
        profile_dict = state.get("user_profile", {})
        profile = UserProfile.from_dict(profile_dict)
        original_question = state.get("original_question", "복지 정보 문의")

        missing = profile.get_missing_essential_fields()

        chain = FOLLOW_UP_PROMPT | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "collected_info": json.dumps(profile_dict, ensure_ascii=False, indent=2),
                "missing_info": ", ".join(missing) if missing else "없음",
                "original_question": original_question
            })
            follow_up_message = result.strip()
        except Exception as e:
            logger.error(f"후속 질문 생성 실패: {e}")
            follow_up_message = "어떤 분야의 복지 서비스가 필요하신가요? (예: 취업, 주거, 창업, 교육 등) 그리고 현재 어느 지역에 거주하고 계신가요?"

        return {
            **state,
            "response_message": follow_up_message,
            "follow_up_questions": [follow_up_message],
            "phase": ConversationPhase.COLLECTING.value
        }

    def _build_query_node(self, state: QueryBuilderState) -> QueryBuilderState:
        """RAG 쿼리 생성 노드"""
        profile_dict = state.get("user_profile", {})
        profile = UserProfile.from_dict(profile_dict)
        original_question = state.get("original_question", "복지 정보 검색")

        chain = BUILD_QUERY_PROMPT | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "user_profile": json.dumps(profile_dict, ensure_ascii=False, indent=2),
                "original_question": original_question
            })
            rag_query = result.strip()
        except Exception as e:
            logger.error(f"RAG 쿼리 생성 실패: {e}")
            rag_query = self._build_fallback_query(profile, original_question)

        # 확인 메시지 생성
        confirmation_message = self._build_confirmation_message(profile)

        return {
            **state,
            "rag_query": rag_query,
            "response_message": confirmation_message,
            "phase": ConversationPhase.READY.value
        }

    # ============ 조건부 라우팅 ============

    def _route_after_check(self, state: QueryBuilderState) -> str:
        """완료도 체크 후 라우팅"""
        if state.get("is_sufficient", False):
            return "build_query"
        return "generate_follow_up"

    # ============ 헬퍼 함수들 ============

    def _merge_profile(self, current: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
        """프로필 병합 (기존 정보 유지하면서 새 정보 추가)"""
        merged = dict(current)

        # 단순 필드 병합
        simple_fields = ["age", "age_group", "region", "region_detail",
                        "income_level", "employment_status", "family_status", "housing_status"]
        for field in simple_fields:
            if extracted.get(field):
                merged[field] = extracted[field]

        # 리스트 필드 병합 (중복 제거)
        for field in ["interests", "specific_needs"]:
            current_list = merged.get(field, []) or []
            new_items = extracted.get(field, []) or []
            for item in new_items:
                if item and item not in current_list:
                    current_list.append(item)
            merged[field] = current_list

        return merged

    def _calculate_confidence(self, profile: UserProfile) -> float:
        """정보 수집 완료도 계산"""
        score = 0.0

        # 필수 정보 (0.6)
        if profile.interests or profile.specific_needs:
            score += 0.4
        if profile.region:
            score += 0.2

        # 선택 정보 (0.4)
        if profile.age or profile.age_group:
            score += 0.1
        if profile.income_level:
            score += 0.1
        if profile.employment_status:
            score += 0.1
        if profile.family_status or profile.housing_status:
            score += 0.1

        return min(score, 1.0)

    def _build_fallback_query(self, profile: UserProfile, original_question: str) -> str:
        """폴백 쿼리 생성"""
        parts = []

        if profile.age_group:
            parts.append(profile.age_group)
        elif profile.age:
            if 18 <= profile.age <= 34:
                parts.append("청년")
            elif 35 <= profile.age <= 64:
                parts.append("중장년")
            else:
                parts.append("노년")

        if profile.region:
            parts.append(profile.region)

        if profile.interests:
            parts.extend(profile.interests)

        if profile.specific_needs:
            parts.extend(profile.specific_needs)

        if parts:
            return " ".join(parts) + " 복지 서비스 지원 정책"
        return original_question

    def _build_confirmation_message(self, profile: UserProfile) -> str:
        """확인 메시지 생성"""
        parts = []

        if profile.age_group or profile.age:
            age_str = profile.age_group or f"{profile.age}세"
            parts.append(f"{age_str}")

        if profile.region:
            region_str = profile.region
            if profile.region_detail:
                region_str += f" {profile.region_detail}"
            parts.append(f"{region_str} 거주")

        if profile.interests:
            parts.append(f"{', '.join(profile.interests)} 분야")

        condition = " ".join(parts) if parts else "요청하신 조건"
        return f"네, {condition}에 맞는 복지 서비스를 찾아드릴게요."

    # ============ 세션 관리 ============

    def _get_or_create_session(self, session_id: str) -> QueryBuilderState:
        """세션 가져오기 또는 생성"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "message": "",
                "session_id": session_id,
                "emotion": None,
                "conversation_history": [],
                "original_question": None,
                "user_profile": {},
                "phase": ConversationPhase.GREETING.value,
                "extracted_info": {},
                "is_sufficient": False,
                "confidence": 0.0,
                "response_message": None,
                "rag_query": None,
                "follow_up_questions": []
            }
        return self._sessions[session_id]

    # ============ 공개 메서드 ============

    def process(
        self,
        message: str,
        session_id: str = "default",
        emotion: Optional[str] = None
    ) -> GraphAgentResponse:
        """
        사용자 메시지 처리

        Args:
            message: 사용자 입력 메시지
            session_id: 세션 식별자
            emotion: 감지된 감정 (선택)

        Returns:
            GraphAgentResponse: 에이전트 응답
        """
        # 세션 상태 가져오기
        session_state = self._get_or_create_session(session_id)

        # 첫 질문 저장
        if session_state["original_question"] is None:
            session_state["original_question"] = message

        # 입력 업데이트
        session_state["message"] = message
        session_state["emotion"] = emotion

        # 그래프 실행
        result = self.graph.invoke(session_state)

        # 세션 상태 업데이트
        self._sessions[session_id] = result

        # 응답 생성
        profile = UserProfile.from_dict(result.get("user_profile", {}))
        phase = ConversationPhase(result.get("phase", ConversationPhase.COLLECTING.value))

        return GraphAgentResponse(
            message=result.get("response_message", ""),
            phase=phase,
            user_profile=profile,
            rag_query=result.get("rag_query"),
            follow_up_questions=result.get("follow_up_questions", []),
            confidence=result.get("confidence", 0.0)
        )

    def reset_session(self, session_id: str):
        """세션 초기화"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_profile(self, session_id: str) -> Optional[UserProfile]:
        """세션의 사용자 프로필 반환"""
        if session_id in self._sessions:
            profile_dict = self._sessions[session_id].get("user_profile", {})
            return UserProfile.from_dict(profile_dict)
        return None

    def get_session_state(self, session_id: str) -> Optional[QueryBuilderState]:
        """세션 상태 반환 (디버깅용)"""
        return self._sessions.get(session_id)

    def visualize(self) -> str:
        """그래프 시각화 (Mermaid 형식)"""
        return self.graph.get_graph().draw_mermaid()


# ============ 싱글톤 패턴 ============

_default_graph_agent: Optional[QueryBuilderGraph] = None


def get_query_builder_graph() -> QueryBuilderGraph:
    """기본 QueryBuilderGraph 인스턴스 반환"""
    global _default_graph_agent
    if _default_graph_agent is None:
        _default_graph_agent = QueryBuilderGraph()
    return _default_graph_agent
