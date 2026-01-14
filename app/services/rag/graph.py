"""
LangGraph RAG Workflow Implementation

LangGraph를 사용한 고급 RAG 워크플로우입니다.
조건부 분기, 쿼리 재작성, 응답 검증 등을 지원합니다.
"""

import logging
from typing import TypedDict, Annotated, List, Optional
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .chain import RAGChain
from .prompts import build_system_prompt, QUERY_REWRITE_PROMPT
from ..emotion import Emotion
from ...config import settings

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """LangGraph 상태 정의"""
    # 입력
    question: str                          # 사용자 질문
    emotion: Optional[Emotion]             # 감지된 감정
    
    # 처리 중
    rewritten_query: Optional[str]         # 재작성된 쿼리
    context: Optional[str]                 # 검색된 컨텍스트
    
    # 출력
    answer: Optional[str]                  # 최종 답변
    
    # 메타
    messages: Annotated[list, add_messages]  # 대화 히스토리
    retry_count: int                        # 재시도 횟수


class RAGGraph:
    """
    LangGraph 기반 RAG 워크플로우
    
    단순 RAG 체인보다 복잡한 로직을 처리할 수 있습니다:
    - 쿼리 재작성
    - 조건부 검색
    - 응답 검증 및 재생성
    
    Workflow:
        질문 → [쿼리 분석] → [문서 검색] → [응답 생성] → [응답 검증] → 답변
                    ↓                                        ↓
               [쿼리 재작성]                          [재생성 필요?]
                                                         ↓
                                                    [응답 재생성]
    
    Example:
        graph = RAGGraph()
        result = graph.invoke({
            "question": "환불 정책이 뭐예요?",
            "emotion": Emotion.ANGRY
        })
        print(result["answer"])
    """
    
    def __init__(self, rag_chain: RAGChain = None):
        """
        Args:
            rag_chain: RAG 체인 인스턴스 (없으면 새로 생성)
        """
        self.rag_chain = rag_chain or RAGChain()
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=0.3  # 분석용은 낮은 temperature
        )
        
        self._graph = None
    
    @property
    def graph(self):
        """컴파일된 그래프 (Lazy Loading)"""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph
    
    def _build_graph(self) -> StateGraph:
        """워크플로우 그래프 구성"""
        
        # 그래프 정의
        workflow = StateGraph(GraphState)
        
        # 노드 추가
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("validate_response", self._validate_response)
        
        # 엣지 정의
        workflow.set_entry_point("analyze_query")
        
        # 쿼리 분석 후 분기
        workflow.add_conditional_edges(
            "analyze_query",
            self._should_rewrite,
            {
                "rewrite": "rewrite_query",
                "continue": "retrieve_context"
            }
        )
        
        workflow.add_edge("rewrite_query", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "validate_response")
        
        # 검증 후 분기
        workflow.add_conditional_edges(
            "validate_response",
            self._should_retry,
            {
                "retry": "generate_response",
                "end": END
            }
        )
        
        return workflow.compile()
    
    # ============ 노드 함수들 ============
    
    def _analyze_query(self, state: GraphState) -> GraphState:
        """쿼리 분석 노드"""
        question = state["question"]
        
        # 간단한 쿼리인지 복잡한 쿼리인지 분석
        is_simple = len(question) < 20 and "?" in question
        
        logger.debug(f"Query analyzed: {question[:50]}... (simple: {is_simple})")
        
        return {
            **state,
            "needs_rewrite": not is_simple,
            "retry_count": 0
        }
    
    def _rewrite_query(self, state: GraphState) -> GraphState:
        """쿼리 재작성 노드"""
        question = state["question"]
        
        # 쿼리 재작성
        chain = QUERY_REWRITE_PROMPT | self.llm
        result = chain.invoke({"question": question})
        
        rewritten = result.content.strip()
        logger.debug(f"Query rewritten: {question} → {rewritten}")
        
        return {
            **state,
            "rewritten_query": rewritten
        }
    
    def _retrieve_context(self, state: GraphState) -> GraphState:
        """문서 검색 노드"""
        query = state.get("rewritten_query") or state["question"]
        
        # RAG 체인의 retriever 사용
        docs = self.rag_chain.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        logger.debug(f"Retrieved {len(docs)} documents")
        
        return {
            **state,
            "context": context
        }
    
    def _generate_response(self, state: GraphState) -> GraphState:
        """응답 생성 노드"""
        question = state["question"]
        context = state.get("context", "")
        emotion = state.get("emotion", Emotion.NEUTRAL)
        
        # 감정 기반 시스템 프롬프트
        system_prompt = build_system_prompt(emotion)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
컨텍스트:
{context}

질문: {question}

위 컨텍스트를 참고하여 답변해주세요.
""")
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            **state,
            "answer": response.content
        }
    
    def _validate_response(self, state: GraphState) -> GraphState:
        """응답 검증 노드"""
        answer = state.get("answer", "")
        
        # 간단한 검증: 답변이 너무 짧거나 "모르겠습니다"만 있는지
        is_valid = len(answer) > 20 and "모르겠습니다" not in answer[:50]
        
        return {
            **state,
            "is_valid": is_valid,
            "retry_count": state.get("retry_count", 0) + 1
        }
    
    # ============ 조건부 엣지 함수들 ============
    
    def _should_rewrite(self, state: GraphState) -> str:
        """쿼리 재작성 필요 여부"""
        if state.get("needs_rewrite", False):
            return "rewrite"
        return "continue"
    
    def _should_retry(self, state: GraphState) -> str:
        """재시도 필요 여부"""
        is_valid = state.get("is_valid", True)
        retry_count = state.get("retry_count", 0)
        
        if not is_valid and retry_count < 2:
            return "retry"
        return "end"
    
    # ============ 공개 메서드 ============
    
    def invoke(
        self,
        question: str,
        emotion: Emotion = None,
        **kwargs
    ) -> GraphState:
        """
        워크플로우 실행
        
        Args:
            question: 사용자 질문
            emotion: 감지된 감정
            
        Returns:
            GraphState: 최종 상태 (answer 포함)
        """
        initial_state: GraphState = {
            "question": question,
            "emotion": emotion or Emotion.NEUTRAL,
            "rewritten_query": None,
            "context": None,
            "answer": None,
            "messages": [],
            "retry_count": 0
        }
        
        result = self.graph.invoke(initial_state)
        return result
    
    async def ainvoke(
        self,
        question: str,
        emotion: Emotion = None,
        **kwargs
    ) -> GraphState:
        """비동기 워크플로우 실행"""
        initial_state: GraphState = {
            "question": question,
            "emotion": emotion or Emotion.NEUTRAL,
            "rewritten_query": None,
            "context": None,
            "answer": None,
            "messages": [],
            "retry_count": 0
        }
        
        result = await self.graph.ainvoke(initial_state)
        return result
    
    def stream(
        self,
        question: str,
        emotion: Emotion = None
    ):
        """스트리밍 실행 (각 노드 결과 yield)"""
        initial_state: GraphState = {
            "question": question,
            "emotion": emotion or Emotion.NEUTRAL,
            "rewritten_query": None,
            "context": None,
            "answer": None,
            "messages": [],
            "retry_count": 0
        }
        
        for event in self.graph.stream(initial_state):
            yield event
    
    def visualize(self) -> str:
        """그래프 시각화 (Mermaid 형식)"""
        return self.graph.get_graph().draw_mermaid()
