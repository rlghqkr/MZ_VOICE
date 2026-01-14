"""
Hybrid RAG Service - 통합 RAG 서비스

QueryRouter를 사용하여 질문 유형에 따라 적절한 RAG 시스템으로 라우팅합니다.
- 법령 관련 질문 → RAGChain + GraphRAG 결합 검색 후 Reranking
- 일반 질문 → RAGChain (Hybrid Search + Reranker)
"""

import logging
from typing import Optional, List, Literal
from dataclasses import dataclass

from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from .chain import RAGChain, RAGResponse
from .graphrag_retriever import GraphRAGRetriever, GraphRAGResponse
from .query_router import QueryRouter, QueryType, RouterResult, quick_law_check
from .prompts import build_system_prompt, RAG_PROMPT_TEMPLATE
from .contextual_retriever import Reranker
from ..emotion import Emotion
from ...config import settings

logger = logging.getLogger(__name__)


@dataclass
class HybridRAGResponse:
    """통합 RAG 응답"""
    answer: str
    source_documents: List[Document]
    emotion: Optional[Emotion] = None
    retrieval_method: str = "hybrid"
    query_type: str = "general"
    router_confidence: str = "high"
    router_reason: str = ""


class HybridRAGService:
    """
    통합 RAG 서비스

    QueryRouter(LLM 기반)를 사용하여 질문을 분류하고,
    - 법령 질문: RAGChain(5개) + GraphRAG(5개) 결합 → Reranking → 상위 5개로 LLM 응답 생성
    - 일반 질문: RAGChain만 사용 (Hybrid Search + Reranker)

    Example:
        service = HybridRAGService()

        # 법령 질문 → RAGChain + GraphRAG 결합
        response = service.query("청년기본법 제3조가 뭐야?")

        # 일반 질문 → RAGChain
        response = service.query("청년 취업 지원 프로그램 알려줘")
    """

    def __init__(
        self,
        # RAGChain 설정
        collection_name: str = None,
        persist_directory: str = None,
        search_type: Literal["similarity", "bm25", "hybrid"] = "hybrid",
        use_reranker: bool = True,
        initial_k: int = 5,
        final_k: int = 5,
        # GraphRAG 설정
        graphrag_root_dir: str = None,
        graphrag_community_level: int = 2,
        graphrag_method: Literal["local", "global"] = "local",
        # Router 설정
        use_quick_check: bool = False,
        router_model: str = "gpt-5.2"
    ):
        """
        Args:
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: ChromaDB 영구 저장 디렉토리
            search_type: RAGChain 검색 방식
            use_reranker: Reranker 사용 여부
            initial_k: 초기 검색 문서 수
            final_k: 최종 사용 문서 수
            graphrag_root_dir: GraphRAG 프로젝트 디렉토리
            graphrag_community_level: GraphRAG 커뮤니티 수준
            graphrag_method: GraphRAG 검색 방법 ("local" 또는 "global")
            use_quick_check: 키워드 사전 필터링 사용 여부
            router_model: QueryRouter LLM 모델
        """
        self.search_type = search_type
        self.use_reranker = use_reranker
        self.initial_k = initial_k
        self.final_k = final_k
        self.graphrag_method = graphrag_method
        self.use_quick_check = use_quick_check

        # 컴포넌트 Lazy Loading
        self._rag_chain = None
        self._graphrag_retriever = None
        self._query_router = None
        self._reranker = None
        self._llm = None

        # 설정 저장
        self._rag_chain_config = {
            "collection_name": collection_name,
            "persist_directory": persist_directory,
            "search_type": search_type,
            "use_reranker": use_reranker,
            "initial_k": initial_k,
            "final_k": final_k
        }
        self._graphrag_config = {
            "root_dir": graphrag_root_dir,
            "community_level": graphrag_community_level
        }
        self._router_model = router_model

    @property
    def rag_chain(self) -> RAGChain:
        """RAGChain 인스턴스 (Lazy Loading)"""
        if self._rag_chain is None:
            self._rag_chain = RAGChain(**self._rag_chain_config)
        return self._rag_chain

    @property
    def graphrag_retriever(self) -> GraphRAGRetriever:
        """GraphRAGRetriever 인스턴스 (Lazy Loading)"""
        if self._graphrag_retriever is None:
            self._graphrag_retriever = GraphRAGRetriever(**self._graphrag_config)
        return self._graphrag_retriever

    @property
    def query_router(self) -> QueryRouter:
        """QueryRouter 인스턴스 (Lazy Loading)"""
        if self._query_router is None:
            self._query_router = QueryRouter(model=self._router_model)
        return self._query_router

    def route_query(self, query: str) -> RouterResult:
        """
        질문 라우팅 (키워드 사전 필터 + LLM 분류)

        Args:
            query: 사용자 질문

        Returns:
            RouterResult: 라우팅 결과
        """
        # 1. 키워드 기반 빠른 체크 (선택적)
        if self.use_quick_check and quick_law_check(query):
            logger.info(f"Quick check: LAW query detected - '{query[:50]}...'")
            return RouterResult(
                query_type=QueryType.LAW,
                confidence="high",
                reason="키워드 기반 법령 질문 감지"
            )

        # 2. LLM 기반 정밀 분류
        result = self.query_router.route(query)
        logger.info(f"Router result: {result.query_type.value} ({result.confidence}) - {result.reason}")

        return result

    def query(
        self,
        question: str,
        emotion: Emotion = None,
        chat_history: List = None,
        force_rag_type: Literal["auto", "general", "law"] = "auto"
    ) -> HybridRAGResponse:
        """
        통합 RAG 쿼리 실행

        Args:
            question: 사용자 질문
            emotion: 감지된 감정 (프롬프트에 반영)
            chat_history: 대화 히스토리
            force_rag_type: 강제 RAG 유형 지정
                - "auto": 자동 라우팅 (기본)
                - "general": 강제로 RAGChain 사용
                - "law": 강제로 GraphRAG 사용

        Returns:
            HybridRAGResponse: 통합 응답
        """
        emotion = emotion or Emotion.NEUTRAL

        # 1. 라우팅 결정
        if force_rag_type == "general":
            query_type = QueryType.GENERAL
            router_result = RouterResult(
                query_type=QueryType.GENERAL,
                confidence="forced",
                reason="강제 일반 RAG 모드"
            )
        elif force_rag_type == "law":
            query_type = QueryType.LAW
            router_result = RouterResult(
                query_type=QueryType.LAW,
                confidence="forced",
                reason="강제 법령 RAG 모드"
            )
        else:
            router_result = self.route_query(question)
            query_type = router_result.query_type

        # 2. 라우팅에 따라 적절한 RAG 실행
        if query_type == QueryType.LAW:
            return self._query_graphrag(
                question=question,
                emotion=emotion,
                router_result=router_result,
                chat_history=chat_history
            )
        else:
            return self._query_ragchain(
                question=question,
                emotion=emotion,
                chat_history=chat_history,
                router_result=router_result
            )

    def _query_ragchain(
        self,
        question: str,
        emotion: Emotion,
        chat_history: List,
        router_result: RouterResult
    ) -> HybridRAGResponse:
        """RAGChain으로 일반 질문 처리"""
        logger.info(f"Using RAGChain for general query: '{question[:50]}...'")

        try:
            response: RAGResponse = self.rag_chain.query(
                question=question,
                emotion=emotion,
                chat_history=chat_history,
                search_type=self.search_type
            )

            return HybridRAGResponse(
                answer=response.answer,
                source_documents=response.source_documents,
                emotion=emotion,
                retrieval_method=response.retrieval_method,
                query_type="general",
                router_confidence=router_result.confidence,
                router_reason=router_result.reason
            )

        except Exception as e:
            logger.error(f"RAGChain query failed: {e}")
            raise

    def _query_graphrag(
        self,
        question: str,
        emotion: Emotion,
        router_result: RouterResult,
        chat_history: List = None
    ) -> HybridRAGResponse:
        """
        GraphRAG + RAGChain 결합으로 법령 질문 처리

        두 검색 소스에서 각각 5개씩 문서를 가져와 Reranking 후
        상위 5개 문서로 LLM 응답을 생성합니다.
        """
        logger.info(f"Using combined RAGChain + GraphRAG for law query: '{question[:50]}...'")

        try:
            # 1. RAGChain에서 5개 문서 검색
            ragchain_docs = self._retrieve_from_ragchain(question, k=5)
            logger.info(f"RAGChain retrieved {len(ragchain_docs)} documents")

            # 2. GraphRAG에서 5개 문서 검색
            graphrag_docs = self._retrieve_from_graphrag(question, k=5)
            logger.info(f"GraphRAG retrieved {len(graphrag_docs)} documents")

            # 3. 두 결과 병합
            combined_docs = ragchain_docs + graphrag_docs
            logger.info(f"Combined {len(combined_docs)} documents")

            # 4. Reranking으로 상위 5개 선별
            if len(combined_docs) > 5:
                reranked_docs = self._rerank_documents(question, combined_docs, top_k=5)
            else:
                reranked_docs = combined_docs

            logger.info(f"After reranking: {len(reranked_docs)} documents")

            # 5. LLM으로 최종 답변 생성
            answer = self._generate_answer(
                question=question,
                documents=reranked_docs,
                emotion=emotion,
                chat_history=chat_history
            )

            return HybridRAGResponse(
                answer=answer,
                source_documents=reranked_docs,
                emotion=emotion,
                retrieval_method="ragchain+graphrag+rerank",
                query_type="law",
                router_confidence=router_result.confidence,
                router_reason=router_result.reason
            )

        except Exception as e:
            logger.error(f"Combined RAG query failed: {e}")
            # 실패 시 RAGChain으로 폴백
            logger.warning("Falling back to RAGChain only...")
            return self._query_ragchain(
                question=question,
                emotion=emotion,
                chat_history=chat_history,
                router_result=RouterResult(
                    query_type=QueryType.GENERAL,
                    confidence="fallback",
                    reason=f"결합 RAG 실패로 폴백: {str(e)}"
                )
            )

    def _retrieve_from_ragchain(self, question: str, k: int = 5) -> List[Document]:
        """RAGChain에서 문서 검색"""
        try:
            retrieval_result = self.rag_chain.contextual_retriever.retrieve(
                query=question,
                initial_k=k,
                final_k=k,
                search_type=self.search_type
            )
            # 메타데이터에 출처 표시
            for doc in retrieval_result.documents:
                doc.metadata["retrieval_source"] = "ragchain"
            return retrieval_result.documents
        except Exception as e:
            logger.error(f"RAGChain retrieval failed: {e}")
            return []

    def _retrieve_from_graphrag(self, question: str, k: int = 5) -> List[Document]:
        """GraphRAG에서 문서 검색"""
        try:
            docs = self.graphrag_retriever.retrieve_documents(
                query=question,
                k=k,
                method=self.graphrag_method
            )
            # 메타데이터에 출처 표시
            for doc in docs:
                doc.metadata["retrieval_source"] = "graphrag"
            return docs
        except Exception as e:
            logger.error(f"GraphRAG retrieval failed: {e}")
            return []

    def _rerank_documents(
        self,
        question: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """문서 Reranking"""
        if self._reranker is None:
            self._reranker = Reranker()

        try:
            return self._reranker.rerank(question, documents, top_k=top_k)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Reranking 실패 시 원본 상위 k개 반환
            return documents[:top_k]

    def _generate_answer(
        self,
        question: str,
        documents: List[Document],
        emotion: Emotion,
        chat_history: List = None
    ) -> str:
        """검색된 문서를 기반으로 LLM 응답 생성"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.llm_model,
                temperature=settings.llm_temperature
            )

        # 컨텍스트 구성
        context = "\n\n".join([doc.page_content for doc in documents])

        # 감정 기반 시스템 프롬프트
        system_prompt = build_system_prompt(emotion)

        # 체인 구성 및 실행
        chain = (
            RAG_PROMPT_TEMPLATE
            | self._llm
            | StrOutputParser()
        )

        chain_input = {
            "system_prompt": system_prompt,
            "context": context,
            "question": question,
            "chat_history": chat_history or []
        }

        answer = chain.invoke(chain_input)

        # 감정에 따른 톤 조정
        return self._adjust_answer_for_emotion(answer, emotion)

    def _adjust_answer_for_emotion(self, answer: str, emotion: Emotion) -> str:
        """
        감정에 따른 답변 톤 조정

        GraphRAG 응답은 기본적으로 중립적이므로,
        필요시 감정에 맞는 인트로/아웃트로 추가
        """
        if emotion == Emotion.NEUTRAL:
            return answer

        emotion_intros = {
            Emotion.ANGRY: "불편을 드려 죄송합니다. ",
            Emotion.HAPPY: "",
            Emotion.SAD: "힘드신 상황이시군요. ",
            Emotion.FEARFUL: "걱정되시는 마음 충분히 이해합니다. ",
            Emotion.SURPRISED: ""
        }

        emotion_outros = {
            Emotion.ANGRY: "\n\n추가로 도움이 필요하시면 말씀해 주세요.",
            Emotion.HAPPY: "",
            Emotion.SAD: "\n\n언제든 더 궁금한 점이 있으시면 편하게 물어봐 주세요.",
            Emotion.FEARFUL: "\n\n더 자세한 설명이 필요하시면 말씀해 주세요.",
            Emotion.SURPRISED: ""
        }

        intro = emotion_intros.get(emotion, "")
        outro = emotion_outros.get(emotion, "")

        return f"{intro}{answer}{outro}"

    def get_stats(self) -> dict:
        """서비스 통계"""
        stats = {
            "service": "HybridRAGService",
            "search_type": self.search_type,
            "use_reranker": self.use_reranker,
            "graphrag_method": self.graphrag_method,
            "use_quick_check": self.use_quick_check,
            "router_model": self._router_model
        }

        # RAGChain 통계 (초기화된 경우)
        if self._rag_chain is not None:
            stats["rag_chain"] = self._rag_chain.get_stats()

        # GraphRAG 통계 (초기화된 경우)
        if self._graphrag_retriever is not None:
            stats["graphrag"] = self._graphrag_retriever.get_stats()

        return stats
