"""
Hybrid RAG Service - 통합 RAG 서비스

QueryRouter를 사용하여 질문 유형에 따라 적절한 RAG 시스템으로 라우팅합니다.
- 법령 관련 질문 → LawRAGChain (법령 전용 ChromaDB)
- 일반 질문 → RAGChain (일반 정책 ChromaDB)
"""

import logging
import os
from typing import Optional, List, Literal, Generator
from dataclasses import dataclass

from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from .chain import RAGChain, RAGResponse
from .query_router import QueryRouter, QueryType, RouterResult, quick_law_check
from .prompts import build_system_prompt, RAG_PROMPT_TEMPLATE
from .contextual_retriever import Reranker
from ..emotion import Emotion
from ...config import settings
from ...utils.logging_utils import log_prompt, log_llm_response

logger = logging.getLogger(__name__)

# 환경변수로 프롬프트 로깅 활성화 여부 제어
ENABLE_PROMPT_LOGGING = os.getenv("ENABLE_PROMPT_LOGGING", "false").lower() == "true"


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
    - 법령 질문: LawRAGChain (법령 전용 ChromaDB)
    - 일반 질문: RAGChain (일반 정책 ChromaDB)

    Example:
        service = HybridRAGService()

        # 법령 질문 → LawRAGChain
        response = service.query("청년기본법 제3조가 뭐야?")

        # 일반 질문 → RAGChain
        response = service.query("청년 취업 지원 프로그램 알려줘")
    """

    def __init__(
        self,
        # RAGChain 설정 (일반 정책)
        collection_name: str = None,
        persist_directory: str = None,
        # LawRAGChain 설정 (법령)
        law_collection_name: str = None,
        law_persist_directory: str = None,
        # 공통 설정
        search_type: Literal["similarity", "bm25", "hybrid"] = "hybrid",
        use_reranker: bool = True,
        initial_k: int = 5,
        final_k: int = 3,
        # Router 설정
        use_quick_check: bool = False,
        router_model: str = "gpt-4o-mini",
        # 로딩 설정
        eager_loading: bool = True
    ):
        """
        Args:
            collection_name: 일반 정책 ChromaDB 컬렉션 이름
            persist_directory: 일반 정책 ChromaDB 영구 저장 디렉토리
            law_collection_name: 법령 ChromaDB 컬렉션 이름
            law_persist_directory: 법령 ChromaDB 영구 저장 디렉토리
            search_type: RAGChain 검색 방식
            use_reranker: Reranker 사용 여부
            initial_k: 초기 검색 문서 수
            final_k: 최종 사용 문서 수
            use_quick_check: 키워드 사전 필터링 사용 여부
            router_model: QueryRouter LLM 모델
            eager_loading: True면 즉시 로딩, False면 Lazy Loading
        """
        self.search_type = search_type
        self.use_reranker = use_reranker
        self.initial_k = initial_k
        self.final_k = final_k
        self.use_quick_check = use_quick_check
        self.eager_loading = eager_loading

        # 컴포넌트 초기화
        self._rag_chain = None
        self._law_rag_chain = None
        self._query_router = None
        self._reranker = None
        self._llm = None

        # 일반 정책 RAGChain 설정
        self._rag_chain_config = {
            "collection_name": collection_name,
            "persist_directory": persist_directory,
            "search_type": search_type,
            "use_reranker": use_reranker,
            "initial_k": initial_k,
            "final_k": final_k,
            "eager_loading": eager_loading
        }

        # 법령 RAGChain 설정
        self._law_rag_chain_config = {
            "collection_name": law_collection_name or settings.chroma_law_collection_name,
            "persist_directory": law_persist_directory or str(settings.chroma_law_persist_dir),
            "search_type": search_type,
            "use_reranker": use_reranker,
            "initial_k": initial_k,
            "final_k": final_k,
            "eager_loading": eager_loading
        }

        self._router_model = router_model

        # Eager Loading: 즉시 컴포넌트 로드
        if eager_loading:
            self._init_all()

    def _init_all(self):
        """모든 컴포넌트 즉시 초기화"""
        logger.info("Initializing HybridRAGService components")

        _ = self.rag_chain  # RAGChain (일반 정책)
        logger.info("RAGChain (general) initialized")

        _ = self.law_rag_chain  # LawRAGChain (법령)
        logger.info("LawRAGChain (law) initialized")

        _ = self.query_router  # QueryRouter
        logger.info("QueryRouter initialized")

        # 법령 질문용 별도 Reranker
        if settings.enable_reranking:
            from .contextual_retriever import Reranker
            logger.info("Initializing reranker for HybridRAG")
            self._reranker = Reranker(eager_loading=True)
        else:
            logger.info("Reranking disabled (ENABLE_RERANKING=false)")

        logger.info("HybridRAGService initialization completed")

    @property
    def rag_chain(self) -> RAGChain:
        """일반 정책용 RAGChain 인스턴스 (Lazy Loading)"""
        if self._rag_chain is None:
            self._rag_chain = RAGChain(**self._rag_chain_config)
        return self._rag_chain

    @property
    def law_rag_chain(self) -> RAGChain:
        """법령용 RAGChain 인스턴스 (Lazy Loading)"""
        if self._law_rag_chain is None:
            self._law_rag_chain = RAGChain(**self._law_rag_chain_config)
        return self._law_rag_chain

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
        # 1. LAW 키워드 기반 빠른 체크 (선택적)
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
                - "law": 강제로 LawRAGChain 사용

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
            return self._query_law(
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

    def _query_law(
        self,
        question: str,
        emotion: Emotion,
        router_result: RouterResult,
        chat_history: List = None
    ) -> HybridRAGResponse:
        """
        LawRAGChain으로 법령 질문 처리

        법령 전용 ChromaDB에서 문서를 검색하여 응답을 생성합니다.
        """
        logger.info(f"Using LawRAGChain for law query: '{question[:50]}...'")

        try:
            # 1. LawRAGChain에서 문서 검색
            law_docs = self._retrieve_from_law_ragchain(question, k=self.final_k)
            logger.info(f"LawRAGChain retrieved {len(law_docs)} documents")

            # 2. Reranking (옵션)
            if settings.enable_reranking and len(law_docs) > self.final_k:
                reranked_docs = self._rerank_documents(question, law_docs, top_k=self.final_k)
                logger.info(f"After reranking: {len(reranked_docs)} documents")
            else:
                reranked_docs = law_docs[:self.final_k]
                if not settings.enable_reranking:
                    logger.info(f"Reranking disabled, using first {len(reranked_docs)} documents")

            # 3. LLM으로 최종 답변 생성
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
                retrieval_method="law_ragchain",
                query_type="law",
                router_confidence=router_result.confidence,
                router_reason=router_result.reason
            )

        except Exception as e:
            logger.error(f"LawRAGChain query failed: {e}")
            # 실패 시 일반 RAGChain으로 폴백
            logger.warning("Falling back to RAGChain...")
            return self._query_ragchain(
                question=question,
                emotion=emotion,
                chat_history=chat_history,
                router_result=RouterResult(
                    query_type=QueryType.GENERAL,
                    confidence="fallback",
                    reason=f"법령 RAG 실패로 폴백: {str(e)}"
                )
            )

    def _retrieve_from_law_ragchain(self, question: str, k: int = 5) -> List[Document]:
        """LawRAGChain에서 문서 검색"""
        try:
            retrieval_result = self.law_rag_chain.contextual_retriever.retrieve(
                query=question,
                initial_k=k,
                final_k=k,
                search_type=self.search_type
            )
            # 메타데이터에 출처 표시
            for doc in retrieval_result.documents:
                doc.metadata["retrieval_source"] = "law_ragchain"
            return retrieval_result.documents
        except Exception as e:
            logger.error(f"LawRAGChain retrieval failed: {e}")
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

        # 프롬프트 로깅
        if ENABLE_PROMPT_LOGGING:
            try:
                formatted_prompt = RAG_PROMPT_TEMPLATE.format(**chain_input)
                log_prompt(
                    logger=logger,
                    prompt_name="Hybrid RAG (Law Query)",
                    prompt_text=formatted_prompt,
                    user_input=question
                )
            except Exception as e:
                logger.debug(f"Failed to log prompt: {e}")

        answer = chain.invoke(chain_input)

        # 응답 로깅
        if ENABLE_PROMPT_LOGGING:
            log_llm_response(logger, answer, "Hybrid RAG Answer")

        # 감정에 따른 톤 조정
        return self._adjust_answer_for_emotion(answer, emotion)

    def _adjust_answer_for_emotion(self, answer: str, emotion: Emotion) -> str:
        """
        감정에 따른 답변 톤 조정

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
            "use_quick_check": self.use_quick_check,
            "router_model": self._router_model
        }

        # RAGChain 통계 (일반 정책)
        if self._rag_chain is not None:
            stats["rag_chain"] = self._rag_chain.get_stats()

        # LawRAGChain 통계 (법령)
        if self._law_rag_chain is not None:
            stats["law_rag_chain"] = self._law_rag_chain.get_stats()

        return stats

    def query_stream(
        self,
        question: str,
        emotion: Emotion = None,
        chat_history: List = None,
        force_rag_type: Literal["auto", "general", "law"] = "auto"
    ) -> Generator[str, None, None]:
        """
        스트리밍 RAG 쿼리 실행

        LLM 응답을 청크 단위로 yield합니다.

        Args:
            question: 사용자 질문
            emotion: 감지된 감정 (프롬프트에 반영)
            chat_history: 대화 히스토리
            force_rag_type: 강제 RAG 유형 지정

        Yields:
            str: 응답 텍스트 청크
        """
        emotion = emotion or Emotion.NEUTRAL

        # 1. 라우팅 결정
        if force_rag_type == "general":
            query_type = QueryType.GENERAL
        elif force_rag_type == "law":
            query_type = QueryType.LAW
        else:
            router_result = self.route_query(question)
            query_type = router_result.query_type

        # 2. 라우팅에 따라 적절한 RAG 스트리밍 실행
        if query_type == QueryType.LAW:
            yield from self._query_law_stream(
                question=question,
                emotion=emotion,
                chat_history=chat_history
            )
        else:
            yield from self.rag_chain.query_stream(
                question=question,
                emotion=emotion,
                chat_history=chat_history,
                search_type=self.search_type
            )

    def _query_law_stream(
        self,
        question: str,
        emotion: Emotion,
        chat_history: List = None
    ) -> Generator[str, None, None]:
        """법령 RAG 스트리밍 쿼리"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.llm_model,
                temperature=settings.llm_temperature
            )

        # 1. 법령 문서 검색
        law_docs = self._retrieve_from_law_ragchain(question, k=self.final_k)
        logger.info(f"LawRAGChain retrieved {len(law_docs)} documents for streaming")

        # 2. 컨텍스트 구성
        context = "\n\n".join([doc.page_content for doc in law_docs])

        # 3. 감정 기반 시스템 프롬프트
        system_prompt = build_system_prompt(emotion)

        # 4. 스트리밍 응답 생성
        chain = RAG_PROMPT_TEMPLATE | self._llm

        chain_input = {
            "system_prompt": system_prompt,
            "context": context,
            "question": question,
            "chat_history": chat_history or []
        }

        for chunk in chain.stream(chain_input):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
