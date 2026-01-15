"""
RAG (Retrieval-Augmented Generation) Service Module

LangChain과 LangGraph 기반 RAG 서비스 모듈입니다.

Contextual Retrieval 지원:
- Hybrid Search (Embedding + BM25)
- Reranking (Cross-encoder)

Hybrid RAG 지원:
- QueryRouter: LLM 기반 질문 분류 (법령/일반)
- LawRAGChain: 법령 전용 ChromaDB 검색
- HybridRAGService: 통합 RAG 서비스

Query Builder:
- QueryBuilderGraph: LangGraph 기반 정보 수집 에이전트
"""

from .prompts import (
    build_system_prompt,
    get_emotion_context,
    RAG_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT
)
from .chain import RAGChain, RAGResponse, ContextualIndexer
from .graph import RAGGraph, GraphState
from .contextual_retriever import (
    ContextualRetriever,
    RetrievalResult,
    BM25Retriever,
    Reranker,
    generate_chunk_context,
    create_contextualized_chunk
)
from .query_router import QueryRouter, QueryType, RouterResult, quick_law_check
from .hybrid_rag_service import HybridRAGService, HybridRAGResponse
from .query_builder_graph import (
    QueryBuilderGraph,
    GraphAgentResponse,
    UserProfile,
    ConversationPhase,
    QueryBuilderState,
    get_query_builder_graph
)

__all__ = [
    # Prompts
    "build_system_prompt",
    "get_emotion_context",
    "RAG_PROMPT_TEMPLATE",
    "QUERY_REWRITE_PROMPT",
    # RAG Chain
    "RAGChain",
    "RAGResponse",
    "ContextualIndexer",
    # RAG Graph
    "RAGGraph",
    "GraphState",
    # Contextual Retrieval
    "ContextualRetriever",
    "RetrievalResult",
    "BM25Retriever",
    "Reranker",
    "generate_chunk_context",
    "create_contextualized_chunk",
    # Query Router
    "QueryRouter",
    "QueryType",
    "RouterResult",
    "quick_law_check",
    # Hybrid RAG Service
    "HybridRAGService",
    "HybridRAGResponse",
    # Query Builder Graph (LangGraph)
    "QueryBuilderGraph",
    "GraphAgentResponse",
    "UserProfile",
    "ConversationPhase",
    "QueryBuilderState",
    "get_query_builder_graph",
]
