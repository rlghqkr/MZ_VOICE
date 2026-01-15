"""
Contextual Retrieval Implementation

Anthropic의 Contextual Retrieval 방식을 구현합니다.
- Contextual Embeddings: 청크에 문맥 정보를 추가하여 임베딩
- BM25: 키워드 기반 검색
- Hybrid Search: 임베딩 + BM25 결합
- Reranking: 검색 결과 재정렬

Reference: https://www.anthropic.com/engineering/contextual-retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from ...config import settings

logger = logging.getLogger(__name__)


# ============ Context Generation ============

CONTEXT_GENERATION_PROMPT = """<document>
{doc_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. Write in Korean."""


def generate_chunk_context(
    chunk_content: str,
    doc_content: str,
    llm: ChatOpenAI = None
) -> str:
    """
    청크에 대한 문맥 설명을 생성합니다.

    Args:
        chunk_content: 청크 내용
        doc_content: 전체 문서 내용 (또는 관련 섹션)
        llm: LLM 인스턴스

    Returns:
        str: 생성된 문맥 설명 (50-100 토큰)
    """
    if llm is None:
        llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model="gpt-4o-mini",  # 비용 효율적인 모델 사용
            temperature=0
        )

    prompt = CONTEXT_GENERATION_PROMPT.format(
        doc_content=doc_content[:8000],  # 토큰 제한
        chunk_content=chunk_content
    )

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Context generation failed: {e}")
        return ""


def create_contextualized_chunk(context: str, chunk_content: str) -> str:
    """문맥과 청크를 결합하여 contextualized chunk 생성"""
    if context:
        return f"{context}\n\n{chunk_content}"
    return chunk_content


# ============ BM25 Retriever ============

class BM25Retriever:
    """
    BM25 기반 키워드 검색기

    정확한 용어 매칭에 강점이 있어 임베딩 검색을 보완합니다.
    """

    def __init__(self):
        self.documents: List[Document] = []
        self.bm25 = None
        self._tokenized_corpus = []

    def add_documents(self, documents: List[Document]):
        """문서 추가 및 BM25 인덱스 구축"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed. Run: pip install rank-bm25")
            return

        self.documents.extend(documents)

        # 토큰화 (한국어는 공백 기반, 실제로는 형태소 분석기 권장)
        self._tokenized_corpus = [
            self._tokenize(doc.page_content)
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built with {len(self.documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화 (간단한 공백 기반)"""
        # 실제 프로덕션에서는 konlpy 등의 형태소 분석기 사용 권장
        import re
        # 한글, 영문, 숫자만 추출하고 소문자 변환
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text.lower())
        return tokens

    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        BM25 검색 수행

        Returns:
            List of (Document, score) tuples
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built")
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 상위 k개 인덱스
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = [
            (self.documents[i], scores[i])
            for i in top_indices
            if scores[i] > 0
        ]

        return results


# ============ Reranker ============

class Reranker:
    """
    검색 결과 재정렬기

    Cross-encoder 모델을 사용하여 query-document 관련성을 직접 평가합니다.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", eager_loading: bool = True):
        """
        Args:
            model_name: 사용할 reranker 모델
                - "BAAI/bge-reranker-v2-m3": 다국어 지원, 높은 성능
                - "cross-encoder/ms-marco-MiniLM-L-6-v2": 빠른 속도
            eager_loading: True면 즉시 모델 로드
        """
        self.model_name = model_name
        self._model = None
        self._use_llm_rerank = False
        
        if eager_loading:
            logger.info(f"Loading reranker model: {model_name}")
            _ = self.model  # 즉시 로드
            if self._use_llm_rerank:
                logger.warning("Reranker: Using LLM fallback")
            else:
                logger.info("Reranker model initialized")

    @property
    def model(self):
        """Cross-encoder 모델 (Lazy Loading)"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device="cpu") # CPU 사용
                logger.info(f"Loaded reranker model: {self.model_name} (CPU)")
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed. "
                    "Using LLM-based reranking. "
                    "Run: pip install sentence-transformers"
                )
                self._use_llm_rerank = True
            except Exception as e:
                logger.warning(f"Failed to load reranker model: {e}. Using LLM-based reranking.")
                self._use_llm_rerank = True
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 20
    ) -> List[Document]:
        """
        문서 재정렬

        Args:
            query: 사용자 쿼리
            documents: 후보 문서 리스트
            top_k: 반환할 상위 문서 수

        Returns:
            재정렬된 상위 k개 문서
        """
        if not documents:
            return []

        if self._use_llm_rerank or self.model is None:
            return self._llm_rerank(query, documents, top_k)

        # Cross-encoder로 점수 계산
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        # 점수순 정렬
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in doc_scores[:top_k]]

    def _llm_rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int
    ) -> List[Document]:
        """LLM 기반 간단한 재정렬 (fallback)"""
        # 간단히 문서 길이와 키워드 매칭으로 정렬
        query_tokens = set(query.lower().split())

        def score_doc(doc: Document) -> float:
            content = doc.page_content.lower()
            # 키워드 매칭 점수
            match_score = sum(1 for token in query_tokens if token in content)
            # 메타데이터에 keywords가 있으면 추가 점수
            keywords = doc.metadata.get("keywords", "").lower()
            keyword_score = sum(1 for token in query_tokens if token in keywords)
            return match_score + keyword_score * 2

        scored_docs = [(doc, score_doc(doc)) for doc in documents]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:top_k]]


# ============ Hybrid Retriever ============

@dataclass
class RetrievalResult:
    """검색 결과"""
    documents: List[Document]
    scores: Dict[str, List[float]]  # 검색 방법별 점수
    method: str  # 사용된 검색 방법


class ContextualRetriever:
    """
    Contextual Retrieval 통합 검색기

    Anthropic의 Contextual Retrieval 방식을 구현:
    1. Contextual Embeddings (semantic search)
    2. Contextual BM25 (keyword search)
    3. Hybrid Search (결합)
    4. Reranking (재정렬)

    Example:
        retriever = ContextualRetriever()

        # 검색
        results = retriever.retrieve(
            query="청년 취업 지원",
            initial_k=150,
            final_k=20
        )
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        use_reranker: bool = False,
        reranker_model: str = "BAAI/bge-reranker-v2-m3"
    ):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or str(settings.chroma_persist_dir)
        self.use_reranker = use_reranker

        # 컴포넌트 초기화
        self._embeddings = None
        self._vectorstore = None
        self._bm25_retriever = BM25Retriever()
        self._reranker = Reranker(reranker_model) if use_reranker else None

        # BM25용 문서 캐시
        self._documents_loaded = False

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.embedding_model
            )
        return self._embeddings

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            # BM25 인덱스 구축
            self._load_documents_for_bm25()
        return self._vectorstore

    def _load_documents_for_bm25(self):
        """ChromaDB에서 문서를 로드하여 BM25 인덱스 구축"""
        if self._documents_loaded:
            return

        try:
            # ChromaDB에서 모든 문서 가져오기
            collection = self._vectorstore._collection
            result = collection.get(include=["documents", "metadatas"])

            documents = []
            for i, content in enumerate(result["documents"]):
                metadata = result["metadatas"][i] if result["metadatas"] else {}
                documents.append(Document(page_content=content, metadata=metadata))

            if documents:
                self._bm25_retriever.add_documents(documents)
                self._documents_loaded = True
                logger.info(f"Loaded {len(documents)} documents for BM25")
        except Exception as e:
            logger.error(f"Failed to load documents for BM25: {e}")

    def retrieve(
        self,
        query: str,
        initial_k: int = 30,
        final_k: int = 5,
        search_type: str = "hybrid"
    ) -> RetrievalResult:
        """
        Contextual Retrieval 수행

        Args:
            query: 검색 쿼리
            initial_k: 초기 검색에서 가져올 문서 수 (reranking 전)
            final_k: 최종 반환할 문서 수 (reranking 후)
            search_type: "embedding", "bm25", "hybrid" 중 선택

        Returns:
            RetrievalResult: 검색 결과
        """
        scores = {"embedding": [], "bm25": []}

        if search_type == "embedding":
            documents = self._embedding_search(query, initial_k)
        elif search_type == "bm25":
            documents = self._bm25_search(query, initial_k)
        else:  # hybrid
            documents = self._hybrid_search(query, initial_k)

        # Reranking
        if self.use_reranker and self._reranker and len(documents) > final_k:
            documents = self._reranker.rerank(query, documents, final_k)
            method = f"{search_type}+rerank"
        else:
            documents = documents[:final_k]
            method = search_type

        return RetrievalResult(
            documents=documents,
            scores=scores,
            method=method
        )

    def _embedding_search(self, query: str, k: int) -> List[Document]:
        """임베딩 기반 시맨틱 검색"""
        return self.vectorstore.similarity_search(query, k=k)

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """BM25 키워드 검색"""
        results = self._bm25_retriever.search(query, k=k)
        return [doc for doc, score in results]

    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """
        Hybrid Search: 임베딩 + BM25 결합

        Reciprocal Rank Fusion (RRF)을 사용하여 결과 결합
        """
        # 각 방법으로 검색
        embedding_docs = self._embedding_search(query, k)
        bm25_results = self._bm25_retriever.search(query, k)

        # RRF (Reciprocal Rank Fusion)
        rrf_k = 60  # RRF 상수
        doc_scores: Dict[str, Tuple[Document, float]] = {}

        # 임베딩 결과 점수 계산
        for rank, doc in enumerate(embedding_docs):
            doc_id = doc.page_content[:100]  # 간단한 ID
            score = 1 / (rrf_k + rank + 1)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + score)
            else:
                doc_scores[doc_id] = (doc, score)

        # BM25 결과 점수 계산
        for rank, (doc, bm25_score) in enumerate(bm25_results):
            doc_id = doc.page_content[:100]
            score = 1 / (rrf_k + rank + 1)
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + score)
            else:
                doc_scores[doc_id] = (doc, score)

        # 점수순 정렬
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, score in sorted_docs[:k]]

    def as_retriever(self, **kwargs):
        """LangChain Retriever 인터페이스 반환"""
        return ContextualRetrieverWrapper(self, **kwargs)


class ContextualRetrieverWrapper:
    """LangChain Retriever 인터페이스 래퍼"""

    def __init__(
        self,
        contextual_retriever: ContextualRetriever,
        initial_k: int = 100,
        final_k: int = 20,
        search_type: str = "hybrid"
    ):
        self.contextual_retriever = contextual_retriever
        self.initial_k = initial_k
        self.final_k = final_k
        self.search_type = search_type

    def invoke(self, query: str) -> List[Document]:
        """검색 수행"""
        result = self.contextual_retriever.retrieve(
            query=query,
            initial_k=self.initial_k,
            final_k=self.final_k,
            search_type=self.search_type
        )
        return result.documents

    def get_relevant_documents(self, query: str) -> List[Document]:
        """LangChain 호환 메서드"""
        return self.invoke(query)
