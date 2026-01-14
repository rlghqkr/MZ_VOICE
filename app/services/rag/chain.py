"""
LangChain RAG Chain Implementation

LangChain을 사용한 RAG 체인 구현입니다.
감정 기반 동적 프롬프트를 지원합니다.

Contextual Retrieval 지원:
- Hybrid Search (Embedding + BM25)
- Reranking
"""

import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from .prompts import build_system_prompt, RAG_PROMPT_TEMPLATE
from .contextual_retriever import (
    ContextualRetriever,
    RetrievalResult,
    generate_chunk_context,
    create_contextualized_chunk
)
from ..emotion import Emotion
from ...config import settings
from ...repositories import VectorStoreRepository

logger = logging.getLogger(__name__)


# ============ Contextual Indexing ============

class ContextualIndexer:
    """
    Contextual Retrieval을 위한 문서 인덱서

    각 청크에 LLM으로 문맥 정보를 추가하여 검색 정확도를 향상시킵니다.

    Reference: https://www.anthropic.com/engineering/contextual-retrieval
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        use_contextual: bool = True,
        batch_size: int = 10
    ):
        """
        Args:
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: 영구 저장 디렉토리
            use_contextual: Contextual Processing 사용 여부
            batch_size: 배치 처리 크기 (API 호출 최적화)
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or str(settings.chroma_persist_dir)
        self.use_contextual = use_contextual
        self.batch_size = batch_size

        self._llm = None
        self._embeddings = None
        self._vectorstore = None

    @property
    def llm(self) -> ChatOpenAI:
        """Context 생성용 LLM (비용 효율적인 모델)"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model="gpt-4o-mini",
                temperature=0
            )
        return self._llm

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.embedding_model
            )
        return self._embeddings

    def clear_collection(self):
        """기존 컬렉션 삭제"""
        import chromadb

        client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception as e:
            logger.info(f"No existing collection to delete: {e}")

    def index_from_json_directory(
        self,
        directory: str,
        content_field: str = "content",
        show_progress: bool = True
    ) -> int:
        """
        디렉토리 내 모든 JSON 파일을 Contextual Processing하여 인덱싱

        Args:
            directory: JSON 파일들이 있는 디렉토리
            content_field: 청크 내용 필드명
            show_progress: 진행 상황 표시 여부

        Returns:
            인덱싱된 문서 수
        """
        from pathlib import Path

        json_dir = Path(directory)
        if not json_dir.exists():
            logger.error(f"Directory not found: {directory}")
            return 0

        json_files = list(json_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {directory}")

        all_docs = []
        total_chunks = 0

        for file_idx, json_file in enumerate(json_files):
            if show_progress:
                logger.info(f"[{file_idx + 1}/{len(json_files)}] Processing: {json_file.name}")

            try:
                docs = self._process_json_file(json_file, content_field, show_progress)
                all_docs.extend(docs)
                total_chunks += len(docs)
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {e}")

        if all_docs:
            # 벡터 저장소에 저장
            self._vectorstore = Chroma.from_documents(
                documents=all_docs,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            logger.info(f"Indexed {len(all_docs)} contextualized documents")

        return total_chunks

    def _process_json_file(
        self,
        json_file: Path,
        content_field: str,
        show_progress: bool
    ) -> List[Document]:
        """단일 JSON 파일 처리"""
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 문서 구조 파악
        if isinstance(data, dict) and "chunks" in data:
            # chunks 배열 구조
            source_doc = data.get("source_document", {})
            chunks = data.get("chunks", [])

            # 전체 문서 내용 구성 (문맥 생성용)
            full_doc_content = self._build_full_document_content(source_doc, chunks)

            return self._process_chunks(
                chunks,
                source_doc,
                full_doc_content,
                json_file.name,
                content_field,
                show_progress
            )
        elif isinstance(data, list):
            # 리스트 구조
            return self._process_list_documents(data, json_file.name, content_field, show_progress)
        else:
            # 단일 문서
            return self._process_single_document(data, json_file.name, content_field)

    def _build_full_document_content(
        self,
        source_doc: Dict,
        chunks: List[Dict]
    ) -> str:
        """전체 문서 내용 구성 (Context 생성용)"""
        parts = []

        # 문서 메타 정보
        if source_doc.get("source_name"):
            parts.append(f"문서명: {source_doc['source_name']}")
        if source_doc.get("source_type"):
            parts.append(f"분류: {source_doc['source_type']}")

        parts.append("\n--- 문서 내용 ---\n")

        # 모든 청크 내용 결합
        for chunk in chunks:
            content = chunk.get("content", "")
            if content:
                parts.append(content)

        return "\n\n".join(parts)

    def _process_chunks(
        self,
        chunks: List[Dict],
        source_doc: Dict,
        full_doc_content: str,
        source_file: str,
        content_field: str,
        show_progress: bool
    ) -> List[Document]:
        """청크 리스트 처리 (Contextual Processing 적용)"""
        documents = []

        for i, chunk in enumerate(chunks):
            content = chunk.get(content_field, "")
            if not content:
                continue

            # Contextual Processing
            if self.use_contextual:
                if show_progress and (i + 1) % 5 == 0:
                    logger.info(f"  Generating context for chunk {i + 1}/{len(chunks)}")

                context = generate_chunk_context(
                    chunk_content=content,
                    doc_content=full_doc_content,
                    llm=self.llm
                )
                contextualized_content = create_contextualized_chunk(context, content)
            else:
                contextualized_content = content

            # 메타데이터 구성
            metadata = self._build_metadata(chunk, source_doc, source_file)

            # 원본 content도 메타데이터에 저장 (검색 결과 표시용)
            metadata["original_content"] = content
            metadata["has_context"] = self.use_contextual and bool(context)

            documents.append(Document(
                page_content=contextualized_content,
                metadata=metadata
            ))

        return documents

    def _process_list_documents(
        self,
        data: List[Dict],
        source_file: str,
        content_field: str,
        show_progress: bool
    ) -> List[Document]:
        """리스트 형태 문서 처리"""
        documents = []

        # 전체 문서 내용 구성
        full_content = "\n\n".join([
            item.get(content_field, "") for item in data if item.get(content_field)
        ])

        for i, item in enumerate(data):
            content = item.get(content_field, "")
            if not content:
                continue

            if self.use_contextual:
                if show_progress and (i + 1) % 5 == 0:
                    logger.info(f"  Generating context for item {i + 1}/{len(data)}")

                context = generate_chunk_context(
                    chunk_content=content,
                    doc_content=full_content,
                    llm=self.llm
                )
                contextualized_content = create_contextualized_chunk(context, content)
            else:
                contextualized_content = content

            metadata = {
                "doc_id": item.get("doc_id", f"doc_{i}"),
                "source_file": source_file,
                "original_content": content,
                "has_context": self.use_contextual
            }

            # 추가 메타데이터 병합
            for key in ["source_type", "source_name", "keywords"]:
                if key in item:
                    val = item[key]
                    if isinstance(val, (list, dict)):
                        metadata[key] = json.dumps(val, ensure_ascii=False)
                    else:
                        metadata[key] = val

            documents.append(Document(
                page_content=contextualized_content,
                metadata=metadata
            ))

        return documents

    def _process_single_document(
        self,
        data: Dict,
        source_file: str,
        content_field: str
    ) -> List[Document]:
        """단일 문서 처리"""
        content = data.get(content_field, "")
        if not content:
            return []

        metadata = {
            "doc_id": data.get("doc_id", "single_doc"),
            "source_file": source_file,
            "original_content": content,
            "has_context": False  # 단일 문서는 context 불필요
        }

        return [Document(page_content=content, metadata=metadata)]

    def _build_metadata(
        self,
        chunk: Dict,
        source_doc: Dict,
        source_file: str
    ) -> Dict[str, Any]:
        """메타데이터 구성"""
        metadata = {
            "chunk_id": chunk.get("chunk_id", ""),
            "parent_doc_id": chunk.get("parent_doc_id", source_doc.get("doc_id", "")),
            "chunk_type": chunk.get("chunk_type", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "source_service": chunk.get("source_service", source_doc.get("source_type", "")),
            "category": chunk.get("category") or "",
            "subcategory": chunk.get("subcategory") or "",
            "source_file": source_file,
        }

        # keywords 처리
        keywords = chunk.get("keywords", [])
        if isinstance(keywords, list):
            metadata["keywords"] = json.dumps(keywords, ensure_ascii=False)
        elif keywords:
            metadata["keywords"] = str(keywords)
        else:
            metadata["keywords"] = ""

        # sample_questions 처리
        sample_q = chunk.get("sample_questions", [])
        if isinstance(sample_q, list):
            metadata["sample_questions"] = json.dumps(sample_q, ensure_ascii=False)
        else:
            metadata["sample_questions"] = ""

        return metadata


@dataclass
class RAGResponse:
    """RAG 응답 데이터 클래스"""
    answer: str                           # 생성된 답변
    source_documents: List[Document]      # 참조 문서
    emotion: Optional[Emotion] = None     # 적용된 감정
    retrieval_method: str = "similarity"  # 사용된 검색 방법


class RAGChain:
    """
    LangChain 기반 RAG 체인

    감정 인식 결과를 프롬프트에 반영하여
    고객 감정에 맞는 응답을 생성합니다.

    Contextual Retrieval 지원:
    - search_type="similarity": 기존 임베딩 검색
    - search_type="bm25": BM25 키워드 검색
    - search_type="hybrid": 임베딩 + BM25 결합
    - use_reranker=True: 검색 결과 재정렬

    Example:
        # 기존 방식
        rag = RAGChain()
        response = rag.query("청년 정책이 뭐가 있어요?")

        # Contextual Retrieval 방식
        rag = RAGChain(
            search_type="hybrid",
            use_reranker=True
        )
        response = rag.query("청년 정책이 뭐가 있어요?")
    """

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        search_type: Literal["similarity", "bm25", "hybrid"] = "hybrid",
        use_reranker: bool = True,
        initial_k: int = 5,
        final_k: int = 5
    ):
        """
        Args:
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: 영구 저장 디렉토리
            search_type: 검색 방식 ("similarity", "bm25", "hybrid")
            use_reranker: Reranker 사용 여부
            initial_k: 초기 검색 문서 수 (reranking 전)
            final_k: 최종 사용 문서 수 (reranking 후)
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or str(settings.chroma_persist_dir)
        self.search_type = search_type
        self.use_reranker = use_reranker
        self.initial_k = initial_k
        self.final_k = final_k

        # 컴포넌트 초기화
        self._llm = None
        self._embeddings = None
        self._vectorstore = None
        self._retriever = None
        self._contextual_retriever = None

    @property
    def llm(self) -> ChatOpenAI:
        """LLM 인스턴스 (Lazy Loading)"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.llm_model,
                temperature=settings.llm_temperature
            )
        return self._llm

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """임베딩 모델 (Lazy Loading)"""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                model=settings.embedding_model
            )
        return self._embeddings

    @property
    def vectorstore(self) -> Chroma:
        """벡터 저장소 (Lazy Loading)"""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self._vectorstore

    @property
    def retriever(self):
        """검색기 (기존 similarity 검색용)"""
        if self._retriever is None:
            self._retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.final_k}
            )
        return self._retriever

    @property
    def contextual_retriever(self) -> ContextualRetriever:
        """Contextual Retriever (Hybrid + Rerank)"""
        if self._contextual_retriever is None:
            self._contextual_retriever = ContextualRetriever(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                use_reranker=self.use_reranker
            )
        return self._contextual_retriever

    def load_from_json(
        self,
        file_path: str,
        content_field: str = "content"
    ):
        """
        JSON 파일에서 문서 로드 (청킹 없이 문서 전체 유지)

        Args:
            file_path: JSON 파일 경로
            content_field: 텍스트 내용이 담긴 필드명 (기본: "content")

        JSON 형식 예시:
            [
                {
                    "doc_id": "...",
                    "content": "실제 문서 내용",
                    "metadata": {"키워드": "..."}
                },
                ...
            ]
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 단일 객체인 경우 리스트로 변환
        if isinstance(data, dict):
            data = [data]

        docs = []

        for item in data:
            # content 필드 추출
            content = item.get(content_field, "")
            if not content:
                logger.warning(f"Empty content in document: {item.get('doc_id', 'unknown')}")
                continue

            # 메타데이터 구성
            metadata = {
                "doc_id": item.get("doc_id", ""),
                "source_type": item.get("source_type", ""),
                "source_name": item.get("source_name", ""),
                "source_file": item.get("source_file", Path(file_path).name),
            }

            # 중첩된 metadata 필드가 있으면 병합
            if "metadata" in item and isinstance(item["metadata"], dict):
                metadata.update(item["metadata"])

            # ChromaDB는 리스트/딕셔너리 메타데이터 지원 안함 - 문자열로 변환
            cleaned_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (list, dict)):
                    cleaned_metadata[k] = json.dumps(v, ensure_ascii=False)
                elif v is None:
                    cleaned_metadata[k] = ""
                else:
                    cleaned_metadata[k] = v

            docs.append(Document(page_content=content, metadata=cleaned_metadata))

        if docs:
            # 청킹 없이 바로 벡터 저장소에 추가
            self._vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            self._retriever = None
            logger.info(f"Loaded {len(docs)} documents from JSON (no chunking): {file_path}")
        else:
            logger.warning(f"No valid documents found in: {file_path}")

    def load_from_json_directory(
        self,
        directory: str,
        content_field: str = "content"
    ):
        """
        디렉토리 내 모든 JSON 파일 로드 (청킹 없이 문서 전체 유지)

        Args:
            directory: JSON 파일들이 있는 디렉토리 경로
            content_field: 텍스트 내용이 담긴 필드명
        """
        json_dir = Path(directory)
        if not json_dir.exists():
            logger.error(f"Directory not found: {directory}")
            return

        json_files = list(json_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {directory}")

        # 모든 JSON 파일에서 문서 수집
        all_docs = []

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    data = [data]

                for item in data:
                    content = item.get(content_field, "")
                    if not content:
                        continue

                    metadata = {
                        "doc_id": item.get("doc_id", ""),
                        "source_type": item.get("source_type", ""),
                        "source_name": item.get("source_name", ""),
                        "source_file": item.get("source_file", json_file.name),
                    }

                    if "metadata" in item and isinstance(item["metadata"], dict):
                        metadata.update(item["metadata"])

                    # ChromaDB는 리스트/딕셔너리 메타데이터 지원 안함 - 문자열로 변환
                    cleaned_metadata = {}
                    for k, v in metadata.items():
                        if isinstance(v, (list, dict)):
                            cleaned_metadata[k] = json.dumps(v, ensure_ascii=False)
                        elif v is None:
                            cleaned_metadata[k] = ""
                        else:
                            cleaned_metadata[k] = v

                    all_docs.append(Document(page_content=content, metadata=cleaned_metadata))

            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

        if all_docs:
            self._vectorstore = Chroma.from_documents(
                documents=all_docs,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            self._retriever = None
            logger.info(f"Loaded {len(all_docs)} documents from {len(json_files)} JSON files (no chunking)")

    def query(
        self,
        question: str,
        emotion: Emotion = None,
        chat_history: List = None,
        search_type: str = None
    ) -> RAGResponse:
        """
        질문에 대한 RAG 응답 생성

        Args:
            question: 사용자 질문
            emotion: 감지된 감정 (프롬프트에 반영)
            chat_history: 대화 히스토리 (선택)
            search_type: 검색 방식 (None이면 인스턴스 설정 사용)

        Returns:
            RAGResponse: 답변과 참조 문서
        """
        emotion = emotion or Emotion.NEUTRAL
        search_type = search_type or self.search_type

        # 1. 관련 문서 검색 (Contextual Retrieval 또는 기존 방식)
        if search_type in ["hybrid", "bm25"]:
            # Contextual Retrieval 사용
            retrieval_result = self.contextual_retriever.retrieve(
                query=question,
                initial_k=self.initial_k,
                final_k=self.final_k,
                search_type=search_type
            )
            retrieved_docs = retrieval_result.documents
            retrieval_method = retrieval_result.method
        else:
            # 기존 similarity search
            retrieved_docs = self.retriever.invoke(question)
            retrieval_method = "similarity"

        logger.info(f"Retrieved {len(retrieved_docs)} docs using {retrieval_method}")

        # 2. 컨텍스트 구성
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 3. 감정 기반 시스템 프롬프트 생성
        system_prompt = build_system_prompt(emotion)

        # 4. 체인 구성 및 실행
        chain = (
            RAG_PROMPT_TEMPLATE
            | self.llm
            | StrOutputParser()
        )

        # 입력 구성
        chain_input = {
            "system_prompt": system_prompt,
            "context": context,
            "question": question,
            "chat_history": chat_history or []
        }

        # 5. 응답 생성
        answer = chain.invoke(chain_input)

        return RAGResponse(
            answer=answer,
            source_documents=retrieved_docs,
            emotion=emotion,
            retrieval_method=retrieval_method
        )

    def build_chain(self, emotion: Emotion = None):
        """
        LCEL 체인 반환 (고급 사용)

        Returns:
            Runnable: LangChain 체인
        """
        emotion = emotion or Emotion.NEUTRAL
        system_prompt = build_system_prompt(emotion)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context": self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "system_prompt": RunnableLambda(lambda _: system_prompt),
                "chat_history": RunnableLambda(lambda _: [])
            }
            | RAG_PROMPT_TEMPLATE
            | self.llm
            | StrOutputParser()
        )

        return chain

    def get_stats(self) -> Dict[str, Any]:
        """체인 통계"""
        return {
            "collection_name": self.collection_name,
            "document_count": self.vectorstore._collection.count(),
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "search_type": self.search_type,
            "use_reranker": self.use_reranker,
            "initial_k": self.initial_k,
            "final_k": self.final_k
        }
