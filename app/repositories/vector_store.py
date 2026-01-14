"""
Vector Store Repository

ChromaDB를 사용한 벡터 저장소 Repository 패턴 구현입니다.
데이터 접근 로직을 비즈니스 로직과 분리합니다.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)


class VectorStoreRepository:
    """
    벡터 저장소 Repository
    
    ChromaDB를 래핑하여 문서 저장 및 검색 기능을 제공합니다.
    Repository 패턴을 적용하여 DB 교체가 용이합니다.
    
    Example:
        repo = VectorStoreRepository()
        repo.add_documents(documents)
        results = repo.search("주문 취소 방법")
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_function = None
    ):
        """
        Args:
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: 영구 저장 디렉토리
            embedding_function: 임베딩 함수 (없으면 기본값 사용)
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or str(settings.chroma_persist_dir)
        
        self._client = None
        self._collection = None
        self._embedding_function = embedding_function
        
    @property
    def client(self):
        """ChromaDB 클라이언트 (Lazy Loading)"""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    @property
    def collection(self):
        """ChromaDB 컬렉션 (Lazy Loading)"""
        if self._collection is None:
            self._initialize_collection()
        return self._collection
    
    def _initialize_client(self):
        """ChromaDB 클라이언트 초기화"""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # 영구 저장소 디렉토리 생성
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            logger.info(f"ChromaDB client initialized at {self.persist_directory}")
            
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise
    
    def _initialize_collection(self):
        """컬렉션 초기화"""
        if self._embedding_function is None:
            self._embedding_function = self._get_default_embedding_function()
        
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
        )
        
        logger.info(f"Collection '{self.collection_name}' ready. "
                   f"Documents: {self._collection.count()}")
    
    def _get_default_embedding_function(self):
        """기본 임베딩 함수 반환"""
        try:
            # OpenAI 임베딩 사용
            from chromadb.utils import embedding_functions
            
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key,
                model_name=settings.embedding_model
            )
        except Exception as e:
            logger.warning(f"OpenAI embedding failed, using default: {e}")
            # 기본 임베딩 함수 사용
            from chromadb.utils import embedding_functions
            return embedding_functions.DefaultEmbeddingFunction()
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        문서 추가
        
        Args:
            documents: 문서 텍스트 리스트
            metadatas: 메타데이터 리스트 (선택)
            ids: 문서 ID 리스트 (선택, 없으면 자동 생성)
            
        Returns:
            List[str]: 추가된 문서 ID 목록
        """
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to collection")
        return ids
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        유사 문서 검색
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            where: 메타데이터 필터 (선택)
            
        Returns:
            Dict: 검색 결과 (documents, distances, metadatas 포함)
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }
    
    def delete_documents(self, ids: List[str]):
        """문서 삭제"""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")
    
    def clear_collection(self):
        """컬렉션 초기화"""
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.info(f"Collection '{self.collection_name}' cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """컬렉션 통계"""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.persist_directory
        }
    
    # LangChain 호환 메서드
    def as_retriever(self, search_kwargs: Dict = None):
        """LangChain Retriever로 변환"""
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.embedding_model
        )
        
        langchain_chroma = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        
        search_kwargs = search_kwargs or {"k": 5}
        return langchain_chroma.as_retriever(search_kwargs=search_kwargs)
