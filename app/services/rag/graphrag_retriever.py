"""
GraphRAG Retriever - Microsoft GraphRAG Python API Wrapper

법령 관련 질문에 대해 지식 그래프 기반 검색을 수행합니다.
Local Search: 특정 엔티티/관계 기반 검색
Global Search: 전체 문서 요약 기반 검색
"""

import logging
from pathlib import Path
from typing import Optional, Literal, List
from dataclasses import dataclass

import pandas as pd
from langchain_core.documents import Document

from graphrag.config.load_config import load_config
from graphrag.api.query import local_search, global_search

from ...config import settings

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGResponse:
    """GraphRAG 검색 결과"""
    answer: str
    context: str
    method: Literal["local", "global"]
    source_documents: list


class GraphRAGRetriever:
    """
    GraphRAG 기반 법령 검색기

    Microsoft GraphRAG를 사용하여 법령 지식 그래프에서
    관련 정보를 검색합니다.

    Example:
        retriever = GraphRAGRetriever()
        response = retriever.search("청년기본법 제3조의 내용은?")
        print(response.answer)
    """

    def __init__(
        self,
        root_dir: str = None,
        community_level: int = 2
    ):
        """
        Args:
            root_dir: GraphRAG 프로젝트 루트 디렉토리
            community_level: 커뮤니티 계층 수준 (local search용)
        """
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).parents[3] / "graphrag_project"
        self.community_level = community_level

        self._config = None
        self._entities_df = None
        self._relationships_df = None
        self._text_units_df = None
        self._communities_df = None
        self._community_reports_df = None

        self._initialized = False

    def _load_config(self):
        """GraphRAG 설정 로드"""
        if self._config is None:
            self._config = load_config(self.root_dir)
        return self._config

    def _load_parquet_files(self):
        """Parquet 파일에서 데이터 로드"""
        if self._initialized:
            return

        output_dir = self.root_dir / "output"

        try:
            # 필수 파일 로드
            entities_path = output_dir / "entities.parquet"
            relationships_path = output_dir / "relationships.parquet"
            text_units_path = output_dir / "text_units.parquet"
            communities_path = output_dir / "communities.parquet"
            community_reports_path = output_dir / "community_reports.parquet"

            if entities_path.exists():
                self._entities_df = pd.read_parquet(entities_path)
                logger.info(f"Loaded {len(self._entities_df)} entities")

            if relationships_path.exists():
                self._relationships_df = pd.read_parquet(relationships_path)
                logger.info(f"Loaded {len(self._relationships_df)} relationships")

            if text_units_path.exists():
                self._text_units_df = pd.read_parquet(text_units_path)
                logger.info(f"Loaded {len(self._text_units_df)} text units")

            if communities_path.exists():
                self._communities_df = pd.read_parquet(communities_path)
                logger.info(f"Loaded {len(self._communities_df)} communities")

            if community_reports_path.exists():
                self._community_reports_df = pd.read_parquet(community_reports_path)
                logger.info(f"Loaded {len(self._community_reports_df)} community reports")

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to load parquet files: {e}")
            raise

    async def local_search_async(
        self,
        query: str,
        community_level: int = None
    ) -> GraphRAGResponse:
        """
        Local Search 실행 (비동기)

        특정 엔티티와 관계를 기반으로 검색합니다.
        구체적인 법령 조항이나 특정 개념에 대한 질문에 적합합니다.

        Args:
            query: 검색 질문
            community_level: 커뮤니티 수준 (기본: 인스턴스 설정)

        Returns:
            GraphRAGResponse: 검색 결과
        """
        self._load_parquet_files()
        config = self._load_config()

        community_level = community_level or self.community_level

        try:
            result = await local_search(
                config=config,
                entities=self._entities_df,
                relationships=self._relationships_df,
                text_units=self._text_units_df,
                communities=self._communities_df,
                community_reports=self._community_reports_df,
                community_level=community_level,
                query=query
            )

            # 결과에서 컨텍스트와 응답 추출
            answer = result.response if hasattr(result, 'response') else str(result)
            context = result.context if hasattr(result, 'context') else ""

            # Document 형식으로 변환
            source_documents = []
            if self._text_units_df is not None and context:
                # 컨텍스트에서 관련 문서 추출
                source_documents = [
                    Document(
                        page_content=context,
                        metadata={"source": "graphrag_local", "method": "local"}
                    )
                ]

            return GraphRAGResponse(
                answer=answer,
                context=context,
                method="local",
                source_documents=source_documents
            )

        except Exception as e:
            logger.error(f"Local search failed: {e}")
            raise

    async def global_search_async(
        self,
        query: str
    ) -> GraphRAGResponse:
        """
        Global Search 실행 (비동기)

        전체 문서 요약을 기반으로 검색합니다.
        법령 전반에 대한 개요나 요약 질문에 적합합니다.

        Args:
            query: 검색 질문

        Returns:
            GraphRAGResponse: 검색 결과
        """
        self._load_parquet_files()
        config = self._load_config()

        try:
            result = await global_search(
                config=config,
                entities=self._entities_df,
                communities=self._communities_df,
                community_reports=self._community_reports_df,
                query=query
            )

            answer = result.response if hasattr(result, 'response') else str(result)
            context = result.context if hasattr(result, 'context') else ""

            source_documents = [
                Document(
                    page_content=context if context else answer,
                    metadata={"source": "graphrag_global", "method": "global"}
                )
            ]

            return GraphRAGResponse(
                answer=answer,
                context=context,
                method="global",
                source_documents=source_documents
            )

        except Exception as e:
            logger.error(f"Global search failed: {e}")
            raise

    def search(
        self,
        query: str,
        method: Literal["local", "global"] = "local"
    ) -> GraphRAGResponse:
        """
        동기 검색 (내부적으로 비동기 실행)

        Args:
            query: 검색 질문
            method: 검색 방법 ("local" 또는 "global")

        Returns:
            GraphRAGResponse: 검색 결과
        """
        import asyncio

        if method == "local":
            coro = self.local_search_async(query)
        else:
            coro = self.global_search_async(query)

        # 이벤트 루프 처리
        try:
            loop = asyncio.get_running_loop()
            # 이미 실행 중인 루프가 있으면 새 태스크로 실행
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # 실행 중인 루프가 없으면 직접 실행
            return asyncio.run(coro)

    def retrieve_documents(
        self,
        query: str,
        k: int = 5,
        method: Literal["local", "global"] = "local"
    ) -> List[Document]:
        """
        GraphRAG에서 관련 문서만 검색 (LLM 응답 생성 없이)

        Args:
            query: 검색 질문
            k: 반환할 문서 수
            method: 검색 방법 ("local" 또는 "global")

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        self._load_parquet_files()

        documents = []

        try:
            if self._text_units_df is not None and len(self._text_units_df) > 0:
                # text_units에서 관련 문서 검색
                # 간단한 키워드 매칭으로 관련 문서 찾기
                query_tokens = set(query.lower().split())

                scored_units = []
                for idx, row in self._text_units_df.iterrows():
                    text = row.get('text', '') or row.get('chunk', '') or ''
                    if not text:
                        continue

                    # 키워드 매칭 점수
                    text_lower = text.lower()
                    score = sum(1 for token in query_tokens if token in text_lower)

                    if score > 0:
                        scored_units.append((text, score, row))

                # 점수순 정렬 후 상위 k개 선택
                scored_units.sort(key=lambda x: x[1], reverse=True)

                for text, score, row in scored_units[:k]:
                    metadata = {
                        "source": "graphrag",
                        "method": method,
                        "score": score
                    }
                    # 추가 메타데이터가 있으면 포함
                    if 'id' in row:
                        metadata['text_unit_id'] = str(row['id'])
                    if 'document_ids' in row:
                        metadata['document_ids'] = str(row['document_ids'])

                    documents.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))

            # text_units가 없거나 결과가 부족하면 community_reports에서 보충
            if len(documents) < k and self._community_reports_df is not None:
                remaining = k - len(documents)
                query_tokens = set(query.lower().split())

                scored_reports = []
                for idx, row in self._community_reports_df.iterrows():
                    content = row.get('full_content', '') or row.get('summary', '') or ''
                    if not content:
                        continue

                    content_lower = content.lower()
                    score = sum(1 for token in query_tokens if token in content_lower)

                    if score > 0:
                        scored_reports.append((content, score, row))

                scored_reports.sort(key=lambda x: x[1], reverse=True)

                for content, score, row in scored_reports[:remaining]:
                    metadata = {
                        "source": "graphrag_community",
                        "method": method,
                        "score": score
                    }
                    if 'community' in row:
                        metadata['community_id'] = str(row['community'])
                    if 'title' in row:
                        metadata['title'] = str(row['title'])

                    documents.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))

            logger.info(f"GraphRAG retrieved {len(documents)} documents for query: '{query[:50]}...'")

        except Exception as e:
            logger.error(f"GraphRAG document retrieval failed: {e}")

        return documents

    def get_stats(self) -> dict:
        """GraphRAG 통계 정보"""
        self._load_parquet_files()

        return {
            "root_dir": str(self.root_dir),
            "entities_count": len(self._entities_df) if self._entities_df is not None else 0,
            "relationships_count": len(self._relationships_df) if self._relationships_df is not None else 0,
            "text_units_count": len(self._text_units_df) if self._text_units_df is not None else 0,
            "communities_count": len(self._communities_df) if self._communities_df is not None else 0,
            "community_reports_count": len(self._community_reports_df) if self._community_reports_df is not None else 0,
            "community_level": self.community_level,
            "initialized": self._initialized
        }
