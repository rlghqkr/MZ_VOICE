"""
Voice RAG Pipeline

음성 입력부터 음성 출력까지의 전체 파이프라인을 조율합니다.
Pipeline Pattern을 적용하여 각 단계를 독립적으로 관리합니다.

Hybrid RAG 지원:
- 법령 관련 질문 → GraphRAG
- 일반 질문 → RAGChain (Hybrid Search)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Generator, Literal
import time

from ..services.stt import STTFactory, STTBase, TranscriptionResult
from ..services.tts import TTSFactory, TTSBase, SynthesisResult
from ..services.emotion import AudioEmotionAnalyzer, EmotionResult, Emotion
from ..services.rag import (
    RAGChain, RAGGraph, RAGResponse,
    HybridRAGService, HybridRAGResponse,
    QueryBuilderGraph, GraphAgentResponse, ConversationPhase
)
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    # 입력
    input_audio: Optional[bytes] = None
    input_text: Optional[str] = None

    # STT 결과
    transcription: Optional[TranscriptionResult] = None

    # 감정 분석 결과
    emotion: Optional[EmotionResult] = None

    # Query Builder 결과
    agent_response: Optional[GraphAgentResponse] = None
    needs_more_info: bool = False  # 추가 정보 수집 필요 여부

    # RAG 결과
    rag_response: Optional[RAGResponse] = None
    hybrid_rag_response: Optional[HybridRAGResponse] = None

    # TTS 결과
    output_audio: Optional[SynthesisResult] = None
    output_text: Optional[str] = None

    # 메타
    processing_time: float = 0.0
    error: Optional[str] = None
    query_type: Optional[str] = None  # "law" or "general"


class VoiceRAGPipeline:
    """
    음성 RAG 파이프라인

    전체 처리 흐름:
    1. 음성 입력 → STT → 텍스트
    2. 음성 → 감정 분석 → 감정
    3. (텍스트 + 감정) → RAG → 응답
    4. 응답 → TTS → 음성 출력

    Example:
        pipeline = VoiceRAGPipeline()

        # 음성 입력
        result = pipeline.process_voice(audio_bytes)

        # 또는 텍스트 입력
        result = pipeline.process_text("환불하고 싶어요")

        print(result.output_text)
    """

    def __init__(
        self,
        stt_provider: str = None,
        tts_provider: str = None,
        use_langgraph: bool = False,
        use_mock_emotion: bool = False,
        use_hybrid_rag: bool = True,
        use_query_builder: bool = True,
        rag_mode: Literal["auto", "general", "law"] = "auto"
    ):
        """
        Args:
            stt_provider: STT 제공자 (기본값: config)
            tts_provider: TTS 제공자 (기본값: config)
            use_langgraph: True면 LangGraph 사용, False면 LangChain
            use_mock_emotion: False면 wav2vec2 모델 사용, True면 Mock 감정 분석
            use_hybrid_rag: True면 HybridRAGService 사용 (GraphRAG + RAGChain)
            use_query_builder: True면 QueryBuilderGraph로 정보 수집 후 RAG 실행
            rag_mode: RAG 모드 ("auto": 자동 라우팅, "general": 일반만, "law": 법령만)
        """
        # 컴포넌트 초기화 (Lazy Loading)
        self._stt: Optional[STTBase] = None
        self._tts: Optional[TTSBase] = None
        self._emotion_analyzer: Optional[AudioEmotionAnalyzer] = None
        self._rag: Optional[RAGChain] = None
        self._rag_graph: Optional[RAGGraph] = None
        self._hybrid_rag: Optional[HybridRAGService] = None
        self._query_builder: Optional[QueryBuilderGraph] = None

        # 설정
        self.stt_provider = stt_provider or settings.stt_provider
        self.tts_provider = tts_provider or settings.tts_provider
        self.use_langgraph = use_langgraph
        self.use_mock_emotion = use_mock_emotion
        self.use_hybrid_rag = use_hybrid_rag
        self.use_query_builder = use_query_builder
        self.rag_mode = rag_mode

    # ============ Properties (Lazy Loading) ============

    @property
    def stt(self) -> STTBase:
        if self._stt is None:
            self._stt = STTFactory.create(self.stt_provider)
        return self._stt

    @property
    def tts(self) -> TTSBase:
        if self._tts is None:
            self._tts = TTSFactory.create(self.tts_provider)
        return self._tts

    @property
    def emotion_analyzer(self) -> AudioEmotionAnalyzer:
        if self._emotion_analyzer is None:
            self._emotion_analyzer = AudioEmotionAnalyzer(
                use_mock=self.use_mock_emotion
            )
        return self._emotion_analyzer

    @property
    def rag(self) -> RAGChain:
        if self._rag is None:
            self._rag = RAGChain()
        return self._rag

    @property
    def rag_graph(self) -> RAGGraph:
        if self._rag_graph is None:
            self._rag_graph = RAGGraph(self.rag)
        return self._rag_graph

    @property
    def hybrid_rag(self) -> HybridRAGService:
        """HybridRAGService 인스턴스 (Lazy Loading)"""
        if self._hybrid_rag is None:
            self._hybrid_rag = HybridRAGService()
        return self._hybrid_rag

    @property
    def query_builder(self) -> QueryBuilderGraph:
        """QueryBuilderGraph 인스턴스 (Lazy Loading)"""
        if self._query_builder is None:
            self._query_builder = QueryBuilderGraph()
        return self._query_builder

    # ============ 문서 관리 ============

    def load_documents(self, documents: list, **kwargs):
        """RAG용 문서 로드"""
        self.rag.load_documents(documents, **kwargs)
        logger.info(f"Loaded {len(documents)} documents to RAG")

    def load_documents_from_file(self, file_path: str):
        """파일에서 문서 로드"""
        self.rag.load_from_file(file_path)

    def load_documents_from_json(self, file_path: str, **kwargs):
        """JSON 파일에서 문서 로드"""
        self.rag.load_from_json(file_path, **kwargs)
        logger.info(f"Loaded documents from JSON: {file_path}")

    def load_documents_from_json_directory(self, directory: str, **kwargs):
        """디렉토리 내 모든 JSON 파일 로드"""
        self.rag.load_from_json_directory(directory, **kwargs)
        logger.info(f"Loaded documents from JSON directory: {directory}")

    # ============ 메인 처리 메서드 ============

    def process_voice(
        self,
        audio: bytes,
        return_audio: bool = True
    ) -> PipelineResult:
        """
        음성 입력 처리 (전체 파이프라인)

        Args:
            audio: 음성 데이터 (WAV 형식)
            return_audio: True면 TTS 결과 포함

        Returns:
            PipelineResult: 전체 처리 결과
        """
        start_time = time.time()
        result = PipelineResult(input_audio=audio)

        try:
            # Step 1: STT (음성 → 텍스트)
            logger.debug("Step 1: STT")
            result.transcription = self.stt.transcribe(audio)

            # Step 2: 감정 분석
            logger.debug("Step 2: Emotion Analysis")
            result.emotion = self.emotion_analyzer.analyze(audio)

            # Step 3: RAG (텍스트 + 감정 → 응답)
            logger.debug("Step 3: RAG")
            rag_response, hybrid_response, query_type = self._get_rag_response(
                question=result.transcription.text,
                emotion=result.emotion.primary_emotion
            )
            result.rag_response = rag_response
            result.hybrid_rag_response = hybrid_response
            result.query_type = query_type
            result.output_text = rag_response.answer

            # Step 4: TTS (응답 → 음성)
            if return_audio:
                logger.debug("Step 4: TTS")
                result.output_audio = self.tts.synthesize(result.output_text)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.error = str(e)

        result.processing_time = time.time() - start_time
        return result

    def process_text(
        self,
        text: str,
        emotion: Emotion = None,
        return_audio: bool = False,
        session_id: str = "default"
    ) -> PipelineResult:
        """
        텍스트 입력 처리 (감정 직접 지정)

        Args:
            text: 사용자 텍스트
            emotion: 감정 (None이면 NEUTRAL)
            return_audio: True면 TTS 결과 포함
            session_id: 세션 ID (QueryBuilder 상태 관리용)

        Returns:
            PipelineResult: 처리 결과
        """
        start_time = time.time()
        result = PipelineResult(input_text=text)
        emotion = emotion or Emotion.NEUTRAL

        try:
            if self.use_query_builder:
                # Step 1: QueryBuilderGraph로 정보 수집/쿼리 생성
                agent_response = self.query_builder.process(
                    message=text,
                    session_id=session_id,
                    emotion=emotion.value if emotion else None
                )
                result.agent_response = agent_response

                if agent_response.phase == ConversationPhase.READY:
                    # 정보 충분 - RAG 쿼리 실행
                    result.needs_more_info = False
                    rag_query = agent_response.rag_query or text

                    rag_response, hybrid_response, query_type = self._get_rag_response(
                        question=rag_query,
                        emotion=emotion
                    )
                    result.rag_response = rag_response
                    result.hybrid_rag_response = hybrid_response
                    result.query_type = query_type
                    result.output_text = rag_response.answer
                else:
                    # 추가 정보 필요 - 에이전트 메시지 반환
                    result.needs_more_info = True
                    result.output_text = agent_response.message
            else:
                # QueryBuilder 미사용 - 직접 RAG 실행
                rag_response, hybrid_response, query_type = self._get_rag_response(
                    question=text,
                    emotion=emotion
                )
                result.rag_response = rag_response
                result.hybrid_rag_response = hybrid_response
                result.query_type = query_type
                result.output_text = rag_response.answer

            # TTS (선택적)
            if return_audio:
                result.output_audio = self.tts.synthesize(result.output_text)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.error = str(e)

        result.processing_time = time.time() - start_time
        return result

    def process_text_direct(
        self,
        text: str,
        emotion: Emotion = None,
        return_audio: bool = False
    ) -> PipelineResult:
        """
        텍스트 입력 처리 (QueryBuilder 우회, 직접 RAG 실행)

        Args:
            text: 사용자 텍스트
            emotion: 감정 (None이면 NEUTRAL)
            return_audio: True면 TTS 결과 포함

        Returns:
            PipelineResult: 처리 결과
        """
        start_time = time.time()
        result = PipelineResult(input_text=text)

        try:
            emotion = emotion or Emotion.NEUTRAL
            rag_response, hybrid_response, query_type = self._get_rag_response(
                question=text,
                emotion=emotion
            )
            result.rag_response = rag_response
            result.hybrid_rag_response = hybrid_response
            result.query_type = query_type
            result.output_text = rag_response.answer

            if return_audio:
                result.output_audio = self.tts.synthesize(result.output_text)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.error = str(e)

        result.processing_time = time.time() - start_time
        return result

    def _get_rag_response(
        self,
        question: str,
        emotion: Emotion
    ) -> tuple[RAGResponse, Optional[HybridRAGResponse], Optional[str]]:
        """
        RAG 응답 생성

        Returns:
            tuple: (RAGResponse, HybridRAGResponse, query_type)
        """
        if self.use_hybrid_rag:
            # HybridRAGService 사용 (GraphRAG + RAGChain 자동 라우팅)
            hybrid_response = self.hybrid_rag.query(
                question=question,
                emotion=emotion,
                force_rag_type=self.rag_mode
            )
            # RAGResponse 형식으로도 반환 (호환성)
            rag_response = RAGResponse(
                answer=hybrid_response.answer,
                source_documents=hybrid_response.source_documents,
                emotion=hybrid_response.emotion,
                retrieval_method=hybrid_response.retrieval_method
            )
            return rag_response, hybrid_response, hybrid_response.query_type

        elif self.use_langgraph:
            # LangGraph 사용
            graph_result = self.rag_graph.invoke(question, emotion)
            return RAGResponse(
                answer=graph_result.get("answer", ""),
                source_documents=[],
                emotion=emotion
            ), None, None
        else:
            # LangChain 사용
            return self.rag.query(question, emotion), None, None

    # ============ 스트리밍 처리 ============

    def process_voice_stream(
        self,
        audio_stream: Generator[bytes, None, None]
    ) -> Generator[PipelineResult, None, None]:
        """
        스트리밍 음성 처리

        음성 청크가 들어올 때마다 부분 결과를 yield합니다.
        """
        for partial_text in self.stt.transcribe_stream(audio_stream):
            # 부분 텍스트로 RAG 쿼리 (최종 결과만)
            result = PipelineResult(input_text=partial_text)
            result.output_text = f"[처리 중] {partial_text}"
            yield result

        # 최종 결과
        # Note: 실제 구현에서는 전체 텍스트를 모아서 처리

    # ============ 유틸리티 ============

    def get_stats(self) -> dict:
        """파이프라인 상태"""
        stats = {
            "stt_provider": self.stt_provider,
            "tts_provider": self.tts_provider,
            "use_langgraph": self.use_langgraph,
            "use_hybrid_rag": self.use_hybrid_rag,
            "use_query_builder": self.use_query_builder,
            "rag_mode": self.rag_mode,
        }

        if self.use_hybrid_rag and self._hybrid_rag is not None:
            stats["hybrid_rag_stats"] = self._hybrid_rag.get_stats()
        elif self._rag is not None:
            stats["rag_stats"] = self._rag.get_stats()

        if self._query_builder is not None:
            stats["query_builder_sessions"] = len(self._query_builder._sessions)

        return stats

    def reset_query_builder_session(self, session_id: str = "default"):
        """QueryBuilder 세션 초기화"""
        if self._query_builder is not None:
            self._query_builder.reset_session(session_id)

    def get_user_profile(self, session_id: str = "default"):
        """현재 세션의 사용자 프로필 반환"""
        if self._query_builder is not None:
            return self._query_builder.get_profile(session_id)
        return None
