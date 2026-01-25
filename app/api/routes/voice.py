"""
Voice Processing Routes
"""

import io
import json
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.responses import Response, StreamingResponse
from typing import Optional, Generator

from ..schemas import VoiceProcessResponse, EmotionInfo
from ..dependencies import (
    get_pipeline,
    get_session_manager,
    save_audio_temp,
    get_audio_temp,
)
from ...pipelines import VoiceRAGPipeline
from ...services.session import SessionManager

router = APIRouter(prefix="/voice", tags=["voice"])


def create_sse_event(event_type: str, data: dict) -> str:
    """SSE 이벤트 포맷 생성"""
    json_data = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {json_data}\n\n"


@router.post("/process", response_model=VoiceProcessResponse)
async def process_voice(
    audio: UploadFile = File(..., description="Audio file (WAV format)"),
    session_id: Optional[str] = Form(None),
    return_audio: bool = Form(False),
    pipeline: VoiceRAGPipeline = Depends(get_pipeline),
    manager: SessionManager = Depends(get_session_manager)
):
    """
    Process voice input through the full pipeline

    - Converts speech to text (STT)
    - Analyzes emotion from audio
    - Generates response using RAG
    - Optionally converts response to speech (TTS)
    """
    # Read audio file
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Process through pipeline (session_id 전달하여 QueryBuilder 상태 관리)
    result = pipeline.process_voice(
        audio_bytes,
        return_audio=return_audio,
        session_id=session_id or "default"
    )

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    # Build response
    audio_url = None
    if return_audio and result.output_audio:
        audio_id = save_audio_temp(result.output_audio.audio, "mp3")
        audio_url = f"/api/v1/audio/{audio_id}"

    emotion_info = EmotionInfo(
        emotion=result.emotion.primary_emotion.value if result.emotion else "neutral",
        korean_label=result.emotion.korean_label if result.emotion else "보통",
        confidence=result.emotion.confidence if result.emotion else 0.0
    )

    # Save to session if provided
    if session_id:
        session = manager.get_session(session_id)
        if session:
            manager.add_user_message(
                session_id,
                result.transcription.text if result.transcription else "",
                emotion=emotion_info.emotion
            )
            manager.add_assistant_message(session_id, result.output_text or "")

    return VoiceProcessResponse(
        transcription=result.transcription.text if result.transcription else "",
        response_text=result.output_text or "",
        emotion=emotion_info,
        audio_url=audio_url,
        processing_time=result.processing_time,
        needs_more_info=result.needs_more_info,
        conversation_phase=result.agent_response.phase.value if result.agent_response else None
    )


@router.post("/process/stream")
async def process_voice_stream(
    audio: UploadFile = File(..., description="Audio file (WAV format)"),
    session_id: Optional[str] = Form(None),
    pipeline: VoiceRAGPipeline = Depends(get_pipeline),
    manager: SessionManager = Depends(get_session_manager)
):
    """
    실시간 스트리밍 음성 처리 (SSE)

    전화처럼 응답을 실시간으로 TTS 변환하여 전송합니다.

    SSE 이벤트 타입:
    - transcription: 음성 인식 결과 {"text": "...", "confidence": 0.95}
    - emotion: 감정 분석 결과 {"emotion": "happy", "korean_label": "기쁨", "confidence": 0.9}
    - text_chunk: LLM 응답 텍스트 청크 {"text": "..."}
    - audio_chunk: TTS 오디오 청크 {"audio": "base64...", "text": "...", "format": "mp3"}
    - done: 처리 완료 {"processing_time": 2.5, "needs_more_info": false}
    - error: 에러 발생 {"message": "..."}
    """
    # Read audio file
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    def generate_sse_events() -> Generator[str, None, None]:
        """SSE 이벤트 스트림 생성"""
        transcription_text = None
        full_response_text = ""
        emotion_info = None

        for event in pipeline.process_voice_stream_realtime(
            audio=audio_bytes,
            session_id=session_id or "default"
        ):
            event_type = event.get("event")
            data = event.get("data", {})

            # 데이터 수집 (세션 저장용)
            if event_type == "transcription":
                transcription_text = data.get("text", "")
            elif event_type == "emotion":
                emotion_info = data
            elif event_type == "text_chunk":
                full_response_text += data.get("text", "")

            # SSE 이벤트 전송
            yield create_sse_event(event_type, data)

        # 세션에 메시지 저장
        if session_id and transcription_text:
            session = manager.get_session(session_id)
            if session:
                manager.add_user_message(
                    session_id,
                    transcription_text,
                    emotion=emotion_info.get("emotion") if emotion_info else "neutral"
                )
                if full_response_text:
                    manager.add_assistant_message(session_id, full_response_text)

    return StreamingResponse(
        generate_sse_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
        }
    )


@router.post("/process/{session_id}", response_model=VoiceProcessResponse)
async def process_voice_with_session(
    session_id: str,
    audio: UploadFile = File(..., description="Audio file (WAV format)"),
    return_audio: bool = Form(False),
    pipeline: VoiceRAGPipeline = Depends(get_pipeline),
    manager: SessionManager = Depends(get_session_manager)
):
    """
    Process voice input within a session context
    """
    # Verify session exists
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Read audio file
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Process through pipeline (session_id 전달하여 QueryBuilder 상태 관리)
    result = pipeline.process_voice(
        audio_bytes,
        return_audio=return_audio,
        session_id=session_id
    )

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    # Build response
    audio_url = None
    if return_audio and result.output_audio:
        audio_id = save_audio_temp(result.output_audio.audio, "mp3")
        audio_url = f"/api/v1/audio/{audio_id}"

    emotion_info = EmotionInfo(
        emotion=result.emotion.primary_emotion.value if result.emotion else "neutral",
        korean_label=result.emotion.korean_label if result.emotion else "보통",
        confidence=result.emotion.confidence if result.emotion else 0.0
    )

    # Save to session
    manager.add_user_message(
        session_id,
        result.transcription.text if result.transcription else "",
        emotion=emotion_info.emotion
    )
    manager.add_assistant_message(session_id, result.output_text or "")

    return VoiceProcessResponse(
        transcription=result.transcription.text if result.transcription else "",
        response_text=result.output_text or "",
        emotion=emotion_info,
        audio_url=audio_url,
        processing_time=result.processing_time,
        needs_more_info=result.needs_more_info,
        conversation_phase=result.agent_response.phase.value if result.agent_response else None
    )


@router.get("/tts/{audio_id}")
async def get_tts_audio(audio_id: str):
    """
    Get TTS audio file by ID
    """
    result = get_audio_temp(audio_id)

    if not result:
        raise HTTPException(status_code=404, detail="Audio not found")

    audio_bytes, content_type = result

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={audio_id}.mp3"
        }
    )


@router.post("/process/{session_id}/stream")
async def process_voice_stream_with_session(
    session_id: str,
    audio: UploadFile = File(..., description="Audio file (WAV format)"),
    pipeline: VoiceRAGPipeline = Depends(get_pipeline),
    manager: SessionManager = Depends(get_session_manager)
):
    """
    세션 기반 실시간 스트리밍 음성 처리 (SSE)

    세션 컨텍스트 내에서 실시간으로 응답을 TTS 변환하여 전송합니다.
    """
    # Verify session exists
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Read audio file
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    def generate_sse_events() -> Generator[str, None, None]:
        """SSE 이벤트 스트림 생성"""
        transcription_text = None
        full_response_text = ""
        emotion_info = None

        for event in pipeline.process_voice_stream_realtime(
            audio=audio_bytes,
            session_id=session_id
        ):
            event_type = event.get("event")
            data = event.get("data", {})

            # 데이터 수집 (세션 저장용)
            if event_type == "transcription":
                transcription_text = data.get("text", "")
            elif event_type == "emotion":
                emotion_info = data
            elif event_type == "text_chunk":
                full_response_text += data.get("text", "")

            # SSE 이벤트 전송
            yield create_sse_event(event_type, data)

        # 세션에 메시지 저장
        if transcription_text:
            manager.add_user_message(
                session_id,
                transcription_text,
                emotion=emotion_info.get("emotion") if emotion_info else "neutral"
            )
            if full_response_text:
                manager.add_assistant_message(session_id, full_response_text)

    return StreamingResponse(
        generate_sse_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
