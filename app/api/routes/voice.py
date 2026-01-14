"""
Voice Processing Routes
"""

import io
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi.responses import Response
from typing import Optional

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

    # Process through pipeline
    result = pipeline.process_voice(audio_bytes, return_audio=return_audio)

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
        processing_time=result.processing_time
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

    # Process through pipeline
    result = pipeline.process_voice(audio_bytes, return_audio=return_audio)

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
        processing_time=result.processing_time
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
