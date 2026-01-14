"""
Text Processing Routes
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from ..schemas import TextProcessRequest, TextProcessResponse
from ..dependencies import get_pipeline, get_session_manager, save_audio_temp
from ...pipelines import VoiceRAGPipeline
from ...services.session import SessionManager
from ...services.emotion import Emotion

router = APIRouter(prefix="/text", tags=["text"])


def get_emotion_from_string(emotion_str: str) -> Emotion:
    """Convert emotion string to Emotion enum"""
    emotion_map = {
        "angry": Emotion.ANGRY,
        "happy": Emotion.HAPPY,
        "sad": Emotion.SAD,
        "neutral": Emotion.NEUTRAL,
        "fearful": Emotion.FEARFUL,
        "surprised": Emotion.SURPRISED,
    }
    return emotion_map.get(emotion_str.lower(), Emotion.NEUTRAL)


@router.post("/process", response_model=TextProcessResponse)
async def process_text(
    request: TextProcessRequest,
    pipeline: VoiceRAGPipeline = Depends(get_pipeline),
    manager: SessionManager = Depends(get_session_manager)
):
    """
    Process text input through RAG pipeline

    - Uses specified emotion for response style
    - Optionally converts response to speech (TTS)
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Convert emotion string to enum
    emotion = get_emotion_from_string(request.emotion or "neutral")

    # Process through pipeline (session_id 전달로 QueryBuilderGraph 세션 분리)
    result = pipeline.process_text(
        text=request.text,
        emotion=emotion,
        return_audio=request.return_audio,
        session_id=request.session_id or "default"
    )

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    # Build response
    audio_url = None
    if request.return_audio and result.output_audio:
        audio_id = save_audio_temp(result.output_audio.audio, "mp3")
        audio_url = f"/api/v1/audio/{audio_id}"

    # Save to session if provided
    if request.session_id:
        session = manager.get_session(request.session_id)
        if session:
            manager.add_user_message(
                request.session_id,
                request.text,
                emotion=request.emotion
            )
            manager.add_assistant_message(request.session_id, result.output_text or "")

    return TextProcessResponse(
        response_text=result.output_text or "",
        audio_url=audio_url,
        processing_time=result.processing_time
    )


@router.post("/process/{session_id}", response_model=TextProcessResponse)
async def process_text_with_session(
    session_id: str,
    request: TextProcessRequest,
    pipeline: VoiceRAGPipeline = Depends(get_pipeline),
    manager: SessionManager = Depends(get_session_manager)
):
    """
    Process text input within a session context
    """
    # Verify session exists
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Convert emotion string to enum
    emotion = get_emotion_from_string(request.emotion or "neutral")

    # Process through pipeline (session_id 전달로 QueryBuilderGraph 세션 분리)
    result = pipeline.process_text(
        text=request.text,
        emotion=emotion,
        return_audio=request.return_audio,
        session_id=session_id
    )

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    # Build response
    audio_url = None
    if request.return_audio and result.output_audio:
        audio_id = save_audio_temp(result.output_audio.audio, "mp3")
        audio_url = f"/api/v1/audio/{audio_id}"

    # Save to session
    manager.add_user_message(session_id, request.text, emotion=request.emotion)
    manager.add_assistant_message(session_id, result.output_text or "")

    return TextProcessResponse(
        response_text=result.output_text or "",
        audio_url=audio_url,
        processing_time=result.processing_time
    )
