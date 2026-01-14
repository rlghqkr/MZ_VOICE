"""
Configuration and Health Check Routes
"""

from fastapi import APIRouter, Depends, Response
from typing import Optional

from ..schemas.common import HealthResponse, ConfigResponse
from ..dependencies import get_pipeline, get_audio_temp
from ...pipelines import VoiceRAGPipeline
from ...config import settings

router = APIRouter(tags=["config"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration"""
    return ConfigResponse(
        stt_provider=settings.stt_provider,
        tts_provider=settings.tts_provider,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model
    )


@router.get("/stats")
async def get_stats(pipeline: VoiceRAGPipeline = Depends(get_pipeline)):
    """Get pipeline statistics"""
    return pipeline.get_stats()


@router.get("/audio/{audio_id}")
async def get_audio(audio_id: str):
    """Get audio file by ID"""
    result = get_audio_temp(audio_id)

    if not result:
        return Response(status_code=404)

    audio_bytes, content_type = result

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename={audio_id}.mp3"
        }
    )
