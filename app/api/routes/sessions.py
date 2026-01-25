"""
Session Management Routes
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List

from ..schemas import (
    SessionCreateRequest,
    SessionResponse,
    SessionEndResponse,
    MessageResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from ..schemas.session import SessionMessagesResponse, EmailUpdateRequest
from ...services.summary import LLMSummarizer
from ..dependencies import get_session_manager, get_pipeline
from ...services.session import SessionManager
from ...pipelines import VoiceRAGPipeline

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    request: SessionCreateRequest,
    manager: SessionManager = Depends(get_session_manager)
):
    """Create a new session"""
    session_id = manager.start_session(customer_email=request.customer_email)
    session = manager.get_session(session_id)

    return SessionResponse(
        session_id=session.session_id,
        customer_email=session.customer_email,
        created_at=session.created_at,
        ended_at=session.ended_at,
        message_count=len(session.messages)
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager)
):
    """Get session information"""
    session = manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        session_id=session.session_id,
        customer_email=session.customer_email,
        created_at=session.created_at,
        ended_at=session.ended_at,
        message_count=len(session.messages)
    )


@router.get("/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager)
):
    """Get all messages in a session"""
    session = manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = []
    for msg in session.messages:
        emotion = None
        if msg.metadata and "emotion" in msg.metadata:
            emotion = msg.metadata["emotion"]

        messages.append(MessageResponse(
            role=msg.role.value,
            content=msg.content,
            timestamp=msg.timestamp,
            emotion=emotion
        ))

    return SessionMessagesResponse(
        session_id=session_id,
        messages=messages
    )


@router.put("/{session_id}/email")
async def update_session_email(
    session_id: str,
    request: EmailUpdateRequest,
    manager: SessionManager = Depends(get_session_manager)
):
    """Update customer email for a session"""
    session = manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    success = manager.set_customer_email(session_id, request.email)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update email")

    return {"success": True, "email": request.email}


@router.post("/{session_id}/end", response_model=SessionEndResponse)
async def end_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
    pipeline: VoiceRAGPipeline = Depends(get_pipeline)
):
    """End a session and generate summary"""
    session = manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # QueryBuilderGraph 세션 초기화 (수집된 사용자 정보 리셋)
    pipeline.reset_query_builder_session(session_id)

    result = manager.end_session(session_id)

    return SessionEndResponse(
        session_id=result.session_id,
        success=result.success,
        summary=result.summary,
        mail_sent=result.mail_sent,
        error=result.error
    )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
    pipeline: VoiceRAGPipeline = Depends(get_pipeline)
):
    """Delete a session"""
    session = manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # QueryBuilderGraph 세션 초기화
    pipeline.reset_query_builder_session(session_id)

    manager.cleanup_session(session_id)

    return {"success": True, "message": f"Session {session_id} deleted"}


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_conversation(request: SummarizeRequest):
    """
    대화 내용을 요약합니다 (세션 없이 직접 요약)

    Frontend에서 대화 메시지 목록을 전달하면 LLM이 요약을 생성합니다.
    """
    if not request.messages:
        return SummarizeResponse(
            summary="대화 내용이 없습니다.",
            success=True
        )

    try:
        # 대화 내용을 텍스트로 변환
        lines = []
        for msg in request.messages:
            role_label = "고객" if msg.role == "user" else "상담사"
            lines.append(f"[{role_label}] {msg.content}")
        conversation_text = "\n".join(lines)

        # LLM으로 요약 생성
        summarizer = LLMSummarizer()
        summary = summarizer.summarize(conversation_text)

        return SummarizeResponse(
            summary=summary,
            success=True
        )
    except Exception as e:
        return SummarizeResponse(
            summary="",
            success=False,
            error=str(e)
        )
