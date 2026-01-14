"""
Gradio Application

음성/텍스트 인터페이스를 제공하는 Gradio 앱입니다.
"""

import logging
from typing import Tuple, Optional
import tempfile
import os

import gradio as gr

from ..pipelines import VoiceRAGPipeline
from ..services.emotion import Emotion
from ..config import settings

logger = logging.getLogger(__name__)

# 전역 파이프라인 인스턴스
pipeline: Optional[VoiceRAGPipeline] = None


def initialize_pipeline():
    """파이프라인 초기화"""
    global pipeline
    
    if pipeline is None:
        logger.info("Initializing pipeline...")
        pipeline = VoiceRAGPipeline(
            use_mock_emotion=True  # 개발 모드
        )
        
        # JSON 문서 로드 (우선순위 1)
        json_doc_dir = "./data/documents/json"
        if os.path.exists(json_doc_dir):
            pipeline.load_documents_from_json_directory(json_doc_dir)
            logger.info(f"JSON documents loaded from {json_doc_dir}")
            return pipeline

        # 샘플 문서 로드 (있으면)
        sample_doc_path = "./data/documents/sample_faq.txt"
        if os.path.exists(sample_doc_path):
            pipeline.load_documents_from_file(sample_doc_path)
            logger.info("Sample documents loaded")
        else:
            # 기본 샘플 문서
            sample_docs = [
                """
                # 환불 정책
                - 구매 후 7일 이내 환불 가능합니다.
                - 사용하지 않은 상품만 환불 가능합니다.
                - 환불은 원래 결제 수단으로 처리됩니다.
                - 환불 처리는 영업일 기준 3-5일 소요됩니다.
                """,
                """
                # 배송 안내
                - 배송은 결제 완료 후 1-3일 이내 출발합니다.
                - 배송 추적은 마이페이지에서 확인 가능합니다.
                - 무료배송 기준 금액은 50,000원입니다.
                - 도서산간 지역은 추가 배송비가 발생합니다.
                """,
                """
                # 고객센터 연락처
                - 전화: 1588-0000
                - 이메일: support@example.com
                - 운영시간: 평일 09:00 - 18:00
                - 점심시간: 12:00 - 13:00
                """
            ]
            pipeline.load_documents(sample_docs)
            logger.info("Default sample documents loaded")
    
    return pipeline


def process_voice_input(audio_file) -> Tuple[str, str, str]:
    """
    음성 입력 처리
    
    Args:
        audio_file: Gradio 오디오 파일 (파일 경로 또는 튜플)
        
    Returns:
        Tuple[str, str, str]: (변환된 텍스트, 응답, 감정)
    """
    if audio_file is None:
        return "음성을 녹음해주세요.", "", ""
    
    try:
        pipe = initialize_pipeline()
        
        # 오디오 파일 읽기
        if isinstance(audio_file, tuple):
            # (sample_rate, audio_array) 형태
            sample_rate, audio_array = audio_file
            # numpy array to bytes 변환
            import soundfile as sf
            import io
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            audio_bytes = buffer.getvalue()
        else:
            # 파일 경로
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
        
        # 파이프라인 실행
        result = pipe.process_voice(audio_bytes, return_audio=False)
        
        if result.error:
            return f"오류: {result.error}", "", ""
        
        transcription = result.transcription.text if result.transcription else ""
        response = result.output_text or ""
        emotion = result.emotion.korean_label if result.emotion else "보통"
        
        return transcription, response, emotion
        
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        return f"오류: {str(e)}", "", ""


def process_text_input(
    text: str,
    emotion: str = "보통"
) -> str:
    """
    텍스트 입력 처리
    
    Args:
        text: 사용자 텍스트
        emotion: 선택된 감정
        
    Returns:
        str: AI 응답
    """
    if not text.strip():
        return "질문을 입력해주세요."
    
    try:
        pipe = initialize_pipeline()
        
        # 감정 매핑
        emotion_map = {
            "보통": Emotion.NEUTRAL,
            "화남": Emotion.ANGRY,
            "기쁨": Emotion.HAPPY,
            "슬픔": Emotion.SAD,
            "불안": Emotion.FEARFUL,
            "놀람": Emotion.SURPRISED,
        }
        selected_emotion = emotion_map.get(emotion, Emotion.NEUTRAL)
        
        # 파이프라인 실행
        result = pipe.process_text(text, selected_emotion)
        
        if result.error:
            return f"오류: {result.error}"
        
        return result.output_text or "응답을 생성할 수 없습니다."
        
    except Exception as e:
        logger.error(f"Text processing error: {e}")
        return f"오류: {str(e)}"


def create_gradio_app() -> gr.Blocks:
    """Gradio 앱 생성"""
    
    with gr.Blocks(
        title="🎤 음성 RAG 콜봇",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 800px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        """
    ) as app:
        
        gr.Markdown("""
        # 🎤 음성 RAG 콜봇
        
        음성 또는 텍스트로 질문하면 AI가 답변합니다.
        감정 인식을 통해 고객 상황에 맞는 응대를 제공합니다.
        """)
        
        with gr.Tabs():
            # ===== 음성 인터페이스 탭 =====
            with gr.Tab("🎙️ 음성 입력"):
                gr.Markdown("### 음성으로 질문하기")
                
                with gr.Row():
                    audio_input = gr.Audio(
                        label="음성 녹음",
                        sources=["microphone", "upload"],
                        type="filepath"
                    )
                
                voice_btn = gr.Button("🔊 음성 분석", variant="primary")
                
                with gr.Row():
                    transcription_output = gr.Textbox(
                        label="📝 변환된 텍스트",
                        lines=2
                    )
                    emotion_output = gr.Textbox(
                        label="😊 감지된 감정",
                        lines=1
                    )
                
                voice_response = gr.Textbox(
                    label="🤖 AI 응답",
                    lines=5
                )
                
                voice_btn.click(
                    fn=process_voice_input,
                    inputs=[audio_input],
                    outputs=[transcription_output, voice_response, emotion_output]
                )
            
            # ===== 텍스트 인터페이스 탭 =====
            with gr.Tab("💬 텍스트 입력"):
                gr.Markdown("### 텍스트로 질문하기")
                
                with gr.Row():
                    text_input = gr.Textbox(
                        label="질문 입력",
                        placeholder="예: 환불하고 싶어요...",
                        lines=3
                    )
                    emotion_select = gr.Dropdown(
                        choices=["보통", "화남", "기쁨", "슬픔", "불안", "놀람"],
                        value="보통",
                        label="감정 선택 (시뮬레이션)"
                    )
                
                text_btn = gr.Button("💬 질문하기", variant="primary")
                
                text_response = gr.Textbox(
                    label="🤖 AI 응답",
                    lines=5
                )
                
                text_btn.click(
                    fn=process_text_input,
                    inputs=[text_input, emotion_select],
                    outputs=[text_response]
                )
                
                # 예시 질문
                gr.Examples(
                    examples=[
                        ["환불 받으려면 어떻게 해야 하나요?", "보통"],
                        ["배송이 너무 늦어요! 언제 오는 거예요?", "화남"],
                        ["고객센터 전화번호 알려주세요", "보통"],
                        ["주문한 물건이 잘못 왔어요...", "슬픔"],
                    ],
                    inputs=[text_input, emotion_select],
                    label="예시 질문"
                )
            
            # ===== 설정 탭 =====
            with gr.Tab("⚙️ 설정"):
                gr.Markdown("### 시스템 설정")
                
                with gr.Row():
                    gr.Markdown(f"""
                    **현재 설정**
                    - STT 제공자: `{settings.stt_provider}`
                    - TTS 제공자: `{settings.tts_provider}`
                    - LLM 모델: `{settings.llm_model}`
                    - 임베딩 모델: `{settings.embedding_model}`
                    """)
                
                reload_btn = gr.Button("🔄 파이프라인 재시작")
                status_output = gr.Textbox(label="상태", lines=3)
                
                def reload_pipeline():
                    global pipeline
                    pipeline = None
                    initialize_pipeline()
                    return "파이프라인이 재시작되었습니다."
                
                reload_btn.click(
                    fn=reload_pipeline,
                    outputs=[status_output]
                )
        
        gr.Markdown("""
        ---
        ### 📚 사용 가이드
        1. **음성 입력**: 마이크로 녹음하거나 음성 파일을 업로드하세요.
        2. **텍스트 입력**: 직접 텍스트를 입력하고 감정을 선택하세요.
        3. AI가 질문에 맞는 답변을 제공합니다.
        
        *💡 감정 인식은 현재 Mock 모드로 동작합니다.*
        """)
    
    return app


def launch_app(share: bool = False, port: int = None):
    """앱 실행"""
    app = create_gradio_app()
    app.launch(
        share=share or settings.gradio_share,
        server_port=port or settings.gradio_server_port
    )


# 직접 실행 시
if __name__ == "__main__":
    launch_app()
