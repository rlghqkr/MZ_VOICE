"""
Configuration management for Voice RAG Chatbot
Uses pydantic-settings for type-safe configuration
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # STT Configuration
    stt_provider: Literal["whisper", "sensevoice", "clova", "returnzero"] = Field(
        default="whisper",
        env="STT_PROVIDER"
    )
    whisper_model_size: str = Field(default="base", env="WHISPER_MODEL_SIZE")
    whisper_device: Literal["cuda", "cpu"] = Field(default="cpu", env="WHISPER_DEVICE")
    asr_api_base_url: str = Field(
        default="http://34.130.232.40:8000",
        env="ASR_API_BASE_URL"
    )
    asr_include_emotion: bool = Field(default=True, env="ASR_INCLUDE_EMOTION")
    asr_timeout_seconds: float = Field(default=30.0, env="ASR_TIMEOUT_SECONDS")
    asr_use_emotion: bool = Field(default=True, env="ASR_USE_EMOTION")
    
    # TTS Configuration
    tts_provider: Literal["gtts", "edge", "clova"] = Field(
        default="gtts",
        env="TTS_PROVIDER"
    )
    
    # ChromaDB Configuration
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma_db"),
        env="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        default="faq_documents",
        env="CHROMA_COLLECTION_NAME"
    )

    # Law ChromaDB Configuration
    chroma_law_persist_dir: Path = Field(
        default=Path("./data/chroma_db_law"),
        env="CHROMA_LAW_PERSIST_DIR"
    )
    chroma_law_collection_name: str = Field(
        default="laws",
        env="CHROMA_LAW_COLLECTION_NAME"
    )
    
    # LLM Configuration
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        env="EMBEDDING_MODEL"
    )
    
    # SMTP Mail Configuration
    smtp_host: str = Field(default="smtp.gmail.com", env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: str = Field(default="", env="SMTP_USERNAME")
    smtp_password: str = Field(default="", env="SMTP_PASSWORD")
    smtp_from_email: str = Field(default="", env="SMTP_FROM_EMAIL")
    smtp_from_name: str = Field(default="고객상담센터", env="SMTP_FROM_NAME")
    smtp_use_tls: bool = Field(default=True, env="SMTP_USE_TLS")
    
    # Gradio Configuration
    gradio_server_port: int = Field(default=7860, env="GRADIO_SERVER_PORT")
    gradio_share: bool = Field(default=False, env="GRADIO_SHARE")

    # FastAPI Configuration
    fastapi_host: str = Field(default="0.0.0.0", env="FASTAPI_HOST")
    fastapi_port: int = Field(default=9000, env="FASTAPI_PORT")
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        env="CORS_ORIGINS"
    )

    # RAG Configuration
    enable_reranking: bool = Field(default=True, env="ENABLE_RERANKING")
    
    # Logging Configuration
    enable_prompt_logging: bool = Field(default=False, env="ENABLE_PROMPT_LOGGING")
    
    # Debug
    debug: bool = Field(default=True, env="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance
settings = Settings()


# Emotion labels for classification
EMOTION_LABELS = {
    "angry": "화남",
    "happy": "기쁨",
    "sad": "슬픔",
    "neutral": "보통",
    "fearful": "불안",
    "surprised": "놀람"
}

# Emotion-based response styles
EMOTION_RESPONSE_STYLES = {
    "angry": "고객님의 불편함에 진심으로 사과드립니다. 차분하고 공감하는 어조로 문제 해결에 집중해주세요.",
    "happy": "고객님의 긍정적인 에너지에 맞춰 밝고 친근한 어조로 응대해주세요.",
    "sad": "고객님의 상황에 공감하며, 따뜻하고 위로하는 어조로 도움을 드려주세요.",
    "neutral": "전문적이고 친절한 어조로 정확한 정보를 제공해주세요.",
    "fearful": "고객님을 안심시키며, 차분하고 신뢰감 있는 어조로 안내해주세요.",
    "surprised": "상황을 명확히 설명하며, 이해하기 쉽게 안내해주세요."
}
