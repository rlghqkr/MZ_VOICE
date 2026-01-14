"""
FastAPI Application Entry Point
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import sessions_router, voice_router, text_router, config_router
from .dependencies import get_pipeline, cleanup_old_audio_files

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting FastAPI application...")

    # Pre-initialize pipeline (optional - can be lazy loaded)
    try:
        pipeline = get_pipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.warning(f"Pipeline initialization failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application...")
    cleanup_old_audio_files(max_age_hours=0)  # Clean all temp files


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="MZ-VOICE API",
        description="Voice RAG Chatbot API with emotion recognition",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev server
            "http://localhost:3000",  # Alternative dev server
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers with /api/v1 prefix
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(voice_router, prefix="/api/v1")
    app.include_router(text_router, prefix="/api/v1")
    app.include_router(config_router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {
            "message": "MZ-VOICE API",
            "docs": "/docs",
            "health": "/api/v1/health"
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
