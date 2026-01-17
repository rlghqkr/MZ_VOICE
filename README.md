# MZ-VOICE

**MZ-VOICE**는 한국 정부 청년 상담 서비스를 위한 감정 인식 기반 음성 RAG 챗봇입니다.

음성 입력을 텍스트로 변환하고, 사용자의 감정을 분석하여 공감적인 응답을 생성한 후, 다시 음성으로 출력하는 종합적인 AI 상담 시스템입니다.

## 주요 기능

- **음성 처리 파이프라인**: STT(음성→텍스트) → 감정 분석 → RAG 응답 생성 → TTS(텍스트→음성)
- **감정 인식**: wav2vec2 모델을 활용한 6가지 감정 분류 (화남, 기쁨, 슬픔, 보통, 불안, 놀람)
- **RAG 시스템**: 청년 복지 정책 문서 기반 검색 증강 생성
- **세션 관리**: 상담 이력 저장, 이메일 요약 발송 기능
- **React 프론트엔드**: 음성 녹음 및 채팅 인터페이스

## 기술 스택

### Backend (Python)

| 구성 요소 | 기술 | 용도 |
|----------|------|------|
| Framework | FastAPI + Uvicorn | API 서버 |
| STT | faster-whisper | 음성→텍스트 변환 |
| TTS | gTTS, edge-tts | 텍스트→음성 변환 |
| 감정 인식 | transformers (wav2vec2) | 음성 감정 분류 |
| RAG | LangChain + LangGraph | 검색 증강 생성 |
| Vector Store | ChromaDB | 벡터 데이터베이스 |
| LLM | OpenAI GPT-4o-mini | 응답 생성 |

### Frontend (TypeScript/React)

| 구성 요소 | 기술 | 용도 |
|----------|------|------|
| Framework | React 18.2.0 | UI 프레임워크 |
| Build Tool | Vite 5.0 | 개발 서버 및 번들러 |
| State Management | Zustand 4.5.0 | 전역 상태 관리 |
| HTTP Client | Axios 1.6.0 | API 통신 |
| Styling | Tailwind CSS 3.4 | CSS 프레임워크 |

## 프로젝트 구조

```
MZ-VOICE_front_end/
├── app/                          # Python 백엔드
│   ├── main.py                   # Gradio/FastAPI 진입점
│   ├── config.py                 # 설정 관리
│   ├── api/                      # FastAPI 라우트 및 스키마
│   │   ├── routes/               # API 엔드포인트
│   │   └── schemas/              # Pydantic 모델
│   ├── services/                 # 비즈니스 로직 계층
│   │   ├── stt/                  # Speech-to-Text 서비스
│   │   ├── tts/                  # Text-to-Speech 서비스
│   │   ├── emotion/              # 감정 인식 서비스
│   │   ├── rag/                  # RAG 체인 및 프롬프트
│   │   ├── session/              # 세션 관리
│   │   ├── summary/              # 대화 요약
│   │   └── mail/                 # 이메일 발송
│   ├── pipelines/                # 처리 파이프라인
│   ├── repositories/             # 데이터 접근 계층
│   └── interfaces/               # UI 인터페이스
│
├── frontend/                     # React 프론트엔드
│   ├── src/
│   │   ├── api/                  # API 클라이언트
│   │   ├── components/           # React 컴포넌트
│   │   ├── store/                # Zustand 상태 관리
│   │   ├── hooks/                # 커스텀 훅
│   │   └── types/                # TypeScript 타입
│   ├── package.json
│   └── vite.config.ts
│
├── data/                         # 데이터 디렉토리
│   ├── documents/                # RAG 학습 문서 (17개 JSON)
│   ├── audio_temp/               # 임시 오디오 파일
│   └── chroma_db/                # 벡터 스토어
│
├── tests/                        # 테스트 파일
├── docs/                         # 문서
├── requirements.txt              # Python 의존성
└── .env                          # 환경 변수
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/your-repo/MZ-VOICE_front_end.git
cd MZ-VOICE_front_end
```

### 2. Python 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정합니다:

```env
# 필수 설정
OPENAI_API_KEY=sk-your-openai-api-key

# STT 설정
STT_PROVIDER=whisper
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu

# TTS 설정
TTS_PROVIDER=gtts

# ChromaDB 설정
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=faq_documents

# LLM 설정
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7

# 프롬프트 로깅 (색상 포함)
ENABLE_PROMPT_LOGGING=false  # true로 설정하면 프롬프트와 LLM 응답을 색상과 함께 로깅

# 서버 설정
GRADIO_SERVER_PORT=7860
DEBUG=true
```

### 4. Frontend 설정

```bash
cd frontend
npm install
```

## 실행 방법

### Backend 실행

```bash
# FastAPI 서버 실행
uvicorn app.api.main:app --reload --port 9000

# 또는 Gradio UI로 실행
python -m app.main
```

### Frontend 실행

```bash
cd frontend
npm run dev
```

- Backend API: http://localhost:9000
- Frontend: http://localhost:5173
- Gradio UI: http://localhost:7860 (Gradio 모드 사용 시)

## API 엔드포인트

### 세션 관리

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/sessions` | 새 세션 생성 |
| GET | `/api/v1/sessions/{session_id}` | 세션 정보 조회 |
| GET | `/api/v1/sessions/{session_id}/messages` | 대화 이력 조회 |
| PUT | `/api/v1/sessions/{session_id}/email` | 이메일 업데이트 |
| POST | `/api/v1/sessions/{session_id}/end` | 세션 종료 및 요약 |
| DELETE | `/api/v1/sessions/{session_id}` | 세션 삭제 |

### 음성 처리

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/voice/process` | 음성 처리 (STT + 감정 + RAG + TTS) |
| POST | `/api/v1/voice/process/{session_id}` | 세션 내 음성 처리 |
| GET | `/api/v1/voice/tts/{audio_id}` | TTS 오디오 조회 |

### 텍스트 처리

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/text/process` | 텍스트 RAG 처리 |
| POST | `/api/v1/text/process/{session_id}` | 세션 내 텍스트 처리 |

### 설정

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/v1/config/health` | 헬스체크 |
| GET | `/api/v1/config/status` | 시스템 상태 |

## 응답 형식

### 음성 처리 응답 (VoiceProcessResponse)

```json
{
  "transcription": "사용자 발화 텍스트",
  "response_text": "AI 응답 텍스트",
  "emotion": {
    "emotion": "happy",
    "korean_label": "기쁨",
    "confidence": 0.85
  },
  "audio_url": "/api/v1/voice/tts/audio-id",
  "processing_time": 2.5
}
```

### 감정 유형

| 영문 | 한글 | 설명 |
|------|------|------|
| angry | 화남 | 분노 감정 |
| happy | 기쁨 | 긍정적 감정 |
| sad | 슬픔 | 슬픈 감정 |
| neutral | 보통 | 중립적 감정 |
| fearful | 불안 | 두려움/불안 |
| surprised | 놀람 | 놀란 감정 |

## 테스트

```bash
# 전체 테스트 실행
pytest tests/

# 개별 테스트 실행
pytest tests/test_stt.py
pytest tests/test_tts.py
pytest tests/test_rag.py
pytest tests/test_pipeline.py
pytest tests/test_session.py
```

## 라이선스

MIT License - Copyright (c) 2025 MALDDSS

## 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
