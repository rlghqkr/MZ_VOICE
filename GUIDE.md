# MZ-VOICE 개발자 가이드

이 문서는 MZ-VOICE 프로젝트의 아키텍처, 설계 패턴, 각 모듈의 상세 설명을 제공합니다.

## 목차

1. [아키텍처 개요](#아키텍처-개요)
2. [Backend 상세 구조](#backend-상세-구조)
3. [Frontend 상세 구조](#frontend-상세-구조)
4. [설계 패턴](#설계-패턴)
5. [서비스별 상세 설명](#서비스별-상세-설명)
6. [RAG 시스템](#rag-시스템)
7. [환경 설정 가이드](#환경-설정-가이드)
8. [확장 가이드](#확장-가이드)

---

## 아키텍처 개요

MZ-VOICE는 계층형 아키텍처(Layered Architecture)를 채택하여 관심사 분리를 구현합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                           │
│  ┌─────────────────┐     ┌─────────────────────────────────┐   │
│  │   Gradio UI     │     │       React Frontend            │   │
│  │ (interfaces/)   │     │       (frontend/src/)           │   │
│  └─────────────────┘     └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              FastAPI Routes (api/routes/)               │   │
│  │   voice.py │ text.py │ sessions.py │ config.py         │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Voice RAG Pipeline (pipelines/)               │   │
│  │              Audio → STT → Emotion → RAG → TTS          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │
│  │    STT    │ │    TTS    │ │  Emotion  │ │    RAG    │       │
│  │ services/ │ │ services/ │ │ services/ │ │ services/ │       │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘       │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                     │
│  │  Session  │ │  Summary  │ │   Mail    │                     │
│  │ services/ │ │ services/ │ │ services/ │                     │
│  └───────────┘ └───────────┘ └───────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Access Layer                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Vector Store Repository (repositories/)        │   │
│  │                      ChromaDB                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Backend 상세 구조

### app/main.py

Gradio 기반 메인 진입점입니다. Gradio UI와 FastAPI를 함께 실행합니다.

```python
# Gradio 앱 실행
python -m app.main
```

### app/config.py

pydantic-settings를 사용한 타입 안전 설정 관리입니다.

```python
from app.config import get_settings

settings = get_settings()
print(settings.OPENAI_API_KEY)
print(settings.WHISPER_MODEL_SIZE)
```

### app/api/

FastAPI 애플리케이션 구조입니다.

```
api/
├── main.py              # FastAPI 앱 팩토리
├── dependencies.py      # 의존성 주입 컨테이너
├── routes/
│   ├── voice.py        # POST /api/v1/voice/process
│   ├── text.py         # POST /api/v1/text/process
│   ├── sessions.py     # CRUD /api/v1/sessions
│   └── config.py       # GET /api/v1/config/health
└── schemas/
    ├── voice.py        # VoiceProcessRequest/Response
    ├── text.py         # TextProcessRequest/Response
    ├── session.py      # SessionCreate/Response
    └── common.py       # EmotionInfo 등 공통 모델
```

### app/services/

비즈니스 로직 계층입니다. 각 서비스는 단일 책임 원칙을 따릅니다.

#### STT 서비스 (services/stt/)

```
stt/
├── __init__.py
├── base.py             # STTService 추상 베이스 클래스
├── whisper_stt.py      # faster-whisper 구현체
└── factory.py          # STT 서비스 팩토리
```

**사용 예시:**

```python
from app.services.stt.factory import create_stt_service

stt_service = create_stt_service("whisper")
result = await stt_service.transcribe(audio_bytes)
print(result.text)  # 변환된 텍스트
```

#### TTS 서비스 (services/tts/)

```
tts/
├── __init__.py
├── base.py             # TTSService 추상 베이스 클래스
├── gtts_tts.py         # gTTS 구현체
└── factory.py          # TTS 서비스 팩토리
```

**사용 예시:**

```python
from app.services.tts.factory import create_tts_service

tts_service = create_tts_service("gtts")
audio_bytes = await tts_service.synthesize("안녕하세요")
```

#### 감정 인식 서비스 (services/emotion/)

```
emotion/
├── __init__.py
├── base.py             # EmotionService 추상 베이스 클래스
└── audio_emotion.py    # wav2vec2 기반 감정 분석기
```

**지원 감정:**

| 감정 | 영문 | 설명 |
|------|------|------|
| 화남 | angry | 분노, 짜증 |
| 기쁨 | happy | 행복, 즐거움 |
| 슬픔 | sad | 우울, 슬픔 |
| 보통 | neutral | 감정 없음, 평온 |
| 불안 | fearful | 두려움, 걱정 |
| 놀람 | surprised | 깜짝 놀람 |

**사용 예시:**

```python
from app.services.emotion.audio_emotion import AudioEmotionAnalyzer

analyzer = AudioEmotionAnalyzer()
result = analyzer.analyze(audio_bytes)
print(result.emotion)        # "happy"
print(result.korean_label)   # "기쁨"
print(result.confidence)     # 0.85
```

#### RAG 서비스 (services/rag/)

```
rag/
├── __init__.py
├── chain.py            # LangChain RAG 체인
├── graph.py            # LangGraph 워크플로우
└── prompts.py          # 감정 기반 동적 프롬프트
```

**감정 기반 프롬프트 시스템:**

RAG 시스템은 사용자의 감정에 따라 응답 톤을 조절합니다.

```python
# prompts.py 예시
EMOTION_PROMPTS = {
    "angry": "사용자가 화가 난 상태입니다. 차분하고 이해심 있게 응대해주세요.",
    "sad": "사용자가 슬픈 상태입니다. 공감하며 따뜻하게 위로해주세요.",
    "happy": "사용자가 기분 좋은 상태입니다. 밝고 긍정적으로 응대해주세요.",
    # ...
}
```

#### 세션 서비스 (services/session/)

```
session/
├── __init__.py
├── manager.py          # 세션 생명주기 관리
└── storage.py          # 메시지 저장소
```

**세션 흐름:**

1. `create_session()` - 새 세션 생성
2. `add_message()` - 대화 메시지 추가
3. `get_messages()` - 대화 이력 조회
4. `end_session()` - 세션 종료 및 요약 생성

#### 요약 서비스 (services/summary/)

```
summary/
├── __init__.py
└── llm_summarizer.py   # LLM 기반 대화 요약
```

#### 이메일 서비스 (services/mail/)

```
mail/
├── __init__.py
├── base.py             # MailService 추상 베이스 클래스
└── smtp_sender.py      # SMTP 이메일 발송기
```

### app/pipelines/

처리 파이프라인을 정의합니다.

#### voice_rag_pipeline.py

```python
class VoiceRAGPipeline:
    """
    음성 RAG 처리 파이프라인

    처리 흐름:
    1. 오디오 입력 수신
    2. STT로 텍스트 변환
    3. 감정 분석
    4. RAG 응답 생성 (감정 컨텍스트 포함)
    5. TTS로 음성 변환
    6. 결과 반환
    """

    async def process(self, audio: bytes, session_id: str = None):
        # 1. STT
        transcription = await self.stt_service.transcribe(audio)

        # 2. 감정 분석
        emotion = self.emotion_analyzer.analyze(audio)

        # 3. RAG 응답 생성
        response = await self.rag_chain.invoke(
            query=transcription.text,
            emotion=emotion.emotion
        )

        # 4. TTS
        audio_output = await self.tts_service.synthesize(response)

        return VoiceProcessResult(...)
```

### app/repositories/

데이터 접근 계층입니다.

#### vector_store.py

ChromaDB 벡터 스토어 추상화입니다.

```python
class VectorStoreRepository:
    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)

    def similarity_search(self, query: str, k: int = 5):
        """유사도 기반 문서 검색"""
        pass

    def add_documents(self, documents: List[Document]):
        """문서 추가"""
        pass
```

---

## Frontend 상세 구조

### 디렉토리 구조

```
frontend/src/
├── main.tsx            # React 진입점
├── App.tsx             # 메인 앱 컴포넌트
├── index.css           # Tailwind 글로벌 스타일
├── api/                # API 클라이언트
│   ├── client.ts       # Axios 인스턴스
│   ├── config.ts       # API 설정
│   ├── voice.ts        # 음성 API
│   ├── text.ts         # 텍스트 API
│   └── sessions.ts     # 세션 API
├── components/         # React 컴포넌트
│   ├── Header.tsx
│   ├── ChatContainer.tsx
│   ├── MessageList.tsx
│   ├── VoiceRecorder.tsx
│   ├── TextInput.tsx
│   ├── MessageBubble.tsx
│   ├── EmotionBadge.tsx
│   └── index.ts
├── store/              # Zustand 상태 관리
│   ├── chatStore.ts
│   ├── sessionStore.ts
│   └── index.ts
├── hooks/              # 커스텀 훅
│   ├── useVoiceRecording.ts
│   └── index.ts
└── types/              # TypeScript 타입
    └── index.ts
```

### 주요 컴포넌트

#### ChatContainer.tsx

메인 채팅 인터페이스를 조율하는 컴포넌트입니다.

```tsx
const ChatContainer: React.FC = () => {
  const { messages, sendMessage, isLoading } = useChatStore();
  const { sessionId } = useSessionStore();

  return (
    <div className="chat-container">
      <MessageList messages={messages} />
      <VoiceRecorder onRecordingComplete={handleVoiceInput} />
      <TextInput onSubmit={handleTextInput} />
    </div>
  );
};
```

#### VoiceRecorder.tsx

음성 녹음 인터페이스입니다. WebAudio API를 사용합니다.

```tsx
const VoiceRecorder: React.FC<Props> = ({ onRecordingComplete }) => {
  const { isRecording, startRecording, stopRecording, audioBlob } = useVoiceRecording();

  // 파형 시각화, 녹음 상태 표시
  return (
    <div className="voice-recorder">
      <button onClick={isRecording ? stopRecording : startRecording}>
        {isRecording ? '녹음 중...' : '녹음 시작'}
      </button>
      <WaveformVisualizer isActive={isRecording} />
    </div>
  );
};
```

#### MessageBubble.tsx

개별 메시지를 표시하는 컴포넌트입니다.

```tsx
const MessageBubble: React.FC<Props> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
      <p>{message.content}</p>
      {message.emotion && <EmotionBadge emotion={message.emotion} />}
      {message.audioUrl && <AudioPlayer src={message.audioUrl} />}
      <span className="timestamp">{formatTime(message.timestamp)}</span>
    </div>
  );
};
```

### 상태 관리 (Zustand)

#### chatStore.ts

```typescript
interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;

  // Actions
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearMessages: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isLoading: false,
  error: null,

  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message]
  })),
  // ...
}));
```

#### sessionStore.ts

```typescript
interface SessionState {
  sessionId: string | null;
  customerEmail: string | null;

  // Actions
  createSession: () => Promise<void>;
  endSession: () => Promise<void>;
  setEmail: (email: string) => void;
}
```

### API 클라이언트

#### client.ts

```typescript
import axios from 'axios';

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});
```

#### voice.ts

```typescript
export const voiceApi = {
  processVoice: async (audioBlob: Blob, sessionId?: string) => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    const url = sessionId
      ? `/api/v1/voice/process/${sessionId}`
      : '/api/v1/voice/process';

    const response = await apiClient.post(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });

    return response.data;
  },

  getTTSAudio: async (audioId: string) => {
    const response = await apiClient.get(`/api/v1/voice/tts/${audioId}`, {
      responseType: 'blob'
    });
    return response.data;
  }
};
```

### 커스텀 훅

#### useVoiceRecording.ts

```typescript
export const useVoiceRecording = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    // ...녹음 로직
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  return { isRecording, audioBlob, startRecording, stopRecording };
};
```

### 스타일링

Tailwind CSS를 사용한 네온 다크 테마입니다.

**주요 색상:**

```css
/* tailwind.config.js */
colors: {
  dark: {
    400: '#374151',
    500: '#1f2937',
    600: '#111827',
    700: '#0d1117',
    800: '#0a0e14',
    900: '#050709'
  },
  neon: {
    cyan: '#00f5ff',
    purple: '#8b5cf6',
    pink: '#ec4899'
  }
}
```

---

## 설계 패턴

### 1. Strategy Pattern (전략 패턴)

STT와 TTS 서비스에서 사용됩니다. 런타임에 다른 구현체로 교체 가능합니다.

```python
# 추상 베이스
class STTService(ABC):
    @abstractmethod
    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        pass

# 구체적 구현
class WhisperSTT(STTService):
    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        # Whisper 구현
        pass

class ClovaSTT(STTService):
    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        # Clova 구현
        pass
```

### 2. Factory Pattern (팩토리 패턴)

서비스 인스턴스 생성을 캡슐화합니다.

```python
# factory.py
def create_stt_service(provider: str) -> STTService:
    if provider == "whisper":
        return WhisperSTT()
    elif provider == "clova":
        return ClovaSTT()
    else:
        raise ValueError(f"Unknown STT provider: {provider}")
```

### 3. Pipeline Pattern (파이프라인 패턴)

음성 처리를 단계별로 분리합니다.

```
Audio Input
    │
    ▼
┌─────────┐
│   STT   │ ─── 텍스트 변환
└─────────┘
    │
    ▼
┌─────────┐
│ Emotion │ ─── 감정 분석
└─────────┘
    │
    ▼
┌─────────┐
│   RAG   │ ─── 응답 생성
└─────────┘
    │
    ▼
┌─────────┐
│   TTS   │ ─── 음성 합성
└─────────┘
    │
    ▼
Audio Output
```

### 4. Repository Pattern (저장소 패턴)

데이터 접근 로직을 추상화합니다.

```python
class VectorStoreRepository:
    """ChromaDB 접근 추상화"""

    def similarity_search(self, query: str, k: int = 5):
        # ChromaDB 검색 로직
        pass
```

이 패턴으로 Pinecone, FAISS 등으로 쉽게 마이그레이션 가능합니다.

---

## RAG 시스템

### 데이터 문서

`data/documents/` 폴더에 17개의 JSON 문서가 있습니다.

| 파일명 | 내용 |
|--------|------|
| 복지로_서비스 목록_민간.json | 민간 복지 서비스 정보 |
| 복지멤버십_이용방법.json | 복지 멤버십 제도 안내 |
| 복지서비스_서비스목록_중앙부처.json | 중앙부처 복지 서비스 |
| 청년정책_개요.json | 청년정책 전반적 개요 |
| 청년정책_수립.json | 청년정책 수립 과정 |
| 청년센터_스키마.json | 전국 청년센터 정보 |
| 청년상담실_스키마.json | 청년 상담 서비스 |
| 청년참여_프로그램_스키마.json | 청년 참여 프로그램 |
| 청년꿀팁.json | 청년을 위한 유용한 정보 |
| ... | ... |

### RAG 흐름

```
사용자 질문
    │
    ▼
┌─────────────────┐
│ Embedding Model │ ─── 질문을 벡터로 변환
│ (OpenAI)        │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│    ChromaDB     │ ─── 유사 문서 k=5개 검색
│  Vector Store   │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Prompt Template │ ─── 감정 + 문서 + 질문 조합
│  + Emotion      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│    GPT-4o-mini  │ ─── 응답 생성
│      (LLM)      │
└─────────────────┘
    │
    ▼
최종 응답
```

### 감정 기반 프롬프트

```python
def get_emotion_prompt(emotion: str) -> str:
    prompts = {
        "angry": """
사용자가 화가 난 상태입니다.
- 먼저 사용자의 감정을 인정하고 공감해주세요
- 차분하고 이해심 있는 어조로 응대하세요
- 문제 해결에 집중하되 감정적 지지도 제공하세요
""",
        "sad": """
사용자가 슬픈 상태입니다.
- 따뜻하고 공감적인 어조로 응대하세요
- 위로의 말을 먼저 건네주세요
- 희망적인 정보를 제공해주세요
""",
        # ...
    }
    return prompts.get(emotion, "")
```

---

## 환경 설정 가이드

### 필수 환경 변수

```env
# OpenAI API (필수)
OPENAI_API_KEY=sk-your-key-here

# STT 설정
STT_PROVIDER=whisper              # whisper, clova, returnzero
WHISPER_MODEL_SIZE=base           # tiny, base, small, medium, large
WHISPER_DEVICE=cpu                # cpu, cuda

# TTS 설정
TTS_PROVIDER=gtts                 # gtts, edge, clova

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=faq_documents

# LLM
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7

# Embedding
EMBEDDING_MODEL=text-embedding-3-small
```

### 선택적 환경 변수

```env
# 이메일 발송 (선택)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_EMAIL=your-email@gmail.com
SMTP_FROM_NAME=고객상담센터

# 서버 설정
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false
DEBUG=true
```

### GPU 사용 설정

CUDA GPU를 사용하려면:

```env
WHISPER_DEVICE=cuda
```

그리고 PyTorch CUDA 버전을 설치합니다:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 확장 가이드

### 새로운 STT 제공자 추가

1. `services/stt/` 폴더에 새 파일 생성:

```python
# services/stt/my_stt.py
from .base import STTService, TranscriptionResult

class MySTTService(STTService):
    def __init__(self):
        # 초기화
        pass

    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        # 구현
        pass
```

2. Factory에 등록:

```python
# services/stt/factory.py
from .my_stt import MySTTService

def create_stt_service(provider: str) -> STTService:
    if provider == "my_stt":
        return MySTTService()
    # ...
```

3. 환경 변수로 선택:

```env
STT_PROVIDER=my_stt
```

### 새로운 감정 타입 추가

1. `services/emotion/audio_emotion.py` 수정:

```python
EMOTION_MAP = {
    # 기존 감정
    "angry": "화남",
    "happy": "기쁨",
    # 새 감정 추가
    "confused": "혼란",
    "excited": "흥분",
}
```

2. `services/rag/prompts.py`에 프롬프트 추가:

```python
EMOTION_PROMPTS["confused"] = """
사용자가 혼란스러운 상태입니다.
- 명확하고 단계별로 설명해주세요
- 복잡한 정보는 나누어 전달하세요
"""
```

### 새로운 문서 추가

1. JSON 형식으로 문서 작성:

```json
{
  "title": "문서 제목",
  "content": "문서 내용...",
  "metadata": {
    "category": "카테고리",
    "source": "출처"
  }
}
```

2. `data/documents/` 폴더에 저장

3. 벡터 스토어 재구축:

```python
from app.repositories.vector_store import VectorStoreRepository

repo = VectorStoreRepository()
repo.rebuild_index()
```

### 새로운 API 엔드포인트 추가

1. 스키마 정의:

```python
# api/schemas/my_schema.py
from pydantic import BaseModel

class MyRequest(BaseModel):
    field1: str
    field2: int

class MyResponse(BaseModel):
    result: str
```

2. 라우트 생성:

```python
# api/routes/my_route.py
from fastapi import APIRouter

router = APIRouter(prefix="/my-endpoint", tags=["my-tag"])

@router.post("/", response_model=MyResponse)
async def my_handler(request: MyRequest):
    return MyResponse(result="success")
```

3. 메인 앱에 등록:

```python
# api/main.py
from .routes import my_route

app.include_router(my_route.router, prefix="/api/v1")
```

---

## 문제 해결

### 자주 발생하는 오류

#### 1. CUDA 관련 오류

```
RuntimeError: CUDA out of memory
```

**해결:** WHISPER_MODEL_SIZE를 작은 모델로 변경 (tiny, base)

#### 2. OpenAI API 오류

```
openai.error.AuthenticationError: Invalid API key
```

**해결:** .env 파일의 OPENAI_API_KEY 확인

#### 3. 음성 녹음 오류 (Frontend)

```
NotAllowedError: Permission denied
```

**해결:** 브라우저에서 마이크 권한 허용

#### 4. ChromaDB 오류

```
chromadb.errors.InvalidCollectionException
```

**해결:** data/chroma_db 폴더 삭제 후 재시작

---

## 추가 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [LangChain 공식 문서](https://python.langchain.com/)
- [ChromaDB 공식 문서](https://docs.trychroma.com/)
- [React 공식 문서](https://react.dev/)
- [Zustand 공식 문서](https://zustand-demo.pmnd.rs/)
- [Tailwind CSS 공식 문서](https://tailwindcss.com/)
