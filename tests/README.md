# 간단한 테스트 스크립트

`test_rag_quick.py` 스타일의 간단한 실행 테스트입니다.

## 🚀 실행 방법

```bash
# conda 환경 활성화 (필수!)
conda activate mz-voice

# RAG 테스트
python tests/test_rag.py

# 파이프라인 테스트
python tests/test_pipeline.py

# STT 테스트
python tests/test_stt.py
```

## 📁 파일 설명

- `test_rag.py` - RAGChain, HybridRAGService 간단 테스트
- `test_pipeline.py` - VoiceRAGPipeline 간단 테스트
- `test_stt.py` - STT 서비스 간단 테스트

## 💡 특징

- pytest 없이 그냥 `python` 실행
- 복잡한 fixture 없음
- 결과 바로 확인
- 에러 메시지 명확
- 로그 파일 자동 저장 (`logs/` 디렉토리)

## 📊 로그 파일

테스트 실행 시 다음 위치에 로그 파일이 자동 생성됩니다:

```
logs/
├── test_rag_2026-01-14_16-30-00.log
├── test_pipeline_2026-01-14_16-31-00.log
└── test_stt_2026-01-14_16-32-00.log
```

- 타임스탬프 포함 파일명으로 저장
- 콘솔 출력과 동일한 내용 기록
- UTF-8 인코딩으로 한글 지원

## 🔗 참고

더 상세한 테스트는 프로젝트 루트의 `test_rag_quick.py` 참고
