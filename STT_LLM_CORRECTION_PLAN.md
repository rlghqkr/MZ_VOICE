# STT LLM 보정 적용 계획 (SenseVoice)

## 목표
- 청년 복지 콜센터 도메인에 맞게 STT 결과를 LLM으로 후처리하여 정확도 개선.
- SenseVoice 테스트 시에만 선택적으로 켤 수 있도록 구성.

## 변경안
1) 보정 헬퍼 추가
   - `app/services/stt/llm_correction.py` 신규 생성.
   - 제공한 프롬프트 로직으로 `correct_with_llm(stt_text: str) -> str` 구현.
   - 기존 OpenAI 클라이언트가 있다면 재사용, 없으면 최소 초기화 추가.

2) SenseVoice STT에 보정 연결
   - `app/services/stt/sensevoice.py`에서 ASR 텍스트 파싱 후 `correct_with_llm` 호출.
   - `STT_LLM_CORRECTION=true`일 때만 동작하도록 설정 플래그 적용.
   - 빈 문자열/공백 텍스트는 보정 스킵.

3) 설정 추가
   - `Settings`에 아래 항목 추가:
     - `stt_llm_correction: bool` (env: `STT_LLM_CORRECTION`)
     - `stt_llm_model: str` (env: `STT_LLM_MODEL`, 기본값 `gpt-4o-mini`)
     - `stt_llm_max_tokens: int` (env: `STT_LLM_MAX_TOKENS`, 기본값 `100`)
   - 필요 시 `stt_llm_timeout_seconds`도 추가 검토.

4) 의존성 확인
   - OpenAI SDK가 이미 쓰이고 있으면 재사용.
   - 없다면 `requirements.txt`에 `openai` 추가.

5) 로깅/안전
   - 보정 결과가 달라질 때만 원본/수정 로그 출력.
   - 예외 발생 시 원문 그대로 반환 (STT 파이프라인 실패 방지).

## 수정 대상 파일
- `app/services/stt/sensevoice.py`
- `app/services/stt/llm_correction.py` (신규)
- `app/config.py`
- `requirements.txt` (필요 시)
- `README.md` 또는 `.env.sample` (환경변수 문서화, 선택)

## 확인 필요
- OpenAI 클라이언트가 이미 프로젝트 어디선가 초기화되어 있는지.
- SenseVoice 외 다른 STT에도 보정을 적용할지 여부.
