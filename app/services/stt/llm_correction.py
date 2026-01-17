"""
LLM-based STT correction helper.
"""

import logging

from langchain_openai import ChatOpenAI

from ...config import settings

logger = logging.getLogger(__name__)


def correct_with_llm(stt_text: str) -> str:
    """OpenAI LLM을 사용하여 STT 오류 보정"""
    if not stt_text.strip():
        return stt_text

    correction_prompt = f"""당신은 청년 복지 콜센터의 STT 오류 보정 전문가입니다.

다음은 음성 인식(STT) 결과입니다. 도메인 지식을 활용하여 올바른 텍스트로 보정하세요.

**보정 대상:**
- 복지제도: 청년도약계좌, 청년전월세상담소, 새일센터, 국민취업제도
- 지역명: 전남 보성군, 경북 포항시, 서울 강남구 등
- 직업: 프리랜서, 자영업자, 계약직, 임시직

**규칙:**
1. 원문의 의미를 유지하되 올바른 단어로 변경
2. 복지 관련 전문 용어 우선 적용
3. 띄어쓰기, 맞춤법 수정
4. 원문과 크게 다르지 않으면 원문 유지

**STT 결과:** "{stt_text}"

**보정 결과 (한 줄만 출력):**"""

    try:
        llm = ChatOpenAI(
            model=settings.stt_llm_model,
            api_key=settings.openai_api_key,
            max_tokens=settings.stt_llm_max_tokens,
            temperature=0.0,
        )
        response = llm.invoke(correction_prompt)
        corrected_text = (response.content or "").strip()

        if corrected_text and corrected_text != stt_text:
            logger.info("[LLM 보정] 수정: '%s' -> '%s'", stt_text, corrected_text)
            return corrected_text

        return stt_text
    except Exception as exc:
        logger.warning("[LLM 보정] 오류: %s", exc)
        return stt_text
