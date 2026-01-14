"""
Emotion Recognition Base Class

음성에서 감정을 인식하는 서비스의 추상 베이스 클래스입니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class Emotion(Enum):
    """감정 열거형"""
    ANGRY = "angry"
    HAPPY = "happy"
    SAD = "sad"
    NEUTRAL = "neutral"
    FEARFUL = "fearful"
    SURPRISED = "surprised"


@dataclass
class EmotionResult:
    """감정 인식 결과"""
    primary_emotion: Emotion           # 주요 감정
    confidence: float                  # 신뢰도 (0.0 ~ 1.0)
    emotion_scores: Dict[str, float]   # 각 감정별 점수
    
    @property
    def korean_label(self) -> str:
        """감정의 한국어 라벨"""
        labels = {
            Emotion.ANGRY: "화남",
            Emotion.HAPPY: "기쁨",
            Emotion.SAD: "슬픔",
            Emotion.NEUTRAL: "보통",
            Emotion.FEARFUL: "불안",
            Emotion.SURPRISED: "놀람",
        }
        return labels.get(self.primary_emotion, "알수없음")


class EmotionAnalyzerBase(ABC):
    """
    감정 인식 추상 베이스 클래스
    
    음성 데이터에서 화자의 감정을 분석합니다.
    분석된 감정은 RAG 프롬프트에 활용되어
    고객 감정에 맞는 응대를 가능하게 합니다.
    
    Example:
        analyzer = AudioEmotionAnalyzer()
        result = analyzer.analyze(audio_bytes)
        print(f"감정: {result.korean_label} (신뢰도: {result.confidence:.2%})")
    """
    
    @abstractmethod
    def analyze(self, audio: bytes) -> EmotionResult:
        """
        오디오에서 감정을 분석합니다.
        
        Args:
            audio: 오디오 데이터 (WAV 형식)
            
        Returns:
            EmotionResult: 감정 분석 결과
        """
        pass
    
    def get_response_style(self, emotion: Emotion) -> str:
        """
        감정에 따른 응답 스타일 가이드 반환
        
        Args:
            emotion: 분석된 감정
            
        Returns:
            str: 응답 스타일 가이드 (프롬프트에 포함)
        """
        styles = {
            Emotion.ANGRY: (
                "고객님께서 불편함을 느끼고 계십니다. "
                "차분하고 공감하는 어조로 문제 해결에 집중해주세요. "
                "먼저 사과의 말씀을 전하고, 신속한 해결을 약속해주세요."
            ),
            Emotion.HAPPY: (
                "고객님께서 긍정적인 상태입니다. "
                "밝고 친근한 어조로 응대하며, 좋은 경험을 강화해주세요."
            ),
            Emotion.SAD: (
                "고객님께서 실망하거나 속상해하고 계십니다. "
                "따뜻하고 위로하는 어조로 공감을 표현해주세요."
            ),
            Emotion.NEUTRAL: (
                "전문적이고 친절한 어조로 정확한 정보를 제공해주세요."
            ),
            Emotion.FEARFUL: (
                "고객님께서 불안해하고 계십니다. "
                "안심시키는 어조로 명확한 정보를 제공해주세요."
            ),
            Emotion.SURPRISED: (
                "상황을 명확히 설명하며, 이해하기 쉽게 안내해주세요."
            ),
        }
        return styles.get(emotion, styles[Emotion.NEUTRAL])
