"""
Audio Emotion Analyzer Implementation

오디오에서 감정을 인식하는 구현체입니다.
emotion2vec 모델을 사용하여 감정을 인식합니다.
"""

import logging
from typing import Dict
import random
import io

from .base import EmotionAnalyzerBase, EmotionResult, Emotion

logger = logging.getLogger(__name__)


class AudioEmotionAnalyzer(EmotionAnalyzerBase):
    """
    오디오 감정 분석기

    음성 특징(피치, 에너지, 속도 등)을 분석하여
    화자의 감정 상태를 추론합니다.

    Note:
        emotion2vec 모델을 사용합니다:
        - FunASR: emotion2vec/emotion2vec_plus_base

    Example:
        analyzer = AudioEmotionAnalyzer()
        result = analyzer.analyze(audio_bytes)
        print(result.primary_emotion, result.confidence)
    """
    
    def __init__(self, use_mock: bool = True):
        """
        Args:
            use_mock: True면 Mock 결과 반환 (개발용)
        """
        self.use_mock = use_mock
        self._model = None
        
        if not use_mock:
            self._load_model()
    
    def _load_model(self):
        """감정 인식 모델 로드 (emotion2vec)"""
        try:
            from funasr import AutoModel
            self._model = AutoModel(model="iic/emotion2vec_plus_base")
            logger.info("emotion2vec model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load emotion model, using mock: {e}")
            self.use_mock = True
    
    def analyze(self, audio: bytes) -> EmotionResult:
        """
        오디오에서 감정 분석
        
        Args:
            audio: WAV 형식 오디오 데이터
            
        Returns:
            EmotionResult: 감정 분석 결과
        """
        if self.use_mock:
            return self._mock_analyze(audio)
        
        return self._real_analyze(audio)
    
    def _mock_analyze(self, audio: bytes) -> EmotionResult:
        """
        Mock 감정 분석 (개발/테스트용)
        
        오디오 길이와 간단한 특징으로 감정을 시뮬레이션합니다.
        """
        # 오디오 특징 기반 간단한 휴리스틱
        audio_length = len(audio)
        
        # Mock: 오디오 길이에 따라 다른 감정 시뮬레이션
        if audio_length < 10000:
            # 짧은 발화 → neutral 경향
            primary = Emotion.NEUTRAL
            scores = {
                "neutral": 0.7,
                "happy": 0.15,
                "angry": 0.05,
                "sad": 0.05,
                "fearful": 0.03,
                "surprised": 0.02
            }
        elif audio_length < 50000:
            # 중간 발화 → 다양한 감정
            emotions = [Emotion.NEUTRAL, Emotion.HAPPY, Emotion.ANGRY]
            primary = random.choice(emotions)
            scores = self._generate_mock_scores(primary)
        else:
            # 긴 발화 → 복잡한 감정
            emotions = list(Emotion)
            primary = random.choice(emotions)
            scores = self._generate_mock_scores(primary)
        
        return EmotionResult(
            primary_emotion=primary,
            confidence=scores[primary.value],
            emotion_scores=scores
        )
    
    def _generate_mock_scores(self, primary: Emotion) -> Dict[str, float]:
        """Mock 점수 생성"""
        scores = {}
        remaining = 1.0
        primary_score = random.uniform(0.5, 0.8)
        scores[primary.value] = primary_score
        remaining -= primary_score
        
        # 나머지 감정에 점수 분배
        other_emotions = [e for e in Emotion if e != primary]
        for i, emotion in enumerate(other_emotions):
            if i == len(other_emotions) - 1:
                scores[emotion.value] = remaining
            else:
                score = random.uniform(0, remaining * 0.5)
                scores[emotion.value] = score
                remaining -= score
        
        return scores
    
    def _real_analyze(self, audio: bytes) -> EmotionResult:
        """
        emotion2vec 모델을 사용한 감정 분석
        """
        import numpy as np
        import soundfile as sf
        import tempfile
        import os

        # 오디오 로드
        audio_io = io.BytesIO(audio)
        y, sr = sf.read(audio_io)

        # 모노 변환 (스테레오인 경우)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # 16kHz로 리샘플링 (emotion2vec 요구사항)
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # 임시 파일로 저장 (funasr는 파일 경로 필요)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, y, sr)

        try:
            # 모델 예측
            result = self._model.generate(tmp_path, granularity="utterance", extract_embedding=False)
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # emotion2vec 라벨 매핑
        # emotion2vec labels: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
        label_map = {
            "angry": Emotion.ANGRY,
            "happy": Emotion.HAPPY,
            "sad": Emotion.SAD,
            "neutral": Emotion.NEUTRAL,
            "fearful": Emotion.FEARFUL,
            "surprised": Emotion.SURPRISED,
            "disgusted": Emotion.ANGRY,  # disgust → angry로 매핑
            "other": Emotion.NEUTRAL,    # other → neutral로 매핑
            "unknown": Emotion.NEUTRAL,  # unknown → neutral로 매핑
        }

        # 결과 파싱
        # result 형식: [{'labels': [...], 'scores': [...], 'feats': ...}]
        labels = result[0].get("labels", [])
        scores = result[0].get("scores", [])

        # 점수 변환
        emotion_scores = {}
        for label, score in zip(labels, scores):
            label_lower = label.lower()
            if label_lower in label_map:
                emotion = label_map[label_lower]
                # 같은 emotion에 여러 라벨이 매핑될 수 있으므로 최대값 사용
                current_score = emotion_scores.get(emotion.value, 0.0)
                emotion_scores[emotion.value] = max(current_score, float(score))

        # 누락된 감정에 0점 부여
        for emotion in Emotion:
            if emotion.value not in emotion_scores:
                emotion_scores[emotion.value] = 0.0

        # 주요 감정 결정 (가장 높은 점수)
        if labels and scores:
            max_idx = int(np.argmax(scores))
            primary_label = labels[max_idx].lower()
            primary = label_map.get(primary_label, Emotion.NEUTRAL)
            confidence = float(scores[max_idx])
        else:
            primary = Emotion.NEUTRAL
            confidence = 0.0

        return EmotionResult(
            primary_emotion=primary,
            confidence=confidence,
            emotion_scores=emotion_scores
        )
    
    def analyze_features(self, audio: bytes) -> Dict[str, float]:
        """
        오디오 특징 추출 (디버깅용)
        
        Returns:
            Dict: 피치, 에너지, 속도 등의 특징
        """
        try:
            import librosa
            import numpy as np
            import io
            import soundfile as sf
            
            audio_io = io.BytesIO(audio)
            y, sr = sf.read(audio_io)
            
            # 기본 특징 추출
            features = {
                "duration": len(y) / sr,
                "rms_energy": float(np.sqrt(np.mean(y ** 2))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
            }
            
            # 피치 추출 (선택적)
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
                features["pitch_mean"] = float(pitch_mean)
            except:
                features["pitch_mean"] = 0.0
            
            return features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return {"error": str(e)}


class MockEmotionAnalyzer(EmotionAnalyzerBase):
    """항상 Neutral을 반환하는 테스트용 분석기"""
    
    def analyze(self, audio: bytes) -> EmotionResult:
        return EmotionResult(
            primary_emotion=Emotion.NEUTRAL,
            confidence=0.95,
            emotion_scores={
                "neutral": 0.95,
                "happy": 0.02,
                "angry": 0.01,
                "sad": 0.01,
                "fearful": 0.005,
                "surprised": 0.005
            }
        )
