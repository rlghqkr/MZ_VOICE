// Emotion types
export type EmotionType = 'angry' | 'happy' | 'sad' | 'neutral' | 'fearful' | 'surprised';

export interface EmotionInfo {
  emotion: EmotionType;
  korean_label: string;
  confidence: number;
}

// Message types
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  emotion?: EmotionType;
  audioUrl?: string;
}

// Session types
export interface Session {
  session_id: string;
  customer_email?: string;
  created_at: string;
  ended_at?: string;
  message_count: number;
}

export interface SessionEndResult {
  session_id: string;
  success: boolean;
  summary?: string;
  mail_sent: boolean;
  error?: string;
}

export interface SummarizeResult {
  summary: string;
  success: boolean;
  error?: string;
}

// API Response types
export interface VoiceProcessResponse {
  transcription: string;
  response_text: string;
  emotion: EmotionInfo;
  audio_url?: string;
  processing_time: number;
}

export interface TextProcessResponse {
  response_text: string;
  audio_url?: string;
  processing_time: number;
}

// Emotion labels and colors
export const EMOTION_LABELS: Record<EmotionType, string> = {
  angry: '화남',
  happy: '기쁨',
  sad: '슬픔',
  neutral: '보통',
  fearful: '불안',
  surprised: '놀람',
};

export const EMOTION_COLORS: Record<EmotionType, string> = {
  angry: 'bg-red-500',
  happy: 'bg-green-500',
  sad: 'bg-blue-500',
  neutral: 'bg-gray-500',
  fearful: 'bg-purple-500',
  surprised: 'bg-amber-500',
};
