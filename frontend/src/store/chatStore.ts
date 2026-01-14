import { create } from 'zustand';
import { Message, EmotionType, EmotionInfo } from '../types';
import * as api from '../api';

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  lastEmotion: EmotionInfo | null;

  // Actions
  sendTextMessage: (
    text: string,
    emotion: EmotionType,
    sessionId?: string
  ) => Promise<void>;
  sendVoiceMessage: (audioBlob: Blob, sessionId?: string) => Promise<void>;
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void;
  updateLastAssistantMessage: (content: string) => void;
  clearMessages: () => void;
  setError: (error: string | null) => void;
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isLoading: false,
  error: null,
  lastEmotion: null,

  sendTextMessage: async (
    text: string,
    emotion: EmotionType = 'neutral',
    sessionId?: string
  ) => {
    // Add user message immediately
    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: text,
      timestamp: new Date(),
      emotion,
    };

    set((state) => ({
      messages: [...state.messages, userMessage],
      isLoading: true,
      error: null,
    }));

    try {
      const response = await api.processText(text, emotion, sessionId, false);

      // Add assistant message
      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.response_text,
        timestamp: new Date(),
        audioUrl: response.audio_url || undefined,
      };

      set((state) => ({
        messages: [...state.messages, assistantMessage],
        isLoading: false,
      }));
    } catch (error: any) {
      console.error('Failed to send text message:', error);
      const errorMessage = error.response?.data?.detail || '메시지 전송에 실패했습니다.';
      set({
        error: errorMessage,
        isLoading: false,
      });
    }
  },

  sendVoiceMessage: async (audioBlob: Blob, sessionId?: string) => {
    set({ isLoading: true, error: null });

    try {
      const response = await api.processVoice(audioBlob, sessionId, false);

      // Add user message (transcribed)
      const userMessage: Message = {
        id: generateId(),
        role: 'user',
        content: response.transcription || '(음성 인식 실패)',
        timestamp: new Date(),
        emotion: response.emotion.emotion,
      };

      // Add assistant message
      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.response_text,
        timestamp: new Date(),
        audioUrl: response.audio_url || undefined,
      };

      set((state) => ({
        messages: [...state.messages, userMessage, assistantMessage],
        isLoading: false,
        lastEmotion: response.emotion,
        error: null,
      }));
    } catch (error: any) {
      console.error('Failed to send voice message:', error);
      const errorMessage = error.response?.data?.detail || '음성 처리에 실패했습니다.';
      set({
        error: errorMessage,
        isLoading: false,
      });
    }
  },

  addMessage: (message) => {
    const newMessage: Message = {
      ...message,
      id: generateId(),
      timestamp: new Date(),
    };

    set((state) => ({
      messages: [...state.messages, newMessage],
    }));
  },

  updateLastAssistantMessage: (content: string) => {
    set((state) => {
      const messages = [...state.messages];
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'assistant') {
          messages[i] = { ...messages[i], content };
          break;
        }
      }
      return { messages };
    });
  },

  clearMessages: () => {
    set({ messages: [], lastEmotion: null, error: null });
  },

  setError: (error) => {
    set({ error });
  },
}));
