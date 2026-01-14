import { create } from 'zustand';
import { Session, SessionEndResult } from '../types';
import * as api from '../api';

interface SessionState {
  sessionId: string | null;
  session: Session | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  createSession: (email?: string) => Promise<void>;
  endSession: () => Promise<SessionEndResult | null>;
  setEmail: (email: string) => Promise<void>;
  clearSession: () => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  sessionId: null,
  session: null,
  isLoading: false,
  error: null,

  createSession: async (email?: string) => {
    set({ isLoading: true, error: null });
    try {
      const session = await api.createSession(email);
      set({
        sessionId: session.session_id,
        session,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: '세션 생성에 실패했습니다.',
        isLoading: false,
      });
      throw error;
    }
  },

  endSession: async () => {
    const { sessionId } = get();
    if (!sessionId) return null;

    set({ isLoading: true, error: null });
    try {
      const result = await api.endSession(sessionId);
      set({
        sessionId: null,
        session: null,
        isLoading: false,
      });
      return result;
    } catch (error) {
      set({
        error: '세션 종료에 실패했습니다.',
        isLoading: false,
      });
      throw error;
    }
  },

  setEmail: async (email: string) => {
    const { sessionId } = get();
    if (!sessionId) return;

    try {
      await api.updateSessionEmail(sessionId, email);
      set((state) => ({
        session: state.session
          ? { ...state.session, customer_email: email }
          : null,
      }));
    } catch (error) {
      set({ error: '이메일 설정에 실패했습니다.' });
      throw error;
    }
  },

  clearSession: () => {
    set({
      sessionId: null,
      session: null,
      error: null,
    });
  },
}));
