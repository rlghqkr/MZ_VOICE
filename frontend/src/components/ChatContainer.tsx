import { useState, useEffect } from 'react';
import { useSessionStore, useChatStore } from '../store';
import { MessageList } from './MessageList';
import { TextInput } from './TextInput';
import { VoiceRecorder } from './VoiceRecorder';
import { SummarizeResult } from '../types';
import * as api from '../api';

type InputMode = 'text' | 'voice';

export function ChatContainer() {
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [showSummary, setShowSummary] = useState(false);
  const [summaryResult, setSummaryResult] = useState<SummarizeResult | null>(null);
  const [isSummarizing, setIsSummarizing] = useState(false);

  const {
    sessionId,
    createSession,
    clearSession,
    isLoading: sessionLoading,
    error: sessionError,
  } = useSessionStore();

  const { messages, clearMessages, error: chatError } = useChatStore();

  useEffect(() => {
    if (!sessionId && !sessionLoading) {
      createSession().catch(console.error);
    }
  }, [sessionId, sessionLoading, createSession]);

  const handleEndSession = async () => {
    if (messages.length === 0) {
      // 대화가 없으면 바로 새 세션 시작
      clearMessages();
      clearSession();
      await createSession();
      return;
    }

    setIsSummarizing(true);
    try {
      // 대화 내용을 직접 전달하여 요약 요청
      const result = await api.summarizeConversation(messages);
      setSummaryResult(result);
      setShowSummary(true);
    } catch (error) {
      console.error('Failed to summarize:', error);
      setSummaryResult({
        summary: '',
        success: false,
        error: '요약 생성에 실패했습니다.',
      });
      setShowSummary(true);
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleNewSession = async () => {
    setShowSummary(false);
    setSummaryResult(null);
    clearMessages();
    clearSession();
    await createSession();
  };

  const error = sessionError || chatError;

  return (
    <div className="flex flex-col h-full card-dark overflow-hidden">
      {/* Chat header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
        <div className="flex items-center gap-2">
          {/* Input mode toggle */}
          <div className="flex items-center gap-1 p-1 rounded-xl glass-light">
            <button
              onClick={() => setInputMode('text')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                inputMode === 'text'
                  ? 'bg-gradient-to-r from-neon-cyan to-neon-blue text-white shadow-neon-cyan'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                Text
              </span>
            </button>
            <button
              onClick={() => setInputMode('voice')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                inputMode === 'voice'
                  ? 'bg-gradient-to-r from-neon-purple to-neon-pink text-white shadow-neon-purple'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
                Voice
              </span>
            </button>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {sessionId && (
            <span className="text-xs text-gray-500 font-mono">
              #{sessionId.slice(-8)}
            </span>
          )}
          <button
            onClick={handleEndSession}
            disabled={sessionLoading || isSummarizing || !sessionId}
            className="px-4 py-2 text-sm text-red-400 hover:text-red-300 glass-light rounded-lg transition-all duration-300 hover:shadow-[0_0_15px_rgba(239,68,68,0.3)] disabled:opacity-50 disabled:hover:shadow-none"
          >
            {isSummarizing ? 'Summarizing...' : 'End Session'}
          </button>
        </div>
      </div>

      {/* Session loading state */}
      {sessionLoading && !sessionId && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-12 h-12 rounded-full border-2 border-neon-cyan border-t-transparent animate-spin mx-auto mb-4 glow-cyan" />
            <p className="text-gray-400">Creating session...</p>
          </div>
        </div>
      )}

      {/* Messages area */}
      {sessionId && <MessageList />}

      {/* Error display */}
      {error && (
        <div className="mx-4 mb-2 px-4 py-3 glass-light rounded-xl border border-red-500/30 text-red-400 text-sm flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {error}
        </div>
      )}

      {/* Input area */}
      {sessionId && (
        <div className="p-4 border-t border-white/10">
          {inputMode === 'text' ? <TextInput /> : <VoiceRecorder />}
        </div>
      )}

      {/* Session summary modal */}
      {showSummary && summaryResult && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="card-dark p-8 max-w-lg w-full mx-4 glow-gradient">
            <h3 className="text-2xl font-bold gradient-text mb-6">
              Session Completed
            </h3>

            {summaryResult.success && summaryResult.summary && (
              <div className="mb-6">
                <h4 className="text-sm font-medium text-gray-400 mb-3">
                  Session Summary
                </h4>
                <p className="text-gray-200 glass-light p-4 rounded-xl whitespace-pre-wrap leading-relaxed">
                  {summaryResult.summary}
                </p>
              </div>
            )}

            {summaryResult.error && (
              <p className="text-sm text-red-400 mb-6 flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
                Error: {summaryResult.error}
              </p>
            )}

            <button
              onClick={handleNewSession}
              className="w-full py-4 btn-neon text-white rounded-xl font-semibold text-lg"
            >
              Start New Session
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
