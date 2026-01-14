import { useState } from 'react';
import { useVoiceRecording } from '../hooks';
import { useChatStore, useSessionStore } from '../store';

export function VoiceRecorder() {
  const {
    isRecording,
    audioBlob,
    duration,
    startRecording,
    stopRecording,
    clearRecording,
    error: recordingError,
  } = useVoiceRecording();

  const { sendVoiceMessage, isLoading } = useChatStore();
  const { sessionId } = useSessionStore();
  const [isSending, setIsSending] = useState(false);

  const handleSend = async () => {
    if (!audioBlob || isSending) return;

    setIsSending(true);
    try {
      await sendVoiceMessage(audioBlob, sessionId || undefined);
      clearRecording();
    } finally {
      setIsSending(false);
    }
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const isProcessing = isLoading || isSending;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        {/* Recording button with neon effect */}
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
          className={`relative w-16 h-16 rounded-full flex items-center justify-center transition-all duration-300 ${
            isRecording
              ? 'bg-red-500 pulse-recording'
              : 'bg-gradient-to-br from-neon-purple to-neon-pink glow-purple hover:scale-105'
          } disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100`}
        >
          {isRecording ? (
            <svg
              className="w-7 h-7 text-white"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          ) : (
            <svg
              className="w-8 h-8 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
              />
            </svg>
          )}
        </button>

        {/* Recording status */}
        {isRecording && (
          <div className="flex items-center gap-4 flex-1">
            <div className="flex items-center gap-1 h-10">
              {[...Array(5)].map((_, i) => (
                <div
                  key={i}
                  className="w-1.5 bg-gradient-to-t from-neon-pink to-neon-purple rounded-full waveform-bar"
                  style={{
                    height: '100%',
                    animationDelay: `${i * 0.1}s`
                  }}
                />
              ))}
            </div>
            <span className="text-neon-pink font-mono text-xl font-bold neon-text-subtle">
              {formatDuration(duration)}
            </span>
            <span className="text-sm text-gray-400">Recording...</span>
          </div>
        )}

        {/* Idle state hint */}
        {!isRecording && !audioBlob && !isProcessing && (
          <span className="text-gray-400 text-sm">
            Click the microphone to start recording
          </span>
        )}

        {/* Processing state */}
        {isProcessing && !isRecording && (
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 border-2 border-neon-cyan border-t-transparent rounded-full animate-spin glow-cyan" />
            <span className="text-neon-cyan text-sm">Processing...</span>
          </div>
        )}
      </div>

      {/* Send recorded audio */}
      {audioBlob && !isRecording && !isProcessing && (
        <div className="flex items-center gap-4 p-4 glass-light rounded-xl">
          <div className="flex-1">
            <p className="text-sm font-medium text-white">
              Recording Complete
            </p>
            <p className="text-xs text-gray-400 mt-1">
              Duration: {formatDuration(duration)}
            </p>
          </div>
          <button
            onClick={clearRecording}
            className="px-4 py-2 text-sm text-gray-400 hover:text-white glass-light rounded-lg transition-all duration-300"
          >
            Cancel
          </button>
          <button
            onClick={handleSend}
            className="px-5 py-2 btn-neon text-white rounded-lg text-sm font-medium flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
            Send
          </button>
        </div>
      )}

      {/* Error message */}
      {recordingError && (
        <div className="flex items-center gap-3 text-red-400 text-sm p-4 glass-light rounded-xl border border-red-500/30">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {recordingError}
        </div>
      )}
    </div>
  );
}
