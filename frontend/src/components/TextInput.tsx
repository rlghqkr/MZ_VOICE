import { useState, KeyboardEvent } from 'react';
import { useChatStore, useSessionStore } from '../store';
import { EmotionType, EMOTION_LABELS } from '../types';

const EMOTION_COLORS: Record<EmotionType, string> = {
  neutral: 'from-gray-400 to-gray-500',
  happy: 'from-green-400 to-emerald-500',
  sad: 'from-blue-400 to-blue-500',
  angry: 'from-red-400 to-red-500',
  fearful: 'from-purple-400 to-purple-500',
  surprised: 'from-amber-400 to-orange-500',
};

export function TextInput() {
  const [text, setText] = useState('');
  const [emotion, setEmotion] = useState<EmotionType>('neutral');

  const { sendTextMessage, isLoading } = useChatStore();
  const { sessionId } = useSessionStore();

  const handleSend = async () => {
    if (!text.trim() || isLoading) return;

    await sendTextMessage(text.trim(), emotion, sessionId || undefined);
    setText('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const emotions: EmotionType[] = [
    'neutral',
    'happy',
    'sad',
    'angry',
    'fearful',
    'surprised',
  ];

  return (
    <div className="flex flex-col gap-4">
      {/* Emotion selector */}
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-400">Emotion:</span>
        <div className="flex gap-2 flex-wrap">
          {emotions.map((e) => (
            <button
              key={e}
              onClick={() => setEmotion(e)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-300 ${
                emotion === e
                  ? `bg-gradient-to-r ${EMOTION_COLORS[e]} text-white shadow-lg`
                  : 'glass-light text-gray-400 hover:text-white'
              }`}
            >
              {EMOTION_LABELS[e]}
            </button>
          ))}
        </div>
      </div>

      {/* Text input */}
      <div className="flex items-end gap-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Enter to send)"
          disabled={isLoading}
          rows={1}
          className="flex-1 resize-none rounded-xl input-dark px-4 py-3 text-white placeholder-gray-500 focus:ring-2 focus:ring-neon-cyan/50 disabled:opacity-50 disabled:cursor-not-allowed"
          style={{ minHeight: '52px', maxHeight: '120px' }}
        />
        <button
          onClick={handleSend}
          disabled={!text.trim() || isLoading}
          className="px-6 py-3 btn-neon text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
        >
          {isLoading ? (
            <svg
              className="w-5 h-5 animate-spin"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          ) : (
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
