import { Message } from '../types';
import { EmotionBadge } from './EmotionBadge';

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 message-animate`}
    >
      <div
        className={`max-w-[80%] ${
          isUser ? 'order-2' : 'order-1'
        }`}
      >
        {/* Avatar and timestamp */}
        <div
          className={`flex items-center gap-2 mb-2 ${
            isUser ? 'flex-row-reverse' : 'flex-row'
          }`}
        >
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              isUser
                ? 'bg-gradient-to-br from-neon-cyan to-neon-blue text-white shadow-neon-cyan'
                : 'bg-gradient-to-br from-neon-purple to-neon-pink text-white shadow-neon-purple'
            }`}
          >
            {isUser ? 'U' : 'AI'}
          </div>
          <span className="text-xs text-gray-500">
            {message.timestamp.toLocaleTimeString('ko-KR', {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </span>
          {isUser && message.emotion && (
            <EmotionBadge emotion={message.emotion} size="sm" />
          )}
        </div>

        {/* Message content */}
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-gradient-to-r from-neon-cyan/20 to-neon-blue/20 text-white border border-neon-cyan/30 rounded-tr-sm'
              : 'glass-light text-gray-200 rounded-tl-sm'
          }`}
        >
          <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
        </div>

        {/* Audio player */}
        {message.audioUrl && (
          <div className="mt-3">
            <audio
              controls
              src={message.audioUrl}
              className="h-10 w-full max-w-xs rounded-lg"
              style={{
                filter: 'invert(1) hue-rotate(180deg)',
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
