import { useEffect, useRef } from 'react';
import { useChatStore } from '../store';
import { MessageBubble } from './MessageBubble';

export function MessageList() {
  const { messages, isLoading } = useChatStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          {/* Animated logo */}
          <div className="relative w-24 h-24 mx-auto mb-6">
            <div className="absolute inset-0 rounded-full bg-gradient-to-br from-neon-cyan via-neon-purple to-neon-pink opacity-20 blur-xl animate-pulse" />
            <div className="relative w-24 h-24 rounded-full glass-light flex items-center justify-center floating">
              <svg
                className="w-12 h-12 text-neon-cyan"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
            </div>
          </div>
          <h3 className="text-xl font-semibold gradient-text mb-2">
            Start a Conversation
          </h3>
          <p className="text-gray-500 text-sm">
            Type a message or use voice recording
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}

      {/* Loading indicator */}
      {isLoading && (
        <div className="flex justify-start mb-4">
          <div className="glass-light rounded-2xl px-5 py-3 rounded-tl-sm">
            <div className="flex items-center gap-2">
              <div
                className="w-2 h-2 rounded-full bg-neon-cyan animate-bounce"
                style={{ boxShadow: '0 0 10px rgba(0, 245, 255, 0.8)' }}
              />
              <div
                className="w-2 h-2 rounded-full bg-neon-purple animate-bounce"
                style={{ animationDelay: '0.1s', boxShadow: '0 0 10px rgba(139, 92, 246, 0.8)' }}
              />
              <div
                className="w-2 h-2 rounded-full bg-neon-pink animate-bounce"
                style={{ animationDelay: '0.2s', boxShadow: '0 0 10px rgba(236, 72, 153, 0.8)' }}
              />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
