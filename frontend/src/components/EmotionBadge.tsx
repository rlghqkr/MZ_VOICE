import { EmotionType, EMOTION_LABELS } from '../types';

interface EmotionBadgeProps {
  emotion: EmotionType;
  confidence?: number;
  size?: 'sm' | 'md' | 'lg';
}

const EMOTION_GRADIENT: Record<EmotionType, string> = {
  neutral: 'from-gray-400 to-gray-500',
  happy: 'from-green-400 to-emerald-500',
  sad: 'from-blue-400 to-blue-500',
  angry: 'from-red-400 to-red-500',
  fearful: 'from-purple-400 to-purple-500',
  surprised: 'from-amber-400 to-orange-500',
};

const EMOTION_GLOW: Record<EmotionType, string> = {
  neutral: 'shadow-[0_0_10px_rgba(156,163,175,0.5)]',
  happy: 'shadow-[0_0_10px_rgba(74,222,128,0.5)]',
  sad: 'shadow-[0_0_10px_rgba(96,165,250,0.5)]',
  angry: 'shadow-[0_0_10px_rgba(248,113,113,0.5)]',
  fearful: 'shadow-[0_0_10px_rgba(192,132,252,0.5)]',
  surprised: 'shadow-[0_0_10px_rgba(251,191,36,0.5)]',
};

export function EmotionBadge({
  emotion,
  confidence,
  size = 'md',
}: EmotionBadgeProps) {
  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
    lg: 'text-base px-3 py-1.5',
  };

  const gradient = EMOTION_GRADIENT[emotion] || 'from-gray-400 to-gray-500';
  const glow = EMOTION_GLOW[emotion] || '';
  const label = EMOTION_LABELS[emotion] || emotion;

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full text-white font-medium bg-gradient-to-r ${gradient} ${glow} ${sizeClasses[size]}`}
    >
      {label}
      {confidence !== undefined && (
        <span className="opacity-75 text-xs">
          ({Math.round(confidence * 100)}%)
        </span>
      )}
    </span>
  );
}
