import { useEffect, useState } from 'react';
import { useSessionStore } from '../store';
import { getStats, PipelineStats } from '../api';

export function Header() {
  const { sessionId } = useSessionStore();
  const [stats, setStats] = useState<PipelineStats | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await getStats();
        setStats(data);
        setIsConnected(true);
      } catch (error) {
        console.error('Failed to fetch stats:', error);
        setIsConnected(false);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="glass border-b border-white/10 px-6 py-4">
      <div className="flex items-center justify-between max-w-4xl mx-auto">
        <div className="flex items-center gap-4">
          {/* Logo with neon glow */}
          <div className="relative">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-neon-cyan via-neon-blue to-neon-purple flex items-center justify-center glow-gradient">
              <svg
                className="w-7 h-7 text-white"
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
            </div>
          </div>
          <div>
            <h1 className="text-2xl font-bold gradient-text">MZ-VOICE</h1>
            <p className="text-sm text-gray-400">AI Voice Counseling</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* RAG Status */}
          {stats?.rag_stats && (
            <div className="hidden sm:flex items-center gap-2 text-xs glass-light px-4 py-2 rounded-full">
              <svg className="w-4 h-4 text-neon-cyan" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              <span className="text-gray-300">
                RAG: <span className="text-neon-cyan font-medium">{stats.rag_stats.document_count}</span> docs
              </span>
            </div>
          )}

          {/* Connection Status */}
          <div className="flex items-center gap-2 text-sm glass-light px-4 py-2 rounded-full">
            <div className={`w-2 h-2 rounded-full ${
              isConnected
                ? 'bg-green-400 shadow-[0_0_10px_rgba(74,222,128,0.8)]'
                : 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.8)]'
            }`} />
            <span className="text-gray-300">
              {isConnected ? (sessionId ? 'Active' : 'Connected') : 'Disconnected'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
