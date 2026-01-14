import { Header, ChatContainer } from './components';

function App() {
  return (
    <div className="min-h-screen bg-dark-400 flex flex-col relative overflow-hidden">
      {/* Animated background orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div
          className="bg-orb w-96 h-96 bg-neon-cyan/20"
          style={{ top: '-10%', left: '-5%' }}
        />
        <div
          className="bg-orb w-80 h-80 bg-neon-purple/20"
          style={{ top: '40%', right: '-10%', animationDelay: '-2s' }}
        />
        <div
          className="bg-orb w-64 h-64 bg-neon-pink/20"
          style={{ bottom: '-5%', left: '30%', animationDelay: '-4s' }}
        />
      </div>

      {/* Main content */}
      <div className="relative z-10 flex flex-col min-h-screen">
        <Header />

        <main className="flex-1 max-w-4xl w-full mx-auto p-4">
          <div className="h-[calc(100vh-140px)]">
            <ChatContainer />
          </div>
        </main>

        <footer className="text-center py-4 text-sm text-gray-500">
          <span className="gradient-text font-medium">MZ-VOICE</span>
          <span className="text-gray-600 ml-2">AI Voice Counseling Service</span>
        </footer>
      </div>
    </div>
  );
}

export default App;
