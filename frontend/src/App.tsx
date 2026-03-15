import React, { useState } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';
import { useClaude } from './useClaude';

function App() {
  const [input, setInput] = useState('');
  const { messages, sendMessage, isLoading } = useClaude();

  const handleSend = () => {
    if (!input.trim() || isLoading) return;
    sendMessage(input);
    setInput('');
  };

  return (
    <div className="flex flex-col h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Header */}
      <header className="p-4 border-b bg-white flex items-center gap-2 shadow-sm">
        <Bot className="text-orange-600" />
        <h1 className="font-semibold text-lg">Claude 4.6 Prototype</h1>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 space-y-4 max-w-3xl w-full mx-auto">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-4 rounded-2xl shadow-sm ${
              msg.role === 'user' 
                ? 'bg-orange-600 text-white rounded-tr-none' 
                : 'bg-white border border-slate-200 rounded-tl-none'
            }`}>
              <div className="flex items-center gap-2 mb-1 opacity-70 text-xs font-bold uppercase tracking-wider">
                {msg.role === 'user' ? <User size={12}/> : <Bot size={12}/>}
                {msg.role}
              </div>
              <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start animate-pulse">
            <div className="bg-white border border-slate-200 p-4 rounded-2xl rounded-tl-none shadow-sm">
              <Loader2 className="animate-spin text-slate-400" />
            </div>
          </div>
        )}
      </main>

      {/* Input Area */}
      <footer className="p-6 bg-white border-t">
        <div className="max-w-3xl mx-auto flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
            placeholder="Send a message to Claude..."
            className="flex-1 p-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-orange-500 focus:outline-none resize-none h-12"
          />
          <button 
            onClick={handleSend}
            disabled={isLoading}
            className="bg-orange-600 hover:bg-orange-700 text-white p-3 rounded-xl transition-colors disabled:opacity-50"
          >
            <Send size={20} />
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;