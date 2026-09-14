"use client";

import React, { useState, useRef, useEffect } from 'react';
import { PaperAirplaneIcon, ShieldCheckIcon, ShieldExclamationIcon } from '@heroicons/react/24/solid';
import { motion } from 'framer-motion';

type Turn = {
  role: 'customer' | 'brand';
  text: string;
};

type Insight = {
  intent: string;
  decision: string;
  risk_flags: string[];
  evidence_used: boolean;
  gate_reasons: string[];
};

export default function ChatInterface() {
  const [messages, setMessages] = useState<Turn[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [lastInsight, setLastInsight] = useState<Insight | null>(null);
  
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputValue.trim()) return;

    const newMessage: Turn = { role: 'customer', text: inputValue };
    const currentHistory = [...messages, newMessage];
    
    setMessages(currentHistory);
    setInputValue('');
    setLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: newMessage.text,
          history: messages,
        }),
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }

      const data = await response.json();
      
      setMessages((prev) => [...prev, { role: 'brand', text: data.reply }]);
      setLastInsight({
        intent: data.intent,
        decision: data.decision,
        risk_flags: data.risk_flags,
        evidence_used: data.evidence_used,
        gate_reasons: data.gate_reasons || [],
      });
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [...prev, { role: 'brand', text: "Error: Could not connect to support agent." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-full overflow-hidden p-4 md:p-8 gap-6 max-w-7xl mx-auto">
      {/* Left: Chat Area */}
      <div className="flex-1 flex flex-col glass-panel overflow-hidden shadow-2xl relative">
        {/* Header */}
        <header className="px-6 py-4 border-b border-[rgba(255,255,255,0.1)] flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-[#1db954] flex items-center justify-center shrink-0">
             <svg className="w-5 h-5 text-black" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.6.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.239.54-.959.72-1.56.3z"/>
             </svg>
          </div>
          <div>
            <h1 className="font-semibold text-lg">Spotify AI Support</h1>
            <p className="text-xs text-gray-400">Gemini 2.5 Flash Agent</p>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
             <div className="h-full flex flex-col items-center justify-center text-gray-500 text-sm space-y-4">
                 <div className="p-4 rounded-full bg-[rgba(255,255,255,0.05)] mb-2">
                    <ShieldCheckIcon className="w-12 h-12 text-[#1db954] opacity-80" />
                 </div>
                 <p>Send a message to start simulating a support ticket.</p>
                 <div className="flex flex-wrap gap-2 justify-center max-w-lg mt-4">
                    <button onClick={() => setInputValue("I was charged twice this month for my family plan.")} className="px-3 py-1.5 rounded-full border border-gray-700 hover:bg-gray-800 transition text-xs">"I was charged twice this month..."</button>
                    <button onClick={() => setInputValue("How do I add a new song to my playlist?")} className="px-3 py-1.5 rounded-full border border-gray-700 hover:bg-gray-800 transition text-xs">"How do I add a new song to my playlist?"</button>
                    <button onClick={() => setInputValue("My music keeps pausing on my iPhone.")} className="px-3 py-1.5 rounded-full border border-gray-700 hover:bg-gray-800 transition text-xs">"My music keeps pausing on my iPhone."</button>
                 </div>
             </div>
          )}
          
          {messages.map((msg, idx) => (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              key={idx} 
              className={`flex ${msg.role === 'customer' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[75%] rounded-2xl px-5 py-3 ${
                msg.role === 'customer' 
                  ? 'bg-gradient-to-r from-[#1db954] to-[#1ed760] text-black font-medium' 
                  : 'bg-[rgba(255,255,255,0.08)] text-gray-100 border border-[rgba(255,255,255,0.05)]'
              }`}>
                <p className="text-[15px] whitespace-pre-wrap leading-relaxed">{msg.text}</p>
              </div>
            </motion.div>
          ))}
          
          {loading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
               <div className="max-w-[75%] rounded-2xl px-5 py-4 bg-[rgba(255,255,255,0.08)] flex items-center gap-2">
                 <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                 <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                 <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }} />
               </div>
            </motion.div>
          )}
          <div ref={endOfMessagesRef} />
        </div>

        {/* Input */}
        <form onSubmit={handleSend} className="p-4 border-t border-[rgba(255,255,255,0.1)] bg-[rgba(0,0,0,0.2)]">
          <div className="relative flex items-center">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type customer message..."
              className="w-full bg-[rgba(255,255,255,0.05)] border border-[rgba(255,255,255,0.1)] rounded-full pl-5 pr-12 py-3.5 focus:outline-none focus:border-[#1db954] transition text-sm shadow-inner"
              disabled={loading}
            />
            <button 
              type="submit" 
              disabled={loading || !inputValue.trim()}
              className="absolute right-2 w-9 h-9 rounded-full bg-[#1db954] flex items-center justify-center text-black disabled:opacity-50 hover:bg-[#1ed760] transition"
            >
              <PaperAirplaneIcon className="w-4 h-4" />
            </button>
          </div>
        </form>
      </div>

      {/* Right: Insights Panel */}
      <div className="w-80 flex-shrink-0 flex flex-col gap-4">
        <div className="glass-panel p-5 h-full overflow-y-auto">
          <h2 className="font-semibold text-lg mb-6 flex items-center gap-2">
            <ShieldCheckIcon className="w-5 h-5 text-gray-400" />
            Agent Insights
          </h2>

          {!lastInsight ? (
            <div className="text-gray-500 text-sm text-center py-10">
              No insights yet. Send a message to see routing decisions.
            </div>
          ) : (
            <div className="space-y-6">
              
              {/* Decision Badge */}
              <div className="space-y-2">
                <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Routing Decision</span>
                {lastInsight.decision === 'AUTO_HANDLE' ? (
                  <div className="flex items-center gap-2 p-3 bg-green-500/10 border border-green-500/20 text-green-400 rounded-lg font-medium text-sm">
                    <ShieldCheckIcon className="w-5 h-5" />
                    Auto-Handle
                  </div>
                ) : (
                  <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg font-medium text-sm">
                    <ShieldExclamationIcon className="w-5 h-5" />
                    Escalate to Human
                  </div>
                )}
              </div>

              {/* Intent */}
              <div className="space-y-2">
                <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Detected Intent</span>
                <div className="p-3 bg-[rgba(255,255,255,0.05)] rounded-lg text-sm text-gray-200 border border-[rgba(255,255,255,0.05)]">
                  {lastInsight.intent}
                </div>
              </div>

               {/* Grounding */}
               <div className="space-y-2">
                <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Retrieval</span>
                <div className="p-3 bg-[rgba(255,255,255,0.05)] rounded-lg text-sm text-gray-200 border border-[rgba(255,255,255,0.05)] flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${lastInsight.evidence_used ? 'bg-green-500' : 'bg-gray-500'}`}></div>
                  {lastInsight.evidence_used ? 'Historical Evidence Found' : 'No Evidence Sourced'}
                </div>
              </div>

              {/* Risk Flags */}
              <div className="space-y-2">
                <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Risk Flags</span>
                <div className="flex flex-wrap gap-2">
                  {lastInsight.risk_flags.length > 0 ? lastInsight.risk_flags.map(flag => (
                     <span key={flag} className="px-2.5 py-1 bg-yellow-500/10 text-yellow-500 border border-yellow-500/20 rounded text-xs font-medium">
                       {flag}
                     </span>
                  )) : (
                     <span className="text-gray-500 text-sm">None detected</span>
                  )}
                </div>
              </div>

              {/* Gate Reasons */}
              <div className="space-y-2">
                <span className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Gate Diagnostics</span>
                <div className="flex flex-col gap-2">
                  {lastInsight.gate_reasons.map(reason => (
                     <span key={reason} className={`px-2.5 py-1.5 rounded text-xs font-medium ${
                       reason === 'GATE_PASSED' 
                       ? 'bg-green-500/10 text-green-400 border border-green-500/20' 
                       : 'bg-red-500/10 text-red-400 border border-red-500/20'
                     }`}>
                       {reason}
                     </span>
                  ))}
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}
