"use client";

import { Terminal, Send, TerminalSquare } from "lucide-react";
import { useState } from "react";

export default function AIStrategist() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState([
    { role: "system", content: "KRONECTOR TERMINAL v2.0 READY." },
    { role: "system", content: "LLAMA-3.3-70B CONNECTED." },
    { role: "user", content: "Who wins Canada 2026?" },
    { role: "assistant", content: "> PREDICTING: Canadian GP 2026...\n> MODEL: LightGBM / SHAP\n> RESULT: Kimi Antonelli (45.2% win probability).\n> DOMINANT FACTORS: Grid Position (SHAP: +3.18), Driver Form (SHAP: +1.39).\n> CRITIQUE AGENT: APPROVED. Confidence normal." }
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    setHistory([...history, { role: "user", content: input }]);
    setInput("");
    setTimeout(() => {
      setHistory(prev => [...prev, { role: "assistant", content: "> PROCESSING QUERY...\n> SIMULATING RACE PARAMETERS...\n> NO DATA LEAKAGE DETECTED." }]);
    }, 500);
  };

  return (
    <div className="bg-[#050816] border border-[#1a1f35] rounded-lg overflow-hidden flex flex-col h-[400px]">
      <div className="bg-[#0a0d1e] border-b border-[#1a1f35] p-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TerminalSquare className="w-4 h-4 text-[#00E5FF]" />
          <span className="text-xs font-bold text-white uppercase tracking-wider">AI Strategist Terminal</span>
        </div>
        <div className="flex gap-1.5">
          <div className="w-2.5 h-2.5 rounded-full bg-[#1a1f35]" />
          <div className="w-2.5 h-2.5 rounded-full bg-[#1a1f35]" />
          <div className="w-2.5 h-2.5 rounded-full bg-[#00D084] animate-pulse" />
        </div>
      </div>

      <div className="flex-1 p-4 overflow-y-auto font-mono text-[11px] space-y-3">
        {history.map((msg, i) => (
          <div key={i} className={`flex flex-col ${msg.role === 'user' ? 'text-[#00E5FF]' : msg.role === 'system' ? 'text-[#A0A0A0]' : 'text-white'}`}>
            {msg.role === 'user' && <span className="text-[#A0A0A0] opacity-50 mb-1">STRATEGIST@PITWALL:~$</span>}
            <div className="whitespace-pre-wrap">{msg.content}</div>
          </div>
        ))}
      </div>

      <div className="p-3 border-t border-[#1a1f35] bg-[#0a0d1e]">
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <span className="text-[#00E5FF] font-mono text-xs animate-pulse">{">"}</span>
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Execute query (e.g., 'Compare Norris vs Verstappen')..."
            className="flex-1 bg-transparent border-none outline-none text-white text-xs font-mono placeholder:text-[#A0A0A0]/50"
          />
          <button type="submit" className="text-[#A0A0A0] hover:text-[#00E5FF] transition-colors">
            <Send className="w-3 h-3" />
          </button>
        </form>
      </div>
    </div>
  );
}
