"use client";

import { ArrowRight, Search, Zap } from "lucide-react";
import { useState } from "react";

export default function PrimaryActions() {
  const [query, setQuery] = useState("");

  const handlePredict = (e: React.FormEvent) => {
    e.preventDefault();
    if (query) {
      document.getElementById('upcoming-gp')?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleCompare = (e: React.FormEvent) => {
    e.preventDefault();
    document.getElementById('driver-comparison')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="px-6 max-w-6xl mx-auto mb-24">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Predict Card */}
        <div className="bg-[#111111] border border-[#222222] hover:border-[#333333] transition-colors rounded-xl p-8 flex flex-col justify-between h-[320px]">
          <div>
            <div className="flex items-center gap-2 text-white font-medium mb-2">
              <Zap className="w-5 h-5 text-[#0070F3]" />
              Predict a Race Outcome
            </div>
            <p className="text-[#888888] text-sm mb-6">Ask any natural language question about an upcoming race.</p>
          </div>

          <form onSubmit={handlePredict} className="mt-auto">
            <div className="relative mb-4">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-[#888888]" />
              <input 
                type="text" 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Who will win the Canadian GP?"
                className="w-full bg-[#000000] border border-[#222222] rounded-lg py-4 pl-12 pr-4 text-white placeholder:text-[#444444] focus:outline-none focus:border-[#0070F3] focus:ring-1 focus:ring-[#0070F3] transition-all text-lg"
              />
            </div>
            <button type="submit" className="w-full bg-white text-black hover:bg-gray-200 font-medium py-3 rounded-lg flex items-center justify-center gap-2 transition-colors">
              Predict Outcome <ArrowRight className="w-4 h-4" />
            </button>
            <div className="flex gap-3 mt-4 text-xs text-[#666666]">
              <span className="hover:text-[#888888] cursor-pointer">Who wins Monaco?</span>
              <span className="hover:text-[#888888] cursor-pointer">Predict Silverstone podium</span>
            </div>
          </form>
        </div>

        {/* Compare Card */}
        <div className="bg-[#111111] border border-[#222222] hover:border-[#333333] transition-colors rounded-xl p-8 flex flex-col justify-between h-[320px]">
          <div>
            <div className="flex items-center gap-2 text-white font-medium mb-2">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-[#00D084]"><path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5"/></svg>
              Compare Drivers
            </div>
            <p className="text-[#888888] text-sm mb-6">Analyze head-to-head performance across metrics.</p>
          </div>

          <form onSubmit={handleCompare} className="mt-auto">
            <div className="flex items-center gap-4 mb-6">
              <div className="flex-1">
                <select className="w-full bg-[#000000] border border-[#222222] rounded-lg p-4 text-white focus:outline-none focus:border-[#00D084] appearance-none cursor-pointer">
                  <option>Norris</option>
                  <option>Verstappen</option>
                  <option>Leclerc</option>
                </select>
              </div>
              <div className="text-[#888888] font-mono text-sm">VS</div>
              <div className="flex-1">
                <select className="w-full bg-[#000000] border border-[#222222] rounded-lg p-4 text-white focus:outline-none focus:border-[#00D084] appearance-none cursor-pointer">
                  <option>Verstappen</option>
                  <option>Norris</option>
                  <option>Leclerc</option>
                </select>
              </div>
            </div>
            <button type="submit" className="w-full bg-[#222222] text-white hover:bg-[#333333] border border-[#333333] font-medium py-3 rounded-lg flex items-center justify-center gap-2 transition-colors">
              Compare Stats
            </button>
          </form>
        </div>

      </div>
    </section>
  );
}
