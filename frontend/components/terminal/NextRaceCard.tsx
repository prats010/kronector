"use client";

import { MapPin, CloudRain, ShieldAlert } from "lucide-react";

export default function NextRaceCard() {
  const predictions = [
    { name: "Verstappen", prob: 48, color: "bg-[#00E5FF]" },
    { name: "Norris", prob: 27, color: "bg-[#E10600]" },
    { name: "Leclerc", prob: 14, color: "bg-[#E10600]" },
    { name: "Piastri", prob: 7, color: "bg-[#E10600]" },
    { name: "Russell", prob: 4, color: "bg-[#6D5BFF]" },
  ];

  return (
    <div className="bg-[#0a0d1e] border border-[#1a1f35] rounded-lg p-5 mb-6">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">Next Grand Prix Prediction</h2>
          <div className="flex items-center gap-3 mt-2 text-xs text-[#A0A0A0]">
            <span className="flex items-center gap-1"><MapPin className="w-3 h-3" /> Montreal, Canada</span>
            <span className="flex items-center gap-1"><CloudRain className="w-3 h-3" /> 40% Rain</span>
            <span className="flex items-center gap-1"><ShieldAlert className="w-3 h-3" /> High SC Risk</span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-[#A0A0A0] uppercase">Confidence Level</div>
          <div className="text-sm font-mono text-[#00D084]">HIGH (92%)</div>
        </div>
      </div>

      <div className="space-y-4">
        {predictions.map((p, i) => (
          <div key={i} className="flex items-center gap-4">
            <div className="w-24 text-xs text-white font-mono">{p.name}</div>
            <div className="flex-1 h-2 bg-[#1a1f35] rounded-full overflow-hidden">
              <div className={`h-full ${p.color}`} style={{ width: `${p.prob}%` }} />
            </div>
            <div className="w-12 text-right text-xs font-mono text-[#00E5FF]">{p.prob}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}
