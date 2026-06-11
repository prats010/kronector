"use client";

import { CloudRain, MapPin, Trophy } from "lucide-react";

export default function UpcomingGP() {
  const predictions = [
    { name: "Norris", prob: 42, color: "bg-[#FF8700]" },
    { name: "Verstappen", prob: 31, color: "bg-[#0070F3]" },
    { name: "Leclerc", prob: 17, color: "bg-[#E00000]" },
    { name: "Piastri", prob: 10, color: "bg-[#FF8700]" },
  ];

  return (
    <section id="upcoming-gp" className="px-6 max-w-6xl mx-auto mb-16 pt-16">
      <div className="flex items-center gap-2 mb-8">
        <Trophy className="w-5 h-5 text-white" />
        <h2 className="text-2xl font-semibold text-white tracking-tight">Predicted Podium</h2>
      </div>

      <div className="bg-[#111111] border border-[#222222] rounded-xl p-8">
        <div className="flex items-center gap-6 mb-8 text-sm text-[#888888] border-b border-[#222222] pb-6">
          <span className="flex items-center gap-2"><MapPin className="w-4 h-4" /> Montreal, Canada</span>
          <span className="flex items-center gap-2"><CloudRain className="w-4 h-4" /> 40% Rain Probability</span>
        </div>

        <div className="space-y-6">
          {predictions.map((p, i) => (
            <div key={i} className="flex items-center gap-6">
              <div className="w-24 text-sm font-medium text-white">{p.name}</div>
              <div className="flex-1 h-3 bg-[#222222] rounded-full overflow-hidden">
                <div className={`h-full ${p.color} transition-all duration-1000 ease-out`} style={{ width: `${p.prob}%` }} />
              </div>
              <div className="w-12 text-right text-sm font-mono text-[#888888]">{p.prob}%</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
