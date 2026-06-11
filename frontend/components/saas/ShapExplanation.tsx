"use client";

import { Brain } from "lucide-react";

export default function ShapExplanation() {
  const factors = [
    { name: "Grid Position", value: "+3.18", type: "positive" },
    { name: "Driver Form (Last 3)", value: "+1.39", type: "positive" },
    { name: "Team Perf Index", value: "+1.01", type: "positive" },
    { name: "Wet Weather Risk", value: "-0.85", type: "negative" },
    { name: "Reliability Concerns", value: "-1.22", type: "negative" },
  ];

  return (
    <section className="px-6 max-w-6xl mx-auto mb-24">
      <div className="flex items-center gap-2 mb-8">
        <Brain className="w-5 h-5 text-white" />
        <h2 className="text-2xl font-semibold text-white tracking-tight">Why The Model Thinks This</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-[#111111] border border-[#222222] rounded-xl p-8 flex flex-col justify-center items-center text-center">
          <div className="text-sm text-[#888888] mb-2 uppercase tracking-wider">Prediction Confidence</div>
          <div className="text-5xl font-mono font-medium text-white">92%</div>
          <div className="text-xs text-[#00D084] mt-2 bg-[#00D084]/10 px-2 py-1 rounded">MATHEMATICALLY SOUND</div>
        </div>

        <div className="md:col-span-2 bg-[#111111] border border-[#222222] rounded-xl p-8">
          <div className="text-sm text-[#888888] mb-6 uppercase tracking-wider">Feature Contributions (SHAP)</div>
          
          <div className="space-y-4">
            {factors.map((f, i) => (
              <div key={i} className="flex justify-between items-center text-sm">
                <span className="text-[#EDEDED]">{f.name}</span>
                <div className="flex items-center gap-3">
                  <span className={`font-mono ${f.type === 'positive' ? 'text-[#00D084]' : 'text-[#E00000]'}`}>{f.value}</span>
                  <div className="w-32 h-1.5 bg-[#222222] rounded-full overflow-hidden flex justify-end">
                    {f.type === 'positive' ? (
                      <div className="h-full bg-[#00D084] w-full" />
                    ) : (
                      <div className="h-full bg-[#E00000] w-[40%]" />
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
