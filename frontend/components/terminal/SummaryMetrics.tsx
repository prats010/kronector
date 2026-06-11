"use client";

import { Target, Users, Activity, Zap } from "lucide-react";

export default function SummaryMetrics() {
  const metrics = [
    { label: "Race Predictions", value: "4,402", trend: "+12 this week", icon: Target, color: "text-[#00E5FF]" },
    { label: "Model Accuracy", value: "71.8%", trend: "+2.4% vs Baseline", icon: Zap, color: "text-[#00D084]" },
    { label: "Drivers Tracked", value: "142", trend: "Active Grid: 20", icon: Users, color: "text-[#6D5BFF]" },
    { label: "Telemetry Records", value: "8.2M", trend: "FastF1 Connected", icon: Activity, color: "text-[#E10600]" },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {metrics.map((m, i) => (
        <div key={i} className="bg-[#0a0d1e] border border-[#1a1f35] p-5 rounded-lg flex flex-col relative overflow-hidden group">
          <div className="flex justify-between items-start mb-4">
            <span className="text-xs font-medium text-[#A0A0A0] uppercase tracking-wider">{m.label}</span>
            <m.icon className={`w-4 h-4 ${m.color}`} />
          </div>
          <div className="text-2xl font-bold text-white font-mono mb-1">{m.value}</div>
          <div className="text-[10px] text-[#A0A0A0]">{m.trend}</div>
          
          {/* Subtle gradient hover effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-white/0 to-white/[0.02] opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
      ))}
    </div>
  );
}
