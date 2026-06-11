"use client";

import { Database, ShieldCheck, Cpu } from "lucide-react";

export default function Metrics() {
  const metrics = [
    { label: "Model Accuracy", value: "71.8%", icon: Cpu, sub: "Out of sample (2023-2025)" },
    { label: "Data Coverage", value: "8.2M", icon: Database, sub: "Rows of F1 Telemetry" },
    { label: "System Status", value: "Healthy", icon: ShieldCheck, sub: "No data drift detected" },
  ];

  return (
    <section className="px-6 max-w-6xl mx-auto pb-24 border-t border-[#222222] pt-16 mt-16">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {metrics.map((m, i) => (
          <div key={i} className="flex items-start gap-4">
            <div className="p-3 bg-[#111111] border border-[#222222] rounded-lg">
              <m.icon className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="text-sm text-[#888888]">{m.label}</div>
              <div className="text-xl font-semibold text-white mt-1">{m.value}</div>
              <div className="text-xs text-[#666666] mt-1">{m.sub}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
