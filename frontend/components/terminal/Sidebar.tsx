"use client";

import { Activity, BarChart2, CarFront, Cpu, Flag, LayoutDashboard, Settings, Trophy, Zap } from "lucide-react";

export default function Sidebar() {
  const navItems = [
    { name: "Dashboard", icon: LayoutDashboard, active: true },
    { name: "Predictions", icon: Flag, active: false },
    { name: "Drivers", icon: Trophy, active: false },
    { name: "Constructors", icon: CarFront, active: false },
    { name: "Telemetry", icon: Activity, active: false },
    { name: "Analytics", icon: BarChart2, active: false },
    { name: "Model Metrics", icon: Cpu, active: false },
    { name: "Explainability", icon: Zap, active: false },
    { name: "Settings", icon: Settings, active: false },
  ];

  return (
    <aside className="w-64 h-screen border-r border-[#1a1f35] bg-[#030510] flex flex-col fixed left-0 top-0">
      <div className="p-6 border-b border-[#1a1f35]">
        <h1 className="text-xl font-bold tracking-widest text-[#00E5FF] flex items-center gap-2">
          KRONECTOR <span className="text-[10px] bg-[#E10600] text-white px-1.5 py-0.5 rounded font-mono">v2.0</span>
        </h1>
        <p className="text-[10px] text-[#A0A0A0] mt-1 font-mono uppercase">Mission Control</p>
      </div>

      <nav className="flex-1 py-4 flex flex-col gap-1 overflow-y-auto">
        {navItems.map((item) => (
          <button
            key={item.name}
            className={`w-full flex items-center gap-3 px-6 py-2.5 text-sm transition-colors ${
              item.active 
                ? "text-[#00E5FF] bg-[#00E5FF]/10 border-r-2 border-[#00E5FF]" 
                : "text-[#A0A0A0] hover:text-white hover:bg-white/5"
            }`}
          >
            <item.icon className="w-4 h-4" />
            <span className="font-medium">{item.name}</span>
          </button>
        ))}
      </nav>

      <div className="p-4 border-t border-[#1a1f35]">
        <div className="flex items-center gap-3 p-3 rounded bg-white/5 border border-white/5">
          <div className="w-2 h-2 rounded-full bg-[#00D084] animate-pulse" />
          <div className="flex flex-col text-left">
            <span className="text-xs font-mono text-white">SYSTEM ACTIVE</span>
            <span className="text-[10px] text-[#A0A0A0]">Latency: 42ms</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
