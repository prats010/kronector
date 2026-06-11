"use client";

import { Activity, AlertTriangle, CheckCircle2, Clock, Server } from "lucide-react";

export default function RightSidebar() {
  return (
    <aside className="w-72 h-screen border-l border-[#1a1f35] bg-[#050816] flex flex-col fixed right-0 top-0">
      <div className="p-4 border-b border-[#1a1f35]">
        <h2 className="text-xs font-bold text-[#A0A0A0] uppercase tracking-wider">Live System Status</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        
        {/* MLflow Status */}
        <section>
          <h3 className="text-[10px] font-mono text-[#00E5FF] mb-3 flex items-center gap-2">
            <Server className="w-3 h-3" /> MLFLOW TRACKING
          </h3>
          <div className="bg-[#0a0d1e] border border-[#1a1f35] rounded p-3 text-xs font-mono space-y-2">
            <div className="flex justify-between">
              <span className="text-[#A0A0A0]">Model ID:</span>
              <span className="text-white">v2.1_lgbm_prod</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#A0A0A0]">Last Trained:</span>
              <span className="text-white">2 hrs ago</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#A0A0A0]">Accuracy:</span>
              <span className="text-[#00D084]">71.8%</span>
            </div>
          </div>
        </section>

        {/* Drift Detection */}
        <section>
          <h3 className="text-[10px] font-mono text-[#00E5FF] mb-3 flex items-center gap-2">
            <Activity className="w-3 h-3" /> DRIFT DETECTION (PSI)
          </h3>
          <div className="space-y-3">
            <div className="bg-[#0a0d1e] border border-[#1a1f35] rounded p-3">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-white">Qualifying Pace</span>
                <span className="text-[#00D084]">0.05</span>
              </div>
              <div className="w-full h-1 bg-[#1a1f35] rounded-full overflow-hidden">
                <div className="h-full bg-[#00D084] w-[15%]" />
              </div>
            </div>

            <div className="bg-[#0a0d1e] border border-[#1a1f35] rounded p-3">
              <div className="flex justify-between text-xs mb-1">
                <span className="text-white">Tire Degradation</span>
                <span className="text-[#E10600]">0.22</span>
              </div>
              <div className="w-full h-1 bg-[#1a1f35] rounded-full overflow-hidden">
                <div className="h-full bg-[#E10600] w-[75%]" />
              </div>
              <p className="text-[10px] text-[#E10600] mt-2 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" /> Warning: Threshold exceeded
              </p>
            </div>
          </div>
        </section>

        {/* Recent Activity */}
        <section>
          <h3 className="text-[10px] font-mono text-[#00E5FF] mb-3 flex items-center gap-2">
            <Clock className="w-3 h-3" /> RECENT QUERIES
          </h3>
          <div className="space-y-2">
            {[
              { q: "Norris Win Prob", t: "2m ago", s: "Success" },
              { q: "Ferrari FP2 Pace", t: "15m ago", s: "Success" },
              { q: "Weather Forecast", t: "1h ago", s: "API Error" },
            ].map((item, i) => (
              <div key={i} className="flex items-center justify-between p-2 rounded bg-[#0a0d1e] border border-[#1a1f35] text-xs">
                <span className="text-white truncate max-w-[120px]">{item.q}</span>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-[#A0A0A0]">{item.t}</span>
                  {item.s === "Success" ? (
                    <CheckCircle2 className="w-3 h-3 text-[#00D084]" />
                  ) : (
                    <AlertTriangle className="w-3 h-3 text-[#E10600]" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

      </div>
    </aside>
  );
}
