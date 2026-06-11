"use client";

import { Activity } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Telemetry() {
  const data = Array.from({ length: 50 }).map((_, i) => ({
    lap: i + 1,
    ver: 80 + Math.random() * 2 + (i * 0.05), // Verstappen lap times
    nor: 80.5 + Math.random() * 1.5 + (i * 0.04), // Norris lap times
  }));

  return (
    <section className="px-6 max-w-6xl mx-auto mb-24">
      <div className="flex items-center gap-2 mb-8">
        <Activity className="w-5 h-5 text-white" />
        <h2 className="text-2xl font-semibold text-white tracking-tight">Telemetry Analysis</h2>
      </div>

      <div className="bg-[#111111] border border-[#222222] rounded-xl p-8">
        <div className="flex items-center justify-between mb-8">
          <div className="text-sm text-[#888888]">Simulated Race Pace & Tire Degradation</div>
          <div className="flex gap-4 text-sm font-medium">
            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#0070F3]" /> Verstappen</div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#FF8700]" /> Norris</div>
          </div>
        </div>

        <div className="h-[300px] w-full text-sm">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#222222" vertical={false} />
              <XAxis dataKey="lap" stroke="#888888" tick={{ fill: '#888888' }} />
              <YAxis domain={['auto', 'auto']} stroke="#888888" tick={{ fill: '#888888' }} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#000000', border: '1px solid #333333', color: '#fff', borderRadius: '8px' }}
              />
              <Line type="monotone" dataKey="ver" name="Verstappen (s)" stroke="#0070F3" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="nor" name="Norris (s)" stroke="#FF8700" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
