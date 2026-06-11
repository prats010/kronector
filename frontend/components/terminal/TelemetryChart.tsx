"use client";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function TelemetryChart() {
  const data = Array.from({ length: 50 }).map((_, i) => ({
    lap: i + 1,
    ver: 80 + Math.random() * 2 + (i * 0.05), // Verstappen lap times (slowly degrading)
    nor: 80.5 + Math.random() * 1.5 + (i * 0.04), // Norris lap times (degrading slower)
  }));

  return (
    <div className="bg-[#0a0d1e] border border-[#1a1f35] rounded-lg p-5 mb-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">Telemetry Center</h2>
          <p className="text-xs text-[#A0A0A0] mt-1">Race Pace & Tire Degradation (Simulated)</p>
        </div>
      </div>

      <div className="h-[250px] w-full text-xs">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1f35" vertical={false} />
            <XAxis dataKey="lap" stroke="#A0A0A0" tick={{ fill: '#A0A0A0', fontSize: 10 }} />
            <YAxis domain={['auto', 'auto']} stroke="#A0A0A0" tick={{ fill: '#A0A0A0', fontSize: 10 }} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#050816', border: '1px solid #1a1f35', color: '#fff' }}
              itemStyle={{ fontSize: '12px', fontFamily: 'monospace' }}
            />
            <Line type="monotone" dataKey="ver" name="Verstappen (s)" stroke="#00E5FF" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="nor" name="Norris (s)" stroke="#E10600" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
