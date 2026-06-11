"use client";

import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend, Tooltip } from 'recharts';

export default function HeadToHead() {
  const data = [
    { subject: 'Qualifying Pace', A: 98, B: 95, fullMark: 100 },
    { subject: 'Race Pace', A: 95, B: 97, fullMark: 100 },
    { subject: 'Tire Management', A: 85, B: 92, fullMark: 100 },
    { subject: 'Overtaking', A: 90, B: 88, fullMark: 100 },
    { subject: 'Consistency', A: 92, B: 96, fullMark: 100 },
    { subject: 'Wet Weather', A: 99, B: 85, fullMark: 100 },
  ];

  return (
    <div className="bg-[#0a0d1e] border border-[#1a1f35] rounded-lg p-5 mb-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">Head to Head Analysis</h2>
          <p className="text-xs text-[#A0A0A0] mt-1">Verstappen vs Norris (2025 Performance)</p>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2"><span className="w-3 h-3 bg-[#00E5FF] rounded-sm"/> VER</div>
          <div className="flex items-center gap-2"><span className="w-3 h-3 bg-[#E10600] rounded-sm"/> NOR</div>
        </div>
      </div>

      <div className="h-[300px] w-full text-xs">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
            <PolarGrid stroke="#1a1f35" />
            <PolarAngleAxis dataKey="subject" tick={{ fill: '#A0A0A0', fontSize: 10 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#050816', border: '1px solid #1a1f35', color: '#fff' }}
              itemStyle={{ fontSize: '12px', fontFamily: 'monospace' }}
            />
            <Radar name="Verstappen" dataKey="A" stroke="#00E5FF" fill="#00E5FF" fillOpacity={0.3} />
            <Radar name="Norris" dataKey="B" stroke="#E10600" fill="#E10600" fillOpacity={0.3} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
