"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function ShapPanel() {
  const data = [
    { name: 'Grid Position', value: 3.18 },
    { name: 'Driver Form', value: 1.39 },
    { name: 'Team Perf', value: 1.01 },
    { name: 'Wet Risk', value: -0.85 },
    { name: 'Reliability', value: -1.22 },
  ];

  return (
    <div className="bg-[#0a0d1e] border border-[#1a1f35] rounded-lg p-5 mb-6">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-sm font-bold text-white uppercase tracking-wider">SHAP Explainability</h2>
          <p className="text-xs text-[#A0A0A0] mt-1">LightGBM Feature Contributions</p>
        </div>
      </div>

      <div className="h-[250px] w-full text-xs">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 0, left: 20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1f35" horizontal={false} />
            <XAxis type="number" stroke="#A0A0A0" tick={{ fill: '#A0A0A0', fontSize: 10 }} />
            <YAxis dataKey="name" type="category" stroke="#A0A0A0" tick={{ fill: '#A0A0A0', fontSize: 10 }} width={80} />
            <Tooltip 
              cursor={{fill: '#1a1f35', opacity: 0.4}}
              contentStyle={{ backgroundColor: '#050816', border: '1px solid #1a1f35', color: '#fff' }}
              itemStyle={{ fontSize: '12px', fontFamily: 'monospace' }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#00E5FF' : '#E10600'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
