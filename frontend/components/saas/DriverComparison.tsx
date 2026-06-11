"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function DriverComparison() {
  const data = [
    { metric: 'Qualifying Pace', Norris: 98, Verstappen: 95 },
    { metric: 'Race Pace', Norris: 95, Verstappen: 97 },
    { metric: 'Win Rate', Norris: 45, Verstappen: 52 },
    { metric: 'Avg Finish', Norris: 2.1, Verstappen: 1.8 },
  ];

  return (
    <section id="driver-comparison" className="px-6 max-w-6xl mx-auto mb-24 pt-16">
      <div className="flex items-center gap-2 mb-8">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white"><path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5"/></svg>
        <h2 className="text-2xl font-semibold text-white tracking-tight">Driver Comparison Results</h2>
      </div>

      <div className="bg-[#111111] border border-[#222222] rounded-xl p-8">
        <div className="h-[300px] w-full text-sm">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 0, left: 20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#222222" horizontal={false} />
              <XAxis type="number" stroke="#888888" tick={{ fill: '#888888' }} />
              <YAxis dataKey="metric" type="category" stroke="#888888" tick={{ fill: '#888888' }} width={100} />
              <Tooltip 
                cursor={{fill: '#222222', opacity: 0.4}}
                contentStyle={{ backgroundColor: '#000000', border: '1px solid #333333', color: '#fff', borderRadius: '8px' }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              <Bar dataKey="Norris" fill="#FF8700" radius={[0, 4, 4, 0]} barSize={20} />
              <Bar dataKey="Verstappen" fill="#0070F3" radius={[0, 4, 4, 0]} barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </section>
  );
}
