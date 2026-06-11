"use client";

export default function Hero() {
  return (
    <section className="pt-24 pb-12 px-6 max-w-5xl mx-auto text-center">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#111111] border border-[#222222] text-xs font-medium text-[#888888] mb-8">
        <span className="w-2 h-2 rounded-full bg-[#0070F3]"></span>
        LightGBM Model v2.1 Deployed
      </div>
      
      <h1 className="text-5xl md:text-7xl font-bold tracking-tighter text-white mb-6">
        Formula 1 <br className="hidden md:block" />
        <span className="text-[#888888]">Intelligence Platform</span>
      </h1>
      
      <p className="text-lg md:text-xl text-[#888888] max-w-2xl mx-auto leading-relaxed">
        Predict race outcomes, compare drivers, analyze telemetry, and understand every prediction with mathematically-backed explainable AI.
      </p>
    </section>
  );
}
