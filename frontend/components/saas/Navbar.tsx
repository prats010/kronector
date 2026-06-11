"use client";

import { Activity } from "lucide-react";
import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 w-full border-b border-[#222222] bg-[#000000]/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-6 h-6 bg-white text-black flex items-center justify-center rounded-sm font-bold text-sm">
              K
            </div>
            <span className="font-semibold tracking-tight text-white">Kronector</span>
          </Link>
          
          <div className="hidden md:flex items-center gap-6 text-sm font-medium text-[#888888]">
            <Link href="#dashboard" className="hover:text-white transition-colors">Dashboard</Link>
            <Link href="#predictions" className="text-white">Predictions</Link>
            <Link href="#compare" className="hover:text-white transition-colors">Compare</Link>
            <Link href="#telemetry" className="hover:text-white transition-colors">Telemetry</Link>
            <Link href="#metrics" className="hover:text-white transition-colors">Metrics</Link>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-[#888888] bg-[#111111] px-3 py-1.5 rounded-full border border-[#222222]">
            <span>Season 2026</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-[#00D084] font-medium">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#00D084] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#00D084]"></span>
            </span>
            Live
          </div>
        </div>
      </div>
    </nav>
  );
}
