import Sidebar from "@/components/terminal/Sidebar";
import RightSidebar from "@/components/terminal/RightSidebar";
import SummaryMetrics from "@/components/terminal/SummaryMetrics";
import NextRaceCard from "@/components/terminal/NextRaceCard";
import HeadToHead from "@/components/terminal/HeadToHead";
import TelemetryChart from "@/components/terminal/TelemetryChart";
import ShapPanel from "@/components/terminal/ShapPanel";
import AIStrategist from "@/components/terminal/AIStrategist";

export default function TerminalDashboard() {
  return (
    <div className="flex bg-[#050816] min-h-screen text-white font-sans selection:bg-[#00E5FF]/30">
      <Sidebar />
      
      {/* Main Content Area: Offset by left sidebar (64 = 16rem = 256px) and right sidebar (72 = 18rem = 288px) */}
      <main className="flex-1 ml-64 mr-72 h-screen overflow-y-auto">
        {/* Sticky Header inside main area */}
        <header className="sticky top-0 z-10 bg-[#050816]/90 backdrop-blur-md border-b border-[#1a1f35] p-6 flex justify-between items-center">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">RACE INTELLIGENCE</h1>
            <p className="text-xs font-mono text-[#A0A0A0] mt-1">SEASON 2026 // LIVE TELEMETRY ACQUISITION</p>
          </div>
          <div className="flex gap-4">
            <button className="bg-[#1a1f35] hover:bg-[#1a1f35]/80 text-[#00E5FF] px-4 py-2 rounded text-xs font-mono transition-colors border border-[#00E5FF]/20">
              [ RUN PREDICTION ]
            </button>
            <button className="bg-[#E10600]/10 hover:bg-[#E10600]/20 text-[#E10600] px-4 py-2 rounded text-xs font-mono transition-colors border border-[#E10600]/20">
              [ FORCE RETRAIN ]
            </button>
          </div>
        </header>

        {/* Dense Dashboard Grid */}
        <div className="p-6">
          <SummaryMetrics />
          
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
            <NextRaceCard />
            <HeadToHead />
          </div>

          <div className="grid grid-cols-1 gap-6 mb-6">
            <TelemetryChart />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 pb-12">
            <ShapPanel />
            <AIStrategist />
          </div>
        </div>
      </main>

      <RightSidebar />
    </div>
  );
}
