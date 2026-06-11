import Navbar from "@/components/saas/Navbar";
import Hero from "@/components/saas/Hero";
import PrimaryActions from "@/components/saas/PrimaryActions";
import UpcomingGP from "@/components/saas/UpcomingGP";
import ShapExplanation from "@/components/saas/ShapExplanation";
import DriverComparison from "@/components/saas/DriverComparison";
import Telemetry from "@/components/saas/Telemetry";
import Metrics from "@/components/saas/Metrics";

export default function SaaSPage() {
  return (
    <div className="bg-[#000000] min-h-screen text-[#EDEDED] font-sans selection:bg-[#0070F3]/30">
      <Navbar />
      
      <main className="flex flex-col items-center w-full">
        <Hero />
        <PrimaryActions />
        
        {/* Everything below the fold */}
        <div className="w-full bg-[#0A0A0A] border-t border-[#222222]">
          <UpcomingGP />
          <ShapExplanation />
        </div>
        
        <div className="w-full">
          <DriverComparison />
          <Telemetry />
          <Metrics />
        </div>
      </main>
    </div>
  );
}
