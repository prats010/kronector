"use client";

import { SendHorizonal } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { Textarea } from "@/components/ui/textarea";
import { useAutoResizeTextarea } from "@/hooks/use-auto-resize-textarea";

export default function RuixenQueryBox({ onSubmit }: { onSubmit: (query: string) => void }) {
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 56,
    maxHeight: 220,
  });

  const [inputValue, setInputValue] = useState("");

  const handleSend = () => {
    if (!inputValue.trim()) return;
    // Call the parent's submit handler instead of just console.log
    onSubmit(inputValue);
    setInputValue("");
    adjustHeight(true);
  };

  return (
    <div className="w-full py-6">
      <div
        className="relative w-full mx-auto bg-white rounded-2xl border shadow-sm overflow-hidden"
        style={{
          backgroundImage:
            "url('https://pub-940ccf6255b54fa799a9b01050e6c227.r2.dev/ruixen_chat_gradient.png')",
          backgroundSize: "cover",
          backgroundPosition: "center",
          borderColor: "var(--border-glass)"
        }}
      >
        <Textarea
          id="ai-textarea"
          ref={textareaRef}
          placeholder="Ask KRONECTOR about F1 stats, predictions, or telemetry..."
          className={cn(
            "w-full resize-none border-none bg-transparent",
            "text-base text-white placeholder:text-gray-400",
            "px-5 py-4 pr-14 rounded-2xl leading-[1.4]",
            "transition-all focus-visible:ring-0 focus-visible:ring-offset-0",
            "min-h-[56px]"
          )}
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
            adjustHeight();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
        />

        {/* Icon Buttons */}
        <div className="absolute bottom-3 right-3 flex items-center gap-2">
          <button
            type="button"
            onClick={handleSend}
            disabled={!inputValue.trim()}
            className={cn(
              "p-2 rounded-full transition-all duration-200 border",
              inputValue.trim()
                ? "bg-cyan-500 hover:bg-cyan-400 text-black border-transparent shadow-[0_0_15px_rgba(0,240,255,0.4)]"
                : "bg-black/20 text-gray-500 border-white/5 cursor-not-allowed"
            )}
          >
            <SendHorizonal className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
