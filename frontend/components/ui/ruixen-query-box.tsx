"use client";

import { Mic, SendHorizonal, Upload } from "lucide-react";
import { useState, useRef } from "react";
import { cn } from "@/lib/utils";
import { Textarea } from "@/components/ui/textarea";
import { useAutoResizeTextarea } from "@/hooks/use-auto-resize-textarea";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";

export default function RuixenQueryBox({ onSubmit }: { onSubmit: (query: string) => void }) {
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 56,
    maxHeight: 220,
  });

  const [inputValue, setInputValue] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleSend = () => {
    if (!inputValue.trim()) return;
    // Call the parent's submit handler instead of just console.log
    onSubmit(inputValue);
    setInputValue("");
    adjustHeight(true);
  };

  const handleFileUpload = (files: FileList | null) => {
    if (!files) return;
    console.log("Uploaded files:", files);
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
            "px-5 py-4 pr-32 rounded-2xl leading-[1.4]",
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
            className="p-2 rounded-full bg-black/20 hover:bg-black/40 text-gray-300 transition-colors border border-white/5"
            onClick={() => alert("Voice input is currently disabled. Coming soon!")}
          >
            <Mic className="w-4 h-4" />
          </button>

          {/* File Upload Popover */}
          <Popover>
            <PopoverTrigger asChild>
              <button
                type="button"
                className="p-2 rounded-full bg-black/20 hover:bg-black/40 text-gray-300 transition-colors border border-white/5"
              >
                <Upload className="w-4 h-4" />
              </button>
            </PopoverTrigger>
            <PopoverContent className="w-60 p-4 border border-white/10" style={{background: 'var(--bg-primary)'}}>
              <p className="text-sm mb-2 text-white">Upload F1 Data:</p>
              <input
                type="file"
                multiple
                ref={fileInputRef}
                onChange={(e) => handleFileUpload(e.target.files)}
                className="w-full border border-gray-600 rounded p-1 text-xs text-white"
              />
              <Button
                className="mt-2 w-full bg-cyan-600 hover:bg-cyan-500 text-white"
                onClick={() => fileInputRef.current?.click()}
              >
                Choose Files
              </Button>
            </PopoverContent>
          </Popover>

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
