"use client";

import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Info, HelpCircle } from "lucide-react";

interface NodeInfo {
  title: string;
  subtitle: string;
  description: string;
  color: string;
  glow: string;
}

const NODES_DATA: Record<string, NodeInfo> = {
  classifier: {
    title: "MRI Classifier",
    subtitle: "Next.js Frontend component",
    description: "Accepts MRI scans (.jpg, .png, .dcm) and sends upload payloads to FastAPI. Renders predictions and Grad-CAM base64 heatmaps.",
    color: "border-blue-500/40 bg-blue-950/20 text-blue-300",
    glow: "shadow-blue-500/20",
  },
  rag_chat: {
    title: "Medical RAG Chat",
    subtitle: "Next.js Frontend component",
    description: "Allows conversational queries grounded in medical literature. Sends base64 attached scans and prompts to LangChain, displaying citations with hover snippets.",
    color: "border-violet-500/40 bg-violet-950/20 text-violet-300",
    glow: "shadow-violet-500/20",
  },
  diagnostics: {
    title: "Model Diagnostics",
    subtitle: "Next.js Frontend component",
    description: "Queries backend metrics, fetching base64 training accuracy/loss curves, ROC graphs, and confusion matrix plots to render statistical diagnostics.",
    color: "border-amber-500/40 bg-amber-950/20 text-amber-300",
    glow: "shadow-amber-500/20",
  },
  gateway: {
    title: "FastAPI Gateway",
    subtitle: "FastAPI Microservice (api/main.py)",
    description: "Handles HTTP routes, validates input models, CORS configuration, file buffering, and coordinates task scheduling to the ML and RAG services.",
    color: "border-zinc-500/40 bg-zinc-950/30 text-zinc-300",
    glow: "shadow-zinc-500/20",
  },
  model_service: {
    title: "Model Service",
    subtitle: "PyTorch Framework (model_service.py)",
    description: "Factory loading of EfficientNet/Custom CNN checkpoints. Performs image decoding (including VOI LUT windowing on raw DICOM) and forward passes.",
    color: "border-emerald-500/40 bg-emerald-950/20 text-emerald-300",
    glow: "shadow-emerald-500/20",
  },
  rag_service: {
    title: "RAG Service",
    subtitle: "LangChain Pipeline (rag_service.py)",
    description: "Manages document parsing (PDF, TXT, multi-sheet Excel), RecursiveTextSplitting, embedding creation, MMR vector searches, and LLM orchestration.",
    color: "border-purple-500/40 bg-purple-950/20 text-purple-300",
    glow: "shadow-purple-500/20",
  },
  checkpoints: {
    title: "Weights (.pth)",
    subtitle: "PyTorch Model Checkpoints",
    description: "Pretrained weights (best_accuracy.pth, best_loss.pth) stored inside outputs/models/ for model factory loading.",
    color: "border-teal-500/40 bg-teal-950/20 text-teal-300",
    glow: "shadow-teal-500/10",
  },
  reports: {
    title: "PDF Reports",
    subtitle: "ReportLab Library",
    description: "Compiles clinical prediction summaries, probability bars, patient details, and Grad-CAM overlays into report PDFs on-the-fly and streams them to client.",
    color: "border-pink-500/40 bg-pink-950/20 text-pink-300",
    glow: "shadow-pink-500/10",
  },
  chromadb: {
    title: "ChromaDB",
    subtitle: "Local Vector Store database",
    description: "Indexes document embeddings (all-MiniLM-L6-v2) in persistent sqlite cache. Returns MMR document chunks matching user clinical search queries.",
    color: "border-rose-500/40 bg-rose-950/20 text-rose-300",
    glow: "shadow-rose-500/10",
  },
  medgemma: {
    title: "MedGemma / Cloud VLM",
    subtitle: "Ollama / Gemini / GPT API",
    description: "Invokes selected VLM (local MedGemma / Llama or Google Gemini / OpenAI) using multimodal messages (combining RAG context and visual scan uploads).",
    color: "border-sky-500/40 bg-sky-950/20 text-sky-300",
    glow: "shadow-sky-500/10",
  },
};

export function ArchitectureDiagram() {
  const [activeNode, setActiveNode] = useState<string | null>(null);

  const activeInfo = activeNode ? NODES_DATA[activeNode] : null;

  return (
    <Card className="bg-zinc-950/60 border-zinc-900 rounded-2xl shadow-2xl overflow-hidden glass w-full max-w-5xl mx-auto">
      <CardContent className="p-6 space-y-5">
        <div className="flex justify-between items-center pb-2 border-b border-zinc-900">
          <div>
            <h4 className="text-sm font-bold text-zinc-200 flex items-center gap-1.5">
              <Info className="w-4 h-4 text-violet-400" /> Interactive Platform Pipelines
            </h4>
            <p className="text-xs text-zinc-500 mt-0.5">
              Hover over or click system components to inspect data flow connections and detailed operations.
            </p>
          </div>
          <div className="flex items-center gap-3 text-[10px] text-zinc-600 font-semibold uppercase tracking-wider">
            <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-blue-500 pulse-glow" /> Client Flow</span>
            <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-purple-500 pulse-glow" /> RAG / VLM Flow</span>
            <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500 pulse-glow" /> Model / CV Flow</span>
          </div>
        </div>

        {/* ── Interactive SVG Diagram ── */}
        <div className="w-full overflow-x-auto py-2 bg-black/25 rounded-xl border border-zinc-900/60 flex justify-center">
          <svg
            width="860"
            height="360"
            viewBox="0 0 860 360"
            className="select-none shrink-0"
          >
            {/* ── SVG Filters for glowing nodes ── */}
            <defs>
              <filter id="glow-blue" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="4" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            {/* ── CONNECTION FLOW LINES ── */}
            {/* Client -> FastAPI Gateway */}
            <path d="M 160 65 C 210 65, 210 180, 260 180" className="animate-flow-blue stroke-blue-500/60 fill-none stroke-[2]" />
            <path d="M 160 180 L 260 180" className="animate-flow-purple stroke-purple-500/60 fill-none stroke-[2]" />
            <path d="M 160 295 C 210 295, 210 180, 260 180" className="animate-flow-amber stroke-amber-500/60 fill-none stroke-[2]" />

            {/* FastAPI -> Services */}
            <path d="M 410 180 C 445 180, 445 85, 480 85" className="animate-flow-blue stroke-blue-500/60 fill-none stroke-[2]" />
            <path d="M 410 180 C 445 180, 445 265, 480 265" className="animate-flow-purple stroke-purple-500/60 fill-none stroke-[2]" />

            {/* ML Service -> Storage/Checkpoints */}
            <path d="M 630 85 C 665 85, 665 45, 700 45" className="animate-flow-emerald stroke-emerald-500/50 fill-none stroke-[1.5]" />
            <path d="M 630 85 C 665 85, 665 125, 700 125" className="animate-flow-emerald stroke-emerald-500/50 fill-none stroke-[1.5]" />

            {/* RAG Service -> DB/Ollama */}
            <path d="M 630 265 C 665 265, 665 225, 700 225" className="animate-flow-purple stroke-purple-500/50 fill-none stroke-[1.5]" />
            <path d="M 630 265 C 665 265, 665 315, 700 315" className="animate-flow-purple stroke-purple-500/50 fill-none stroke-[1.5]" />

            {/* ── NODES ── */}

            {/* Next.js Client Stack */}
            <g
              transform="translate(10, 25)"
              className="cursor-pointer"
              onMouseEnter={() => setActiveNode("classifier")}
              onMouseLeave={() => setActiveNode(null)}
            >
              <rect x="0" y="0" width="150" height="80" rx="12" className="fill-zinc-950/80 stroke-blue-500/30 stroke-[1.5] transition-all hover:stroke-blue-500" />
              <text x="15" y="32" className="fill-zinc-100 font-bold text-xs">MRI Classifier</text>
              <text x="15" y="50" className="fill-blue-400 font-medium text-[9px] uppercase tracking-wider font-mono">Classifier.tsx</text>
              <text x="15" y="65" className="fill-zinc-500 text-[9px]">MRI & DICOM ML Scan</text>
              <circle cx="140" cy="40" r="4" className="fill-blue-400 animate-pulse" />
            </g>

            <g
              transform="translate(10, 140)"
              className="cursor-pointer"
              onMouseEnter={() => setActiveNode("rag_chat")}
              onMouseLeave={() => setActiveNode(null)}
            >
              <rect x="0" y="0" width="150" height="80" rx="12" className="fill-zinc-950/80 stroke-violet-500/30 stroke-[1.5] transition-all hover:stroke-violet-500" />
              <text x="15" y="32" className="fill-zinc-100 font-bold text-xs">Medical RAG Chat</text>
              <text x="15" y="50" className="fill-violet-400 font-medium text-[9px] uppercase tracking-wider font-mono">RagChat.tsx</text>
              <text x="15" y="65" className="fill-zinc-500 text-[9px]">Multimodal Chat & Citations</text>
              <circle cx="140" cy="40" r="4" className="fill-violet-400 animate-pulse" />
            </g>

            <g
              transform="translate(10, 255)"
              className="cursor-pointer"
              onMouseEnter={() => setActiveNode("diagnostics")}
              onMouseLeave={() => setActiveNode(null)}
            >
              <rect x="0" y="0" width="150" height="80" rx="12" className="fill-zinc-950/80 stroke-amber-500/30 stroke-[1.5] transition-all hover:stroke-amber-500" />
              <text x="15" y="32" className="fill-zinc-100 font-bold text-xs">Model Diagnostics</text>
              <text x="15" y="50" className="fill-amber-400 font-medium text-[9px] uppercase tracking-wider font-mono">ModelMetrics.tsx</text>
              <text x="15" y="65" className="fill-zinc-500 text-[9px]">Confusion Matrix & curves</text>
              <circle cx="140" cy="40" r="4" className="fill-amber-400 animate-pulse" />
            </g>

            {/* FastAPI API Gateway Router */}
            <g
              transform="translate(260, 140)"
              className="cursor-pointer"
              onMouseEnter={() => setActiveNode("gateway")}
              onMouseLeave={() => setActiveNode(null)}
            >
              <rect x="0" y="0" width="150" height="80" rx="12" className="fill-zinc-950/90 stroke-zinc-700 stroke-[2] transition-all hover:stroke-zinc-400" />
              <text x="15" y="32" className="fill-zinc-100 font-bold text-xs">FastAPI Gateway</text>
              <text x="15" y="50" className="fill-zinc-400 font-medium text-[9px] uppercase tracking-wider font-mono">api/main.py</text>
              <text x="15" y="65" className="fill-zinc-500 text-[9px]">Uvicorn Router Port 8000</text>
              <circle cx="140" cy="40" r="4.5" className="fill-emerald-400 animate-pulse" />
            </g>

            {/* ML Core Processor */}
            <g
              transform="translate(480, 45)"
              className="cursor-pointer"
              onMouseEnter={() => setActiveNode("model_service")}
              onMouseLeave={() => setActiveNode(null)}
            >
              <rect x="0" y="0" width="150" height="80" rx="12" className="fill-zinc-950/80 stroke-emerald-500/30 stroke-[1.5] transition-all hover:stroke-emerald-500" />
              <text x="15" y="32" className="fill-zinc-100 font-bold text-xs">Model Service</text>
              <text x="15" y="50" className="fill-emerald-400 font-medium text-[9px] uppercase tracking-wider font-mono">PyTorch + Grad-CAM</text>
              <text x="15" y="65" className="fill-zinc-500 text-[9px]">DICOM normalization</text>
              <circle cx="140" cy="40" r="4" className="fill-emerald-400 animate-pulse" />
            </g>

            {/* RAG Core Processor */}
            <g
              transform="translate(480, 225)"
              className="cursor-pointer"
              onMouseEnter={() => setActiveNode("rag_service")}
              onMouseLeave={() => setActiveNode(null)}
            >
              <rect x="0" y="0" width="150" height="80" rx="12" className="fill-zinc-950/80 stroke-purple-500/30 stroke-[1.5] transition-all hover:stroke-purple-500" />
              <text x="15" y="32" className="fill-zinc-100 font-bold text-xs">RAG Service</text>
              <text x="15" y="50" className="fill-purple-400 font-medium text-[9px] uppercase tracking-wider font-mono">LangChain Core</text>
              <text x="15" y="65" className="fill-zinc-500 text-[9px]">ChromaDB search & MMR</text>
              <circle cx="140" cy="40" r="4" className="fill-purple-400 animate-pulse" />
            </g>

            {/* ML Storage Elements */}
            <g
              transform="translate(700, 15)"
              className="cursor-pointer"
              onMouseEnter={() => setActiveNode("checkpoints")}
              onMouseLeave={() => setActiveNode(null)}
            >
              <rect x="0" y="0" width="140" height="60" rx="8" className="fill-zinc-950/80 stroke-teal-500/20 stroke-[1] transition-all hover:stroke-teal-500" />
              <text x="12" y="24" className="fill-zinc-200 font-bold text-[10px]">Model Weights</text>
              <text x="12" y="42" className="fill-teal-400 text-[8px] font-mono">best_accuracy.pth</text>
            </g>

            <g
              transform="translate(700, 95)"
              className="cursor-pointer"
              onMouseEnter={() => setActiveNode("reports")}
              onMouseLeave={() => setActiveNode(null)}
            >
              <rect x="0" y="0" width="140" height="60" rx="8" className="fill-zinc-950/80 stroke-pink-500/20 stroke-[1] transition-all hover:stroke-pink-500" />
              <text x="12" y="24" className="fill-zinc-200 font-bold text-[10px]">PDF Reports</text>
              <text x="12" y="42" className="fill-pink-400 text-[8px] font-mono">ReportLab Builder</text>
            </g>

            {/* RAG Database Elements */}
            <g
              transform="translate(700, 195)"
              className="cursor-pointer"
              onMouseEnter={() => setActiveNode("chromadb")}
              onMouseLeave={() => setActiveNode(null)}
            >
              <rect x="0" y="0" width="140" height="60" rx="8" className="fill-zinc-950/80 stroke-rose-500/20 stroke-[1] transition-all hover:stroke-rose-500" />
              <text x="12" y="24" className="fill-zinc-200 font-bold text-[10px]">ChromaDB</text>
              <text x="12" y="42" className="fill-rose-400 text-[8px] font-mono">sqlite vector persist</text>
            </g>

            <g
              transform="translate(700, 285)"
              className="cursor-pointer"
              onMouseEnter={() => setActiveNode("medgemma")}
              onMouseLeave={() => setActiveNode(null)}
            >
              <rect x="0" y="0" width="140" height="60" rx="8" className="fill-zinc-950/80 stroke-sky-500/20 stroke-[1] transition-all hover:stroke-sky-500" />
              <text x="12" y="24" className="fill-zinc-200 font-bold text-[10px]">Ollama / cloud VLM</text>
              <text x="12" y="42" className="fill-sky-400 text-[8px] font-mono">medgemma:latest / Gemini</text>
            </g>
          </svg>
        </div>

        {/* ── Dynamic Description Panel ── */}
        <div className="min-h-[75px] bg-zinc-900/30 border border-zinc-800/80 p-3.5 rounded-xl flex gap-3 items-start transition-all">
          <div className="w-8 h-8 rounded-lg bg-violet-500/10 flex items-center justify-center shrink-0 mt-0.5">
            <HelpCircle className="w-4.5 h-4.5 text-violet-400 animate-pulse" />
          </div>
          <div className="space-y-1">
            {activeInfo ? (
              <>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-zinc-100">{activeInfo.title}</span>
                  <span className="text-[9px] text-zinc-500 font-medium font-mono">{activeInfo.subtitle}</span>
                </div>
                <p className="text-xs text-zinc-400 leading-relaxed font-medium">{activeInfo.description}</p>
              </>
            ) : (
              <>
                <span className="text-xs font-bold text-zinc-100">System Architecture Viewer</span>
                <p className="text-xs text-zinc-500 leading-relaxed font-medium">
                  Hover over any node in the pipeline schematic diagram above to drill down into its localized responsibilities, underlying libraries, and data transport details.
                </p>
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
