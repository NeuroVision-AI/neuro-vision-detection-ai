"use client";

import React, { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Classifier } from "@/components/Classifier";
import { RagChat } from "@/components/RagChat";
import { ModelMetrics } from "@/components/ModelMetrics";
import { ArchitectureDiagram } from "@/components/ArchitectureDiagram";
import { checkApiHealth } from "@/lib/api";
import {
  Brain, MessageSquare, Activity, Sparkles,
  Cpu, GitBranch, Database, Layers, LineChart,
  Network, ChevronDown
} from "lucide-react";

const STATS = [
  { label: "Model Results", value: "Not run", icon: <Activity className="w-4 h-4" />, color: "text-blue-400" },
  { label: "Tumor Classes", value: "4", icon: <Layers className="w-4 h-4" />, color: "text-purple-400" },
  { label: "LLM Providers", value: "4", icon: <Cpu className="w-4 h-4" />, color: "text-emerald-400" },
  { label: "Vector Store", value: "ChromaDB", icon: <Database className="w-4 h-4" />, color: "text-amber-400" },
];

export default function Home() {
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [activeTab, setActiveTab] = useState("classifier");

  useEffect(() => {
    checkApiHealth().then(setApiOnline);
  }, []);

  return (
    <div className="min-h-screen animated-bg flex flex-col">
      {/* ── Header ── */}
      <header className="sticky top-0 z-50 glass border-b border-zinc-800/60">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center gap-4">
          {/* Logo */}
          <div className="flex items-center gap-3 shrink-0">
            <div className="relative">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-violet-600 flex items-center justify-center shadow-lg shadow-blue-900/40">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-zinc-900 pulse-glow" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-base font-black gradient-text leading-none">AI NeuroOnco</h1>
              <p className="text-[10px] text-zinc-600 font-medium leading-none mt-0.5">Research-only MRI AI Platform</p>
            </div>
          </div>

          {/* Stats Bar */}
          <div className="hidden md:flex items-center gap-1 flex-1 justify-center">
            {STATS.map((s) => (
              <div key={s.label} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800/40 border border-zinc-700/40">
                <span className={s.color}>{s.icon}</span>
                <span className="text-xs font-bold text-zinc-300">{s.value}</span>
                <span className="text-xs text-zinc-600">{s.label}</span>
              </div>
            ))}
          </div>

          {/* API Status */}
          <div className="ml-auto flex items-center gap-2">
            <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-semibold ${
              apiOnline === true
                ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                : apiOnline === false
                ? "bg-red-500/10 border-red-500/30 text-red-400"
                : "bg-zinc-800 border-zinc-700 text-zinc-500"
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${
                apiOnline === true ? "bg-emerald-400 pulse-glow" : apiOnline === false ? "bg-red-400" : "bg-zinc-500 animate-pulse"
              }`} />
              {apiOnline === true ? "API Online" : apiOnline === false ? "API Offline" : "Checking..."}
            </div>

            <Badge className="hidden sm:flex bg-violet-500/20 text-violet-300 border-violet-500/30 text-[10px] font-bold">
              <Sparkles className="w-3 h-3 mr-1" />
              MedGemma
            </Badge>
            <Badge className="hidden sm:flex bg-zinc-800 text-zinc-400 border-zinc-700 text-[10px] font-bold">
              <GitBranch className="w-3 h-3 mr-1" />
              Track 3
            </Badge>
          </div>
        </div>
      </header>

      {/* ── Hero Section ── */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-900/10 via-transparent to-transparent pointer-events-none" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-10 text-center relative">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-300 text-xs font-semibold mb-5">
            <Sparkles className="w-3.5 h-3.5" />
            Powered by MedGemma · EfficientNet · LangChain · ChromaDB
          </div>
          <h2 className="text-4xl sm:text-5xl font-black text-zinc-100 leading-tight mb-3">
            Neuro-Oncology{" "}
            <span className="gradient-text">Intelligence</span>{" "}
            Platform
          </h2>
          <p className="text-zinc-400 text-lg max-w-2xl mx-auto leading-relaxed">
            Evaluate a public-data 2D MRI classification proof-of-concept and query the supporting
            literature. Outputs are non-diagnostic and require completed model validation.
          </p>
        </div>
      </div>

      {/* ── Architecture Diagram (Collapsible) ── */}
      <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 mb-6">
        <details className="group border border-zinc-800/80 rounded-2xl bg-zinc-900/20 overflow-hidden shadow-lg transition-all duration-300">
          <summary className="list-none flex items-center justify-between p-4 cursor-pointer hover:bg-zinc-800/40 select-none">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-blue-500/20 flex items-center justify-center">
                <Network className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-zinc-200">🔍 View System Architecture & Pipelines Schematic</h3>
                <p className="text-[11px] text-zinc-500 mt-0.5">Explore the live animated requests flow mapping frontend components, backend routers, PyTorch services, and ChromaDB vector store.</p>
              </div>
            </div>
            <div className="text-zinc-500 group-open:rotate-180 transition-transform duration-300 mr-2">
              <ChevronDown className="w-4 h-4" />
            </div>
          </summary>
          <div className="p-4 border-t border-zinc-800/80 bg-zinc-950/40">
            <ArchitectureDiagram />
          </div>
        </details>
      </div>

      {/* ── Main Tabs ── */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 pb-10">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full max-w-xl mx-auto grid-cols-3 mb-8 h-12 bg-zinc-900/80 border border-zinc-800 p-1 rounded-xl shadow-xl">
            <TabsTrigger
              value="classifier"
              id="tab-classifier"
              className="rounded-lg h-full data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-lg text-zinc-400 hover:text-zinc-200 transition-all font-semibold text-sm gap-2"
            >
              <Activity className="w-4 h-4" />
              MRI Classifier
            </TabsTrigger>
            <TabsTrigger
              value="rag"
              id="tab-rag"
              className="rounded-lg h-full data-[state=active]:bg-gradient-to-r data-[state=active]:from-violet-600 data-[state=active]:to-purple-600 data-[state=active]:text-white data-[state=active]:shadow-lg text-zinc-400 hover:text-zinc-200 transition-all font-semibold text-sm gap-2"
            >
              <MessageSquare className="w-4 h-4" />
              Medical RAG
            </TabsTrigger>
            <TabsTrigger
              value="metrics"
              id="tab-metrics"
              className="rounded-lg h-full data-[state=active]:bg-gradient-to-r data-[state=active]:from-amber-600 data-[state=active]:to-orange-600 data-[state=active]:text-white data-[state=active]:shadow-lg text-zinc-400 hover:text-zinc-200 transition-all font-semibold text-sm gap-2"
            >
              <LineChart className="w-4 h-4" />
              Model Diagnostics
            </TabsTrigger>
          </TabsList>

          <TabsContent value="classifier">
            <Classifier />
          </TabsContent>
          <TabsContent value="rag">
            <RagChat />
          </TabsContent>
          <TabsContent value="metrics">
            <ModelMetrics />
          </TabsContent>
        </Tabs>
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-zinc-800/60 glass py-4 px-6">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-zinc-600">
          <span>AI NeuroOnco — Track 3 · Research Project</span>
          <span className="flex items-center gap-3">
            <span>FastAPI · Next.js · LangChain · PyTorch · ChromaDB</span>
          </span>
        </div>
      </footer>
    </div>
  );
}
