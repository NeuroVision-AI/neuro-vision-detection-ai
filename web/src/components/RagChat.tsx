"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Send, Upload, BookOpen, Bot, User, Loader2,
  FileText, Sparkles, Cpu, Cloud, CheckCircle2,
  Database, Wifi, AlertCircle, Trash2, X, Paperclip, ImageIcon
} from "lucide-react";
import { sendChatMessage, uploadDocument, getKBStats, type KBStats, type RAGSource } from "@/lib/api";

// ── Provider Definitions ────────────────────────────────────────────────────────

interface Provider {
  value: string;
  label: string;
  sublabel: string;
  description: string;
  icon: React.ReactNode;
  badge: string;
  badgeColor: string;
  local: boolean;
}

const PROVIDERS: Provider[] = [
  {
    value: "medgemma",
    label: "MedGemma",
    sublabel: "medgemma:latest · Ollama",
    description: "Locally executed research model; still requires normal data-governance controls",
    icon: <Sparkles className="w-4 h-4 text-emerald-400" />,
    badge: "Recommended",
    badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    local: true,
  },
  {
    value: "ollama",
    label: "Llama 3.2",
    sublabel: "llama3.2:latest · Ollama",
    description: "Meta's Llama 3.2 — runs locally via Ollama, no API key needed",
    icon: <Cpu className="w-4 h-4 text-blue-400" />,
    badge: "Local",
    badgeColor: "bg-blue-500/20 text-blue-300 border-blue-500/30",
    local: true,
  },
  {
    value: "gemini",
    label: "Gemini 2.0 Flash",
    sublabel: "gemini-2.0-flash · Google",
    description: "Google's fast multimodal model — requires GOOGLE_API_KEY in api/.env",
    icon: <Cloud className="w-4 h-4 text-purple-400" />,
    badge: "Cloud",
    badgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/30",
    local: false,
  },
  {
    value: "openai",
    label: "GPT-4o",
    sublabel: "gpt-4o · OpenAI",
    description: "OpenAI's flagship model — requires OPENAI_API_KEY in api/.env",
    icon: <Cloud className="w-4 h-4 text-sky-400" />,
    badge: "Cloud",
    badgeColor: "bg-sky-500/20 text-sky-300 border-sky-500/30",
    local: false,
  },
];

// ── Types ────────────────────────────────────────────────────────────────────────

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  provider?: string;
  error?: boolean;
  sources?: RAGSource[];
  image_base64?: string | null;
}

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ── Sub-components ───────────────────────────────────────────────────────────────

function TypingDots() {
  return (
    <div className="flex items-center gap-1.5 px-1 py-0.5">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-2 h-2 rounded-full bg-violet-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s`, animationDuration: "0.8s" }}
        />
      ))}
    </div>
  );
}

function MessageBubble({ msg, onCitationClick }: { msg: Message; onCitationClick?: (index: number) => void }) {
  const isUser = msg.role === "user";
  const isSystem = msg.role === "system";

  if (isSystem) {
    return (
      <div className="flex justify-center my-1">
        <span className="text-xs text-zinc-600 bg-zinc-800/50 px-3 py-1 rounded-full">{msg.content}</span>
      </div>
    );
  }

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`} style={{ maxWidth: "88%" }}>
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 shadow-md ${
        isUser
          ? "bg-gradient-to-br from-blue-600 to-indigo-600"
          : msg.error
          ? "bg-gradient-to-br from-red-600 to-rose-700"
          : "bg-gradient-to-br from-violet-600 to-purple-700"
      }`}>
        {isUser ? <User className="w-4 h-4 text-white" /> : msg.error ? <AlertCircle className="w-4 h-4 text-white" /> : <Sparkles className="w-4 h-4 text-white" />}
      </div>

      {/* Bubble */}
      <div className="flex flex-col gap-1">
        <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-lg whitespace-pre-wrap ${
          isUser
            ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-tr-sm"
            : msg.error
            ? "bg-red-500/10 border border-red-500/30 text-red-300 rounded-tl-sm"
            : "bg-zinc-800 border border-zinc-700 text-zinc-100 rounded-tl-sm"
        }`}>
          {msg.image_base64 && (
            <div className="mb-2 max-w-xs rounded-lg overflow-hidden border border-zinc-700/50 bg-black/20">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={`data:image/jpeg;base64,${msg.image_base64}`} alt="Visual Scan" className="w-full object-contain max-h-40" />
            </div>
          )}
          {msg.content}

          {/* Sources Section */}
          {!isUser && msg.sources && msg.sources.length > 0 && (
            <div className="mt-3 pt-2.5 border-t border-zinc-700/50 space-y-1.5">
              <div className="flex items-center gap-1 text-[10px] text-zinc-500 font-bold uppercase tracking-wider">
                <Database className="w-3 h-3 text-violet-400" />
                <span>Grounded Sources</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {msg.sources.map((src, i) => (
                  <div key={i} className="group relative">
                    <Badge
                      variant="outline"
                      onClick={() => onCitationClick && onCitationClick(i)}
                      className="text-[10px] bg-zinc-900/60 border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 hover:border-emerald-500 cursor-pointer px-2 py-0.5 rounded-md max-w-[200px] truncate transition-colors animate-pulse-glow"
                    >
                      {src.source} {src.page ? `(Page ${src.page})` : ""}
                    </Badge>
                    <div className="absolute bottom-6 left-0 hidden group-hover:block bg-zinc-950 border border-zinc-800 rounded-lg p-2.5 text-[11px] text-zinc-400 w-64 z-20 shadow-2xl leading-normal">
                      <p className="font-bold text-zinc-300 mb-1 text-[10px] truncate">{src.source} {src.page ? `· Page ${src.page}` : ""}</p>
                      <span className="italic">"{src.snippet}"</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className={`flex items-center gap-1.5 px-1 ${isUser ? "flex-row-reverse" : ""}`}>
          <span className="text-[10px] text-zinc-600">{msg.timestamp}</span>
          {msg.provider && (
            <span className="text-[10px] text-zinc-700">· {msg.provider}</span>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────────────

export function RagChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I'm MedGemma, your specialist neuro-oncology AI assistant running locally via Ollama.\n\n" +
        "📚 **Upload documents** (PDF, TXT, XLSX, CSV) to ground my answers in your literature.\n" +
        "💬 **Ask questions** about tumor types, treatment protocols, MRI interpretations, or any neuro-oncology topic.\n\n" +
        "Your data stays private — no information leaves your machine when using local models.",
      timestamp: now(),
    },
  ]);
  const [input, setInput] = useState("");
  const [provider, setProvider] = useState("medgemma");
  const [loading, setLoading] = useState(false);
  const [kbFile, setKbFile] = useState<File | null>(null);
  const [kbLoading, setKbLoading] = useState(false);
  const [kbStats, setKbStats] = useState<KBStats | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const [attachedImage, setAttachedImage] = useState<string | null>(null);

  const [highlightedCitationIndex, setHighlightedCitationIndex] = useState<number | null>(null);

  // Find last assistant message's sources to populate the Grounded Research Library
  const activeCitations = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant" && messages[i].sources && messages[i].sources!.length > 0) {
        return messages[i].sources!;
      }
    }
    return [];
  })();

  const handleImageAttach = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64Str = (event.target?.result as string).split(",")[1];
        setAttachedImage(base64Str);
      };
      reader.readAsDataURL(f);
    }
  };

  const currentProvider = PROVIDERS.find((p) => p.value === provider) ?? PROVIDERS[0];

  // Fetch KB stats
  const refreshStats = useCallback(() => {
    getKBStats().then(setKbStats).catch(() => {});
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // ── Upload ──
  const handleFileUpload = useCallback(async () => {
    if (!kbFile) return;
    setKbLoading(true);
    setUploadSuccess(false);
    try {
      const data = await uploadDocument(kbFile);
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: `Document ingested`,
          timestamp: now(),
        },
        {
          role: "assistant",
          content: `✅ **${kbFile.name}** successfully added to the knowledge base.\n${data.chunks_ingested} text chunks embedded into ChromaDB. You can now ask questions about this document.`,
          timestamp: now(),
        },
      ]);
      setKbFile(null);
      setUploadSuccess(true);
      refreshStats();
      setTimeout(() => setUploadSuccess(false), 3000);
    } catch (e: unknown) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `❌ Upload failed: ${e instanceof Error ? e.message : "Unknown error"}.\nIs the FastAPI backend running on port 8000?`,
          timestamp: now(),
          error: true,
        },
      ]);
    } finally {
      setKbLoading(false);
    }
  }, [kbFile, refreshStats]);

  // ── Send Message ──
  const handleSend = useCallback(async () => {
    if ((!input.trim() && !attachedImage) || loading) return;
    const userMessage = input.trim();
    const currentImg = attachedImage;
    setInput("");
    setAttachedImage(null);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMessage, timestamp: now(), image_base64: currentImg },
    ]);
    setLoading(true);

    try {
      const data = await sendChatMessage(userMessage, provider, currentImg);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || "No response generated.",
          timestamp: now(),
          provider: currentProvider.sublabel,
          sources: data.sources || [],
        },
      ]);
    } catch (e: unknown) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠️ ${e instanceof Error ? e.message : "Could not reach the backend API. Please ensure FastAPI is running on port 8000."}`,
          timestamp: now(),
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, provider, currentProvider, attachedImage]);

  const clearChat = useCallback(() => {
    setMessages([{
      role: "assistant",
      content: "Chat cleared. How can I help you with neuro-oncology today?",
      timestamp: now(),
    }]);
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) setKbFile(f);
  };

  return (
    <div className="w-full max-w-5xl mx-auto flex flex-col gap-4" style={{ height: "88vh", minHeight: "600px" }}>
      {/* ── Control Bar ── */}
      <div className="glass rounded-2xl p-4 flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        {/* Provider Selector */}
        <div className="flex flex-col gap-1 min-w-[230px]">
          <span className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">LLM Provider</span>
          <Select value={provider} onValueChange={(val) => setProvider(val || "")}>
            <SelectTrigger className="bg-zinc-900 border-zinc-700 text-zinc-100 h-10 rounded-xl">
              <div className="flex items-center gap-2">
                {currentProvider.icon}
                <SelectValue />
              </div>
            </SelectTrigger>
            <SelectContent className="bg-zinc-900 border-zinc-700 rounded-xl">
              {PROVIDERS.map((p) => (
                <SelectItem key={p.value} value={p.value} className="text-zinc-100 focus:bg-zinc-800 rounded-lg cursor-pointer py-3">
                  <div className="flex items-center gap-3">
                    {p.icon}
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">{p.label}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full border font-semibold ${p.badgeColor}`}>{p.badge}</span>
                      </div>
                      <span className="text-xs text-zinc-500">{p.sublabel}</span>
                    </div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-[11px] text-zinc-600 max-w-[230px] leading-relaxed">{currentProvider.description}</p>
        </div>

        {/* Knowledge Base Upload */}
        <div className="flex flex-col gap-2 flex-1 max-w-xs">
          <span className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">
            Knowledge Base
            {kbStats && (
              <span className="ml-2 normal-case text-zinc-600">({kbStats.document_chunks} chunks)</span>
            )}
          </span>
          <div
            className={`drop-zone rounded-xl p-3 flex items-center gap-3 cursor-pointer transition-all ${dragOver ? "drag-over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.txt,.md,.csv,.xlsx,.xls"
              onChange={(e) => setKbFile(e.target.files?.[0] || null)}
            />
            <div className="w-8 h-8 rounded-lg bg-violet-500/20 flex items-center justify-center shrink-0">
              <FileText className="w-4 h-4 text-violet-400" />
            </div>
            <div className="flex-1 min-w-0">
              {kbFile ? (
                <div className="flex items-center gap-1">
                  <p className="text-sm text-violet-300 font-medium truncate">{kbFile.name}</p>
                  <button onClick={(e) => { e.stopPropagation(); setKbFile(null); }} className="shrink-0">
                    <X className="w-3 h-3 text-zinc-500 hover:text-zinc-300" />
                  </button>
                </div>
              ) : (
                <p className="text-sm text-zinc-500">Drop PDF / TXT / XLSX here</p>
              )}
            </div>
          </div>
          <Button
            onClick={handleFileUpload}
            disabled={!kbFile || kbLoading}
            size="sm"
            className="bg-violet-600 hover:bg-violet-500 text-white rounded-lg h-8 text-xs font-semibold w-full"
          >
            {kbLoading ? <Loader2 className="w-3 h-3 animate-spin mr-1" /> : uploadSuccess ? <CheckCircle2 className="w-3 h-3 mr-1 text-emerald-400" /> : <Upload className="w-3 h-3 mr-1" />}
            {kbLoading ? "Ingesting..." : uploadSuccess ? "Ingested!" : "Add to Knowledge Base"}
          </Button>
        </div>

        {/* Status + Actions */}
        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-1.5 text-xs text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-glow" />
            <span className="font-medium">Backend Connected</span>
          </div>
          <div className="flex items-center gap-1 text-xs text-zinc-600">
            <Wifi className="w-3 h-3" />
            <span>localhost:8000</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={clearChat}
            className="text-xs text-zinc-500 hover:text-zinc-300 h-7 px-2"
          >
            <Trash2 className="w-3 h-3 mr-1" />
            Clear chat
          </Button>
        </div>
      </div>

      {/* ── Chat & Grounded Library Sidebar Split Grid ── */}
      <div className="flex-1 flex flex-col md:flex-row gap-4 min-h-0 relative">
        {/* Main Chat Window Card */}
        <Card className="flex-[3] flex flex-col overflow-hidden bg-zinc-900/60 border-zinc-800 rounded-2xl shadow-2xl min-h-0">
          <CardHeader className="border-b border-zinc-800 py-3 px-5 shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-violet-500/20 flex items-center justify-center">
                  <BookOpen className="w-4 h-4 text-violet-400" />
                </div>
                <span className="text-base font-bold text-zinc-100">Medical Literature RAG</span>
              </div>
              <div className="flex items-center gap-2">
                {currentProvider.icon}
                <span className="text-xs text-zinc-400">{currentProvider.sublabel}</span>
                <span className={`text-[10px] px-1.5 py-0.5 rounded-full border font-semibold ${currentProvider.badgeColor}`}>
                  {currentProvider.badge}
                </span>
              </div>
            </div>
          </CardHeader>

          <ScrollArea className="flex-1 px-5 py-4 min-h-0">
            <div className="flex flex-col gap-4 max-w-3xl mx-auto pb-2">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === "user" ? "justify-end" : msg.role === "system" ? "justify-center" : "justify-start"}`}
                >
                  <MessageBubble
                    msg={msg}
                    onCitationClick={(citationIdx) => {
                      setHighlightedCitationIndex(citationIdx);
                      const el = document.getElementById(`citation-${citationIdx}`);
                      if (el) {
                        el.scrollIntoView({ behavior: "smooth", block: "nearest" });
                      }
                    }}
                  />
                </div>
              ))}
              {loading && (
                <div className="flex gap-3 justify-start">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-600 to-purple-700 flex items-center justify-center shrink-0 shadow-lg">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                  <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-zinc-800 border border-zinc-700">
                    <TypingDots />
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>
          </ScrollArea>

          <CardFooter className="p-3 border-t border-zinc-800 shrink-0 flex flex-col gap-2">
            {/* Image preview thumbnail */}
            {attachedImage && (
              <div className="relative w-16 h-16 rounded-lg overflow-hidden border border-zinc-700 bg-black/40 self-start ml-12">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`data:image/jpeg;base64,${attachedImage}`} alt="Attached Preview" className="w-full h-full object-cover" />
                <button
                  onClick={() => setAttachedImage(null)}
                  className="absolute top-0.5 right-0.5 w-4 h-4 rounded-full bg-zinc-950/80 hover:bg-zinc-950 flex items-center justify-center text-zinc-400 hover:text-zinc-100"
                >
                  <X className="w-2.5 h-2.5" />
                </button>
              </div>
            )}

            <div className="flex items-center gap-2 w-full max-w-3xl mx-auto">
              {/* Attachment Button */}
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageAttach}
              />
              <Button
                variant="ghost"
                onClick={() => imageInputRef.current?.click()}
                disabled={loading}
                className="h-11 w-11 rounded-xl p-0 border border-zinc-800 hover:bg-zinc-800 hover:text-zinc-100 text-zinc-500 shrink-0"
              >
                <Paperclip className="w-4 h-4" />
              </Button>

              <Input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                placeholder={`Ask ${currentProvider.label} about MRI, tumors, treatment protocols...`}
                className="bg-zinc-800 border-zinc-700 text-zinc-100 placeholder-zinc-600 h-11 rounded-xl focus-visible:ring-violet-500/50 focus-visible:border-violet-500"
                disabled={loading}
              />
              <Button
                onClick={handleSend}
                disabled={(!input.trim() && !attachedImage) || loading}
                className="h-11 w-11 rounded-xl p-0 bg-violet-600 hover:bg-violet-500 text-white shadow-lg shrink-0 disabled:opacity-40 transition-all hover:scale-105 active:scale-95"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              </Button>
            </div>
          </CardFooter>
        </Card>

        {/* Grounded Research Library Sidebar */}
        <Card className="flex-[2] flex flex-col overflow-hidden bg-zinc-950/40 border-zinc-800 rounded-2xl shadow-2xl min-h-0 md:max-w-[320px] w-full">
          <CardHeader className="border-b border-zinc-800 py-3 px-4 shrink-0 bg-zinc-900/20">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-emerald-400" />
              <span className="text-sm font-bold text-zinc-200">Grounded Research Library</span>
            </div>
            <p className="text-[10px] text-zinc-500 mt-0.5 leading-normal">
              Click grounded citations inside conversation bubbles to highlight literature excerpts below.
            </p>
          </CardHeader>
          <ScrollArea className="flex-1 p-4 min-h-0">
            <div className="space-y-3">
              {activeCitations.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-24 text-center text-zinc-700">
                  <BookOpen className="w-8 h-8 opacity-20 mb-2" />
                  <p className="text-xs font-semibold text-zinc-500">No citations referenced yet</p>
                  <p className="text-[10px] text-zinc-600 mt-1 max-w-[200px]">Ask a question to load matching literature references.</p>
                </div>
              ) : (
                activeCitations.map((src, sIdx) => (
                  <div
                    key={sIdx}
                    id={`citation-${sIdx}`}
                    onClick={() => setHighlightedCitationIndex(sIdx)}
                    className={`p-3 rounded-xl border text-xs cursor-pointer transition-all duration-300 ${
                      highlightedCitationIndex === sIdx
                        ? "border-emerald-500/80 bg-emerald-500/5 shadow-lg shadow-emerald-950/20 scale-[1.01]"
                        : "border-zinc-800 bg-zinc-900/10 hover:border-zinc-700"
                    }`}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-bold text-zinc-300 truncate max-w-[170px]" title={src.source}>
                        📄 {src.source}
                      </span>
                      {src.page !== null && (
                        <Badge variant="outline" className="text-[9px] h-4 px-1.5 font-bold border-zinc-800 text-zinc-400 bg-zinc-900/40">
                          Pg. {src.page}
                        </Badge>
                      )}
                    </div>
                    <p className="text-zinc-400 leading-relaxed font-medium bg-black/10 p-2 rounded-lg border border-zinc-900/40 font-mono text-[10px]">
                      &quot;{src.snippet}&quot;
                    </p>
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </Card>
      </div>
    </div>
  );
}
