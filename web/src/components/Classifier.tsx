"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  UploadCloud, AlertTriangle, Brain, FileImage,
  CheckCircle2, Loader2, Zap, ScanLine, Activity,
  ChevronRight, ImageIcon, BarChart3, Info, RefreshCw, FileText,
  Play, Pause, Download
} from "lucide-react";
import { predictMRI, listModels, downloadPDFReport, type ModelInfo, type PredictionResult } from "@/lib/api";

// ── Class display config (matches backend config.CLASS_NAMES exactly) ──────────
const CLASS_CONFIG: Record<string, { label: string; gradient: string; bg: string; bar: string }> = {
  glioma: {
    label: "Glioma",
    gradient: "from-red-500 to-orange-500",
    bg: "bg-red-500/10 border-red-500/30 text-red-300",
    bar: "bg-gradient-to-r from-red-500 to-orange-500",
  },
  meningioma: {
    label: "Meningioma",
    gradient: "from-yellow-500 to-amber-500",
    bg: "bg-yellow-500/10 border-yellow-500/30 text-yellow-300",
    bar: "bg-gradient-to-r from-yellow-500 to-amber-500",
  },
  pituitary: {
    label: "Pituitary",
    gradient: "from-blue-500 to-cyan-500",
    bg: "bg-blue-500/10 border-blue-500/30 text-blue-300",
    bar: "bg-gradient-to-r from-blue-500 to-cyan-500",
  },
  no_tumor: {
    label: "No Tumor",
    gradient: "from-emerald-500 to-teal-500",
    bg: "bg-emerald-500/10 border-emerald-500/30 text-emerald-300",
    bar: "bg-gradient-to-r from-emerald-500 to-teal-500",
  },
  // Aliases for robustness
  notumor: {
    label: "No Tumor",
    gradient: "from-emerald-500 to-teal-500",
    bg: "bg-emerald-500/10 border-emerald-500/30 text-emerald-300",
    bar: "bg-gradient-to-r from-emerald-500 to-teal-500",
  },
};

function getClassConfig(cls: string) {
  return CLASS_CONFIG[cls] ?? {
    label: cls.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    gradient: "from-zinc-500 to-zinc-400",
    bg: "bg-zinc-500/10 border-zinc-500/30 text-zinc-300",
    bar: "bg-zinc-500",
  };
}

export function Classifier() {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [currentSliceIndex, setCurrentSliceIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [model, setModel] = useState("efficientnet");
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [threshold, setThreshold] = useState(0.7);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  // Clinical PDF Report settings states
  const [patientName, setPatientName] = useState("");
  const [patientId, setPatientId] = useState("");
  const [comments, setComments] = useState("");
  const [pdfLoading, setPdfLoading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Computed active slice variables
  const file = files[currentSliceIndex] || null;
  const preview = previews[currentSliceIndex] || null;

  // Autoplay slice scanner slideshow effect
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (isPlaying && files.length > 1) {
      interval = setInterval(() => {
        setCurrentSliceIndex((prev) => (prev + 1) % files.length);
      }, 700);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying, files.length]);

  // Load available models on mount
  useEffect(() => {
    listModels()
      .then(setAvailableModels)
      .catch(() => {
        setAvailableModels([
          { id: "efficientnet", name: "EfficientNet-B0", description: "Transfer-learning baseline; results not yet available.", params: "~5.3M", recommended: true, trained: false },
          { id: "custom_cnn", name: "Custom CNN", description: "Lightweight comparator; results not yet available.", params: "~2M", recommended: false, trained: false },
        ]);
      });
  }, []);

  const handleFiles = useCallback((fList: FileList | File[]) => {
    const list = Array.from(fList);
    const valid = list.filter(f => {
      const isDcm = f.name.toLowerCase().endsWith(".dcm");
      return f.type.startsWith("image/") || isDcm;
    });

    if (valid.length === 0) {
      setError("Please upload valid JPEG, PNG, or DICOM images.");
      return;
    }

    setFiles(valid);
    setCurrentSliceIndex(0);
    setPrediction(null);
    setError(null);
    setIsPlaying(false);

    // Read previews sequentially
    const loadPreviews = async () => {
      const urls: string[] = [];
      for (const f of valid) {
        if (f.name.toLowerCase().endsWith(".dcm")) {
          urls.push("DICOM_PREVIEW_PLACEHOLDER");
        } else {
          const dataUrl = await new Promise<string>((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target?.result as string);
            reader.readAsDataURL(f);
          });
          urls.push(dataUrl);
        }
      }
      setPreviews(urls);
    };

    loadPreviews();
  }, []);

  const handleDownloadReport = useCallback(async () => {
    if (!prediction) return;
    setPdfLoading(true);
    setError(null);
    try {
      const blob = await downloadPDFReport({
        predicted_class: prediction.predicted_class,
        confidence: prediction.confidence,
        predictions: prediction.predictions,
        model_used: model,
        heatmap_base64: prediction.heatmap_base64,
        patient_name: patientName || "Anonymous Patient",
        patient_id: patientId || "N/A",
        comments: comments,
        calibrated: prediction.calibrated ?? false,
        confidence_threshold: threshold,
      });

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `AI_NeuroOnco_Research_Output_${patientId || "Result"}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (e: any) {
      setError(e.message || "Failed to download PDF report.");
    } finally {
      setPdfLoading(false);
    }
  }, [prediction, model, patientName, patientId, comments]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  }, [handleFiles]);

  const handleSubmit = useCallback(async () => {
    const activeFile = files[currentSliceIndex];
    if (!activeFile) return;
    setLoading(true);
    setPrediction(null);
    setError(null);
    try {
      const result = await predictMRI(activeFile, model, threshold);
      setPrediction(result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, [files, currentSliceIndex, model, threshold]);

  const sortedClasses = prediction
    ? Object.entries(prediction.predictions).sort(([, a], [, b]) => b - a)
    : [];

  const predictedCfg = prediction ? getClassConfig(prediction.predicted_class) : null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 w-full max-w-6xl mx-auto">
      {/* ── Left: Upload & Controls ── */}
      <div className="flex flex-col gap-4">
        <Card className="bg-zinc-900/60 border-zinc-800 rounded-2xl shadow-2xl overflow-hidden">
          <CardHeader className="border-b border-zinc-800 pb-4">
            <CardTitle className="flex items-center gap-2.5 text-zinc-100">
              <div className="w-8 h-8 rounded-xl bg-blue-500/20 flex items-center justify-center">
                <ScanLine className="w-4 h-4 text-blue-400" />
              </div>
              MRI Analysis
            </CardTitle>
            <CardDescription className="text-zinc-500">
              Upload a brain MRI to classify tumor type with Grad-CAM explainability.
            </CardDescription>
          </CardHeader>

          <CardContent className="pt-5 space-y-5">
            {/* ── Drop Zone ── */}
            <div
              className={`drop-zone rounded-xl min-h-[180px] flex flex-col items-center justify-center cursor-pointer relative transition-all ${dragOver ? "drag-over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                id="mri-upload"
                type="file"
                accept="image/*,.dcm"
                multiple
                className="hidden"
                onChange={(e) => { if (e.target.files) handleFiles(e.target.files); }}
              />
              {preview ? (
                <div className="relative w-full h-[180px] p-2 flex items-center justify-center">
                  {file?.name.toLowerCase().endsWith(".dcm") ? (
                    <div className="w-full h-full flex flex-col items-center justify-center bg-zinc-950 rounded-lg border border-zinc-800/80">
                      <FileText className="w-10 h-10 text-blue-400 mb-2 animate-pulse" />
                      <span className="text-xs font-semibold text-zinc-300">DICOM Research Image</span>
                      <span className="text-[10px] text-zinc-500 mt-1 truncate max-w-[220px]">{file.name}</span>
                    </div>
                  ) : (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img src={preview} alt="MRI Preview" className="w-full h-full object-contain rounded-lg" />
                  )}
                  <div className="absolute inset-0 bg-gradient-to-t from-zinc-900/60 to-transparent rounded-lg pointer-events-none" />

                  {/* Multi-slice player controls */}
                  {files.length > 1 && (
                    <div
                      className="absolute bottom-11 left-1/2 transform -translate-x-1/2 flex items-center gap-2.5 bg-zinc-950/90 border border-zinc-800 px-3 py-1 rounded-full shadow-2xl z-20"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <button
                        type="button"
                        className="p-1 rounded-full hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 transition-all"
                        onClick={() => setCurrentSliceIndex(prev => (prev - 1 + files.length) % files.length)}
                      >
                        <ChevronRight className="w-3 h-3 rotate-180" />
                      </button>
                      <button
                        type="button"
                        className="p-1 rounded-full hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 transition-all"
                        onClick={() => setIsPlaying(!isPlaying)}
                      >
                        {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                      </button>
                      <span className="text-[9px] font-mono font-bold text-zinc-400 select-none min-w-[50px] text-center">
                        Slice {currentSliceIndex + 1}/{files.length}
                      </span>
                      <button
                        type="button"
                        className="p-1 rounded-full hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 transition-all"
                        onClick={() => setCurrentSliceIndex(prev => (prev + 1) % files.length)}
                      >
                        <ChevronRight className="w-3 h-3" />
                      </button>
                    </div>
                  )}

                  <div className="absolute bottom-2 left-3 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span className="text-xs text-emerald-300 font-medium truncate max-w-[200px]">
                      {files.length > 1 ? `Loaded ${files.length} Scan Slices` : file?.name}
                    </span>
                  </div>
                  <button
                    className="absolute top-2 right-2 p-1.5 rounded-lg bg-zinc-900/70 text-zinc-400 hover:text-zinc-200 transition-colors z-10"
                    onClick={(e) => { e.stopPropagation(); setFiles([]); setPreviews([]); setCurrentSliceIndex(0); setIsPlaying(false); setPrediction(null); }}
                  >
                    <RefreshCw className="w-3 h-3" />
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3 text-zinc-500 py-8 px-4">
                  <div className="w-14 h-14 rounded-2xl bg-zinc-800 flex items-center justify-center">
                    <UploadCloud className="w-7 h-7 text-zinc-500" />
                  </div>
                  <div className="text-center">
                    <p className="font-semibold text-zinc-400">Drop MRI slices here</p>
                    <p className="text-xs text-zinc-600 mt-1">or click to browse · JPEG, PNG, DICOM (.dcm)</p>
                  </div>
                </div>
              )}
            </div>

            {/* Slice depth selector slider */}
            {files.length > 1 && (
              <div className="space-y-2 border-b border-zinc-800 pb-4">
                <div className="flex justify-between items-center text-xs">
                  <Label className="text-zinc-500 uppercase tracking-widest font-semibold">Axial scan depth</Label>
                  <span className="text-zinc-400 font-bold font-mono">Slice {currentSliceIndex + 1} of {files.length}</span>
                </div>
                <Slider
                  min={0}
                  max={files.length - 1}
                  step={1}
                  value={[currentSliceIndex]}
                  onValueChange={(val) => {
                    setIsPlaying(false);
                    const idx = Array.isArray(val) ? val[0] : val;
                    setCurrentSliceIndex(idx);
                  }}
                />
              </div>
            )}

            {/* ── Model Selection ── */}
            <div className="space-y-2">
              <Label className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">AI Model</Label>
              <Select value={model} onValueChange={(val) => setModel(val || "")}>
                <SelectTrigger className="bg-zinc-800 border-zinc-700 text-zinc-100 rounded-xl h-10">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-zinc-900 border-zinc-700 rounded-xl">
                  {availableModels.map((m) => (
                    <SelectItem key={m.id} value={m.id} className="text-zinc-100 focus:bg-zinc-800 rounded-lg py-3 cursor-pointer">
                      <div>
                        <div className="flex items-center gap-2">
                          {m.recommended
                            ? <Zap className="w-3.5 h-3.5 text-blue-400" />
                            : <Activity className="w-3.5 h-3.5 text-emerald-400" />}
                          <span className="font-semibold">{m.name}</span>
                          {m.recommended && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30 font-semibold">
                              Recommended
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-zinc-500 mt-0.5">{m.description}</p>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* ── Confidence Threshold ── */}
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-1.5">
                  <Label className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">Confidence Threshold</Label>
                  <div className="group relative">
                    <Info className="w-3 h-3 text-zinc-600 cursor-help" />
                    <div className="absolute bottom-5 left-0 hidden group-hover:block bg-zinc-800 border border-zinc-700 rounded-lg p-2 text-xs text-zinc-400 w-52 z-10 shadow-xl">
                      Predictions with confidence below this threshold are flagged for human review.
                    </div>
                  </div>
                </div>
                <span className="text-sm font-bold text-blue-400 font-mono">{(threshold * 100).toFixed(0)}%</span>
              </div>
              <Slider
                value={[threshold]}
                min={0.3}
                max={1.0}
                step={0.05}
                onValueChange={(v) => setThreshold(typeof v === "number" ? v : v[0])}
              />
              <div className="flex justify-between text-[10px] text-zinc-600">
                <span>30% (lenient)</span>
                <span>100% (strict)</span>
              </div>
            </div>

            {/* ── Submit ── */}
            <Button
              className="w-full h-12 rounded-xl text-base font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg shadow-blue-900/30 transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 disabled:scale-100 disabled:cursor-not-allowed"
              disabled={!file || loading}
              onClick={handleSubmit}
            >
              {loading ? (
                <><Loader2 className="w-4 h-4 animate-spin mr-2" />Analyzing...</>
              ) : (
                <><Brain className="w-4 h-4 mr-2" />Run Analysis</>
              )}
            </Button>

            {/* ── Error ── */}
            {error && (
              <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <p className="text-sm text-red-300">{error}</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* ── Right: Results ── */}
      <Card className="bg-zinc-900/60 border-zinc-800 rounded-2xl shadow-2xl overflow-hidden">
        <CardHeader className="border-b border-zinc-800 pb-4">
          <CardTitle className="flex items-center gap-2.5 text-zinc-100">
            <div className="w-8 h-8 rounded-xl bg-emerald-500/20 flex items-center justify-center">
              <BarChart3 className="w-4 h-4 text-emerald-400" />
            </div>
            Analysis Results
          </CardTitle>
          <CardDescription className="text-zinc-500">
            Prediction, Grad-CAM heatmap, and class probabilities.
          </CardDescription>
        </CardHeader>

        <CardContent className="pt-5 space-y-5 overflow-y-auto max-h-[70vh]">
          {/* Empty State */}
          {!prediction && !loading && (
            <div className="flex flex-col items-center justify-center py-16 text-zinc-700">
              <div className="w-20 h-20 rounded-3xl bg-zinc-800/50 flex items-center justify-center mb-4">
                <ImageIcon className="w-10 h-10 opacity-40" />
              </div>
              <p className="font-semibold text-zinc-500">No results yet</p>
              <p className="text-sm text-zinc-600 mt-1">Upload a scan and run analysis</p>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="flex flex-col items-center justify-center py-16 gap-4">
              <div className="relative">
                <div className="w-16 h-16 rounded-full border-4 border-blue-500/20 border-t-blue-500 animate-spin" />
                <Brain className="w-7 h-7 text-blue-400 absolute inset-0 m-auto" />
              </div>
              <div className="text-center">
                <p className="font-semibold text-zinc-300">Processing MRI scan...</p>
                <p className="text-sm text-zinc-500 mt-1">Running {model} + Grad-CAM</p>
              </div>
            </div>
          )}

          {/* Results */}
          {prediction && !loading && predictedCfg && (
            <div className="space-y-5 result-enter">
              {/* Prediction Banner */}
              <div className={`rounded-xl p-4 border ${predictedCfg.bg}`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs font-bold uppercase tracking-widest opacity-70 mb-1">Research Dataset-Label Output</p>
                    <p className="text-2xl font-black">{predictedCfg.label}</p>
                    {prediction.model_used && (
                      <p className="text-xs opacity-60 mt-0.5">via {prediction.model_used}</p>
                    )}
                  </div>
                  <div className="text-right">
                    <p className="text-xs font-bold uppercase tracking-widest opacity-70 mb-1">Confidence</p>
                    <p className="text-2xl font-black">{(prediction.confidence * 100).toFixed(1)}%</p>
                    <Badge className={`text-[10px] mt-1 ${prediction.is_uncertain ? "bg-amber-500/20 text-amber-300 border-amber-500/30" : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"}`}>
                      {prediction.is_uncertain ? "Uncertain" : "High Confidence"}
                    </Badge>
                  </div>
                </div>
              </div>

              {/* Uncertainty Entropy Gauge */}
              {prediction.entropy !== undefined && (
                <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950/20 space-y-3">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-zinc-500 uppercase tracking-widest font-semibold flex items-center gap-1.5">
                      <Activity className="w-3.5 h-3.5 text-blue-400" />
                      Prediction Entropy Index
                    </span>
                    <span className={`font-mono font-bold text-sm ${
                      prediction.entropy < 0.3 ? "text-emerald-400" :
                      prediction.entropy < 0.6 ? "text-amber-400" : "text-red-400"
                    }`}>
                      {prediction.entropy.toFixed(3)}
                    </span>
                  </div>

                  <div className="h-3.5 rounded-full bg-zinc-900 border border-zinc-800/80 p-0.5 overflow-hidden flex relative">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        prediction.entropy < 0.3 ? "bg-emerald-500" :
                        prediction.entropy < 0.6 ? "bg-amber-500" : "bg-red-500"
                      }`}
                      style={{ width: `${prediction.entropy * 100}%` }}
                    />
                    {/* Tick markers */}
                    <div className="absolute left-[30%] top-0 w-0.5 h-full bg-zinc-800" />
                    <div className="absolute left-[60%] top-0 w-0.5 h-full bg-zinc-800" />
                  </div>

                  <div className="flex justify-between text-[9px] text-zinc-500 font-semibold">
                    <span className="text-emerald-500">DECISIVE (&lt;0.3)</span>
                    <span className="text-amber-500">AMBIVALENT (0.3-0.6)</span>
                    <span className="text-red-500">CONFLICTED (&gt;0.6)</span>
                  </div>

                  <p className="text-[10px] text-zinc-500 leading-relaxed font-medium">
                    {prediction.entropy < 0.3
                      ? "The model is highly decisive, concentrating prediction weight on a single outcome class."
                      : prediction.entropy < 0.6
                      ? "The model exhibits moderate decision division. Verify secondary predictions below."
                      : "The model is highly conflicted (prediction entropy exceeds 0.600). Manual radiologist arbitration is required."}
                  </p>
                </div>
              )}

              {/* Uncertainty Warning */}
              {prediction.is_uncertain && (
                <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-semibold text-amber-300">Low Confidence — Manual Review Required</p>
                    <p className="text-xs text-amber-500 mt-0.5">
                      Confidence ({(prediction.confidence * 100).toFixed(1)}%) is below the set threshold ({(threshold * 100).toFixed(0)}%).
                      This prediction should be verified by a radiologist.
                    </p>
                  </div>
                </div>
              )}

              {/* Grad-CAM Heatmap */}
              {prediction.heatmap_base64 ? (
                <div className="space-y-2">
                  <p className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">Grad-CAM Explainability</p>
                  <div className="rounded-xl overflow-hidden border border-zinc-700 bg-black shadow-inner">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${prediction.heatmap_base64}`}
                      alt="Grad-CAM Heatmap"
                      className="w-full object-contain max-h-64"
                    />
                  </div>
                  <p className="text-xs text-zinc-600">🔴 Red regions = most influential areas for this prediction.</p>
                </div>
              ) : (
                <div className="p-3 rounded-xl bg-zinc-800/50 border border-zinc-700 flex items-center gap-2 text-xs text-zinc-500">
                  <FileImage className="w-4 h-4" />
                  Grad-CAM unavailable — model needs to be trained first. Run
                  <code className="font-mono bg-zinc-900 px-1 py-0.5 rounded">python src/train.py</code>
                </div>
              )}

              {/* Class Probabilities */}
              <div className="space-y-3">
                <p className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">Class Probabilities</p>
                {sortedClasses.map(([cls, prob], i) => {
                  const cfg = getClassConfig(cls);
                  return (
                    <div key={cls} className="space-y-1.5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {i === 0 && <ChevronRight className="w-3.5 h-3.5 text-emerald-400" />}
                          <span className={`text-sm font-semibold ${i === 0 ? "text-zinc-100" : "text-zinc-400"}`}>
                            {cfg.label}
                          </span>
                        </div>
                        <span className={`text-sm font-mono font-bold ${i === 0 ? "text-zinc-100" : "text-zinc-500"}`}>
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${cfg.bar} bar-fill`}
                          style={{ width: `${prob * 100}%`, animationDelay: `${i * 0.1}s` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* ── Clinical PDF Report settings & download ── */}
              <div className="border border-zinc-800 rounded-xl p-4 bg-zinc-950/40 space-y-3 mt-6">
                <div className="flex items-center gap-2 text-zinc-300">
                  <FileText className="w-4 h-4 text-emerald-400" />
                  <span className="text-xs uppercase tracking-wider font-bold">Report Generator</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label className="text-[10px] text-zinc-500 uppercase">Patient Name</Label>
                    <Input
                      value={patientName}
                      onChange={(e) => setPatientName(e.target.value)}
                      placeholder="Jane Doe"
                      className="h-8 bg-zinc-900 border-zinc-800 text-xs"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-[10px] text-zinc-500 uppercase">Patient / Study ID</Label>
                    <Input
                      value={patientId}
                      onChange={(e) => setPatientId(e.target.value)}
                      placeholder="MR-9804"
                      className="h-8 bg-zinc-900 border-zinc-800 text-xs"
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px] text-zinc-500 uppercase">Research Notes</Label>
                  <textarea
                    value={comments}
                    onChange={(e) => setComments(e.target.value)}
                    placeholder="Enter non-clinical research observations or quality-control notes..."
                    rows={2}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus-visible:ring-blue-500/50"
                  />
                </div>
                <div className="flex gap-2 pt-1">
                  <Button
                    onClick={handleDownloadReport}
                    disabled={pdfLoading}
                    className="flex-1 h-9 text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg gap-1.5"
                  >
                    {pdfLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FileText className="w-3.5 h-3.5" />}
                    Research PDF
                  </Button>
                  {prediction.fhir_metadata && (
                    <Button
                      variant="outline"
                      onClick={() => {
                        const jsonStr = JSON.stringify(prediction.fhir_metadata, null, 2);
                        const blob = new Blob([jsonStr], { type: "application/json" });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = `FHIR_Research_Observation_${patientId || "Scan"}.json`;
                        document.body.appendChild(a);
                        a.click();
                        URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                      }}
                      className="h-9 text-xs font-semibold bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 rounded-lg gap-1.5"
                    >
                      <Download className="w-3.5 h-3.5" />
                      FHIR Observation
                    </Button>
                  )}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
