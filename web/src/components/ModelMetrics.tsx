"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Activity, Cpu, Sliders, RefreshCw, AlertCircle,
  TrendingUp, Compass, Grid, FileSpreadsheet, Play, Code
} from "lucide-react";
import { getModelMetrics, type ModelMetricsResponse } from "@/lib/api";

export function ModelMetrics() {
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getModelMetrics();
      setMetrics(data);
    } catch (e: any) {
      setError(e.message || "Failed to fetch model metrics from backend.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
  }, [fetchMetrics]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-4 max-w-4xl mx-auto">
        <div className="relative">
          <div className="w-12 h-12 rounded-full border-4 border-blue-500/20 border-t-blue-500 animate-spin" />
          <Activity className="w-5 h-5 text-blue-400 absolute inset-0 m-auto" />
        </div>
        <p className="text-sm text-zinc-400">Loading model performance metrics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="bg-zinc-900/60 border-zinc-800 max-w-2xl mx-auto rounded-2xl shadow-xl overflow-hidden mt-6">
        <CardContent className="pt-6 text-center space-y-4">
          <div className="w-12 h-12 rounded-full bg-red-500/10 flex items-center justify-center mx-auto">
            <AlertCircle className="w-6 h-6 text-red-400" />
          </div>
          <div className="space-y-1.5">
            <p className="font-semibold text-zinc-200">Diagnostics Unavailable</p>
            <p className="text-xs text-zinc-500 leading-relaxed">
              Could not fetch model metrics from the API service. Please verify the FastAPI server is running.
            </p>
          </div>
          <Button onClick={fetchMetrics} size="sm" className="bg-zinc-800 hover:bg-zinc-700 text-zinc-300">
            <RefreshCw className="w-3.5 h-3.5 mr-1" /> Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  const meta = metrics?.metadata;

  return (
    <div className="space-y-6 w-full max-w-6xl mx-auto">
      {/* ── Header Controls ── */}
      <div className="flex justify-between items-center bg-zinc-900/40 border border-zinc-800 p-4 rounded-xl glass">
        <div>
          <h3 className="text-sm font-bold text-zinc-100 flex items-center gap-1.5">
            <Compass className="w-4 h-4 text-blue-400" /> Model Performance Diagnostics
          </h3>
          <p className="text-xs text-zinc-500 mt-0.5">
            Generated test metrics, calibration, uncertainty, and training statistics. Empty until experiments run.
          </p>
        </div>
        <Button onClick={fetchMetrics} variant="outline" size="sm" className="bg-zinc-900/50 border-zinc-800 hover:bg-zinc-800 text-zinc-300">
          <RefreshCw className="w-3.5 h-3.5 mr-1" /> Refresh Diagnostics
        </Button>
      </div>

      {/* ── Active Hyperparameters ── */}
      {meta && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-zinc-900/40 border-zinc-800/80 rounded-xl shadow-md">
            <CardContent className="pt-4 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center">
                <Cpu className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <p className="text-[10px] text-zinc-500 uppercase font-semibold">Active Hardware</p>
                <p className="text-sm font-bold text-zinc-100 font-mono mt-0.5">{meta.device.toUpperCase()}</p>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-zinc-900/40 border-zinc-800/80 rounded-xl shadow-md">
            <CardContent className="pt-4 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center">
                <Sliders className="w-4 h-4 text-purple-400" />
              </div>
              <div>
                <p className="text-[10px] text-zinc-500 uppercase font-semibold">Batch Size</p>
                <p className="text-sm font-bold text-zinc-100 font-mono mt-0.5">{meta.batch_size}</p>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-zinc-900/40 border-zinc-800/80 rounded-xl shadow-md">
            <CardContent className="pt-4 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                <Activity className="w-4 h-4 text-emerald-400" />
              </div>
              <div>
                <p className="text-[10px] text-zinc-500 uppercase font-semibold">Base Learning Rate</p>
                <p className="text-sm font-bold text-zinc-100 font-mono mt-0.5">{meta.learning_rate.toExponential(0)}</p>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-zinc-900/40 border-zinc-800/80 rounded-xl shadow-md">
            <CardContent className="pt-4 flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center">
                <TrendingUp className="w-4 h-4 text-amber-400" />
              </div>
              <div>
                <p className="text-[10px] text-zinc-500 uppercase font-semibold">Max Training Epochs</p>
                <p className="text-sm font-bold text-zinc-100 font-mono mt-0.5">{meta.epochs_max}</p>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* ── Main Diagnostics Content ── */}
      {metrics && !metrics.trained ? (
        /* ── Case 1: Model NOT Trained Yet ── */
        <Card className="bg-zinc-900/60 border-zinc-800 rounded-2xl shadow-xl overflow-hidden">
          <CardHeader className="border-b border-zinc-800 pb-4">
            <div className="flex items-center gap-2">
              <Badge className="bg-amber-500/10 border-amber-500/20 text-amber-400">Results Not Available</Badge>
              <CardTitle className="text-zinc-100 text-lg">Diagnostics Report Empty</CardTitle>
            </div>
            <CardDescription className="text-zinc-500 mt-1">
              No training history or validation metrics were discovered in the outputs folder.
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-6 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Instructions on training */}
              <div className="space-y-4">
                <h4 className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                  <Play className="w-4 h-4 text-blue-400" /> How to Train the Model
                </h4>
                <p className="text-xs text-zinc-400 leading-relaxed">
                  To view accuracy curves, confusion matrices, and ROC evaluations, you must first execute the training script locally. This generates checkpoints and saves evaluation results inside `outputs/metrics/`.
                </p>
                <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800/80 space-y-2">
                  <p className="text-[11px] font-bold text-zinc-500 flex items-center gap-1.5">
                    <Code className="w-3.5 h-3.5 text-zinc-600" /> Command Line Prompt
                  </p>
                  <pre className="text-xs font-mono text-zinc-300 bg-zinc-900 p-2.5 rounded border border-zinc-800 overflow-x-auto">
                    python src/train.py --model efficientnet
                  </pre>
                  <p className="text-[10px] text-zinc-600 italic">
                    Note: Supports standard hardware accelerators (CUDA on Linux/Windows, MPS on Apple Silicon).
                  </p>
                </div>
              </div>

              {/* Class configuration Details */}
              <div className="space-y-4">
                <h4 className="text-sm font-bold text-zinc-200 flex items-center gap-2">
                  <FileSpreadsheet className="w-4 h-4 text-purple-400" /> Active Target Classes
                </h4>
                <p className="text-xs text-zinc-400 leading-relaxed">
                  The model architecture is compiled to perform a 4-class single-label categorization:
                </p>
                <div className="grid grid-cols-2 gap-2">
                  {meta?.classes.map((cls) => (
                    <div key={cls} className="p-3 bg-zinc-800/40 border border-zinc-700/30 rounded-lg text-center">
                      <span className="text-xs font-semibold text-zinc-300 capitalize">
                        {cls.replace("_", " ")}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : (
        /* ── Case 2: Model Trained with Metrics ── */
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Classification Report */}
            <Card className="bg-zinc-900/60 border-zinc-800 rounded-2xl shadow-xl overflow-hidden flex flex-col">
              <CardHeader className="border-b border-zinc-800 pb-4">
                <CardTitle className="text-zinc-100 text-lg flex items-center gap-2">
                  <Grid className="w-4 h-4 text-purple-400" /> Classification Report
                </CardTitle>
                <CardDescription className="text-zinc-500">
                  Precision, Recall, and F1 performance measures.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-6 flex-1 flex flex-col justify-between">
                {metrics?.classification_report_text ? (
                  <pre className="text-xs font-mono text-zinc-300 bg-zinc-950 p-4 rounded-xl border border-zinc-800/60 overflow-x-auto leading-relaxed h-[320px]">
                    {metrics.classification_report_text}
                  </pre>
                ) : (
                  <div className="text-center py-20 text-zinc-600 text-xs">
                    Report details unavailable.
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Confusion Matrix */}
            <Card className="bg-zinc-900/60 border-zinc-800 rounded-2xl shadow-xl overflow-hidden">
              <CardHeader className="border-b border-zinc-800 pb-4">
                <CardTitle className="text-zinc-100 text-lg flex items-center gap-2">
                  <Grid className="w-4 h-4 text-emerald-400" /> Confusion Matrix
                </CardTitle>
                <CardDescription className="text-zinc-500">
                  Visualizes actual classes against model predicted classes.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-6 flex justify-center bg-black/40">
                {metrics?.confusion_matrix_base64 ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img
                    src={`data:image/png;base64,${metrics.confusion_matrix_base64}`}
                    alt="Confusion Matrix"
                    className="max-h-[320px] object-contain rounded-lg border border-zinc-800/80"
                  />
                ) : (
                  <div className="text-center py-20 text-zinc-600 text-xs">
                    Confusion matrix image not found.
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Training History curves */}
            <Card className="bg-zinc-900/60 border-zinc-800 rounded-2xl shadow-xl overflow-hidden">
              <CardHeader className="border-b border-zinc-800 pb-4">
                <CardTitle className="text-zinc-100 text-lg flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-blue-400" /> Training History Curves
                </CardTitle>
                <CardDescription className="text-zinc-500">
                  Loss, accuracy, and learning rate progression over training epochs.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-6 flex justify-center bg-black/40">
                {metrics?.training_history_base64 ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img
                    src={`data:image/png;base64,${metrics.training_history_base64}`}
                    alt="Training History Curves"
                    className="max-h-[320px] object-contain rounded-lg border border-zinc-800/80"
                  />
                ) : (
                  <div className="text-center py-20 text-zinc-600 text-xs">
                    History curves image not found.
                  </div>
                )}
              </CardContent>
            </Card>

            {/* ROC Curves */}
            <Card className="bg-zinc-900/60 border-zinc-800 rounded-2xl shadow-xl overflow-hidden">
              <CardHeader className="border-b border-zinc-800 pb-4">
                <CardTitle className="text-zinc-100 text-lg flex items-center gap-2">
                  <Activity className="w-4 h-4 text-amber-400" /> Receiver Operating Characteristic (ROC)
                </CardTitle>
                <CardDescription className="text-zinc-500">
                  AUC (Area Under the Curve) statistics across classification classes.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-6 flex justify-center bg-black/40">
                {metrics?.roc_curves_base64 ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img
                    src={`data:image/png;base64,${metrics.roc_curves_base64}`}
                    alt="ROC Curves"
                    className="max-h-[320px] object-contain rounded-lg border border-zinc-800/80"
                  />
                ) : (
                  <div className="text-center py-20 text-zinc-600 text-xs">
                    ROC curves image not found.
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="bg-zinc-900/60 border-zinc-800 rounded-2xl shadow-xl overflow-hidden">
              <CardHeader className="border-b border-zinc-800 pb-4">
                <CardTitle className="text-zinc-100 text-lg flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" /> Calibration
                </CardTitle>
                <CardDescription className="text-zinc-500">
                  Reliability after validation-only temperature scaling.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-6 flex justify-center bg-black/40">
                {metrics?.calibration_curve_base64 ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img src={`data:image/png;base64,${metrics.calibration_curve_base64}`} alt="Calibration curve" className="max-h-[320px] object-contain rounded-lg border border-zinc-800/80" />
                ) : <div className="text-center py-20 text-zinc-600 text-xs">Calibration evidence not generated.</div>}
              </CardContent>
            </Card>
            <Card className="bg-zinc-900/60 border-zinc-800 rounded-2xl shadow-xl overflow-hidden">
              <CardHeader className="border-b border-zinc-800 pb-4">
                <CardTitle className="text-zinc-100 text-lg flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-red-400" /> Risk–Coverage
                </CardTitle>
                <CardDescription className="text-zinc-500">
                  Error risk as low-confidence cases are withheld.
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-6 flex justify-center bg-black/40">
                {metrics?.risk_coverage_base64 ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img src={`data:image/png;base64,${metrics.risk_coverage_base64}`} alt="Risk coverage curve" className="max-h-[320px] object-contain rounded-lg border border-zinc-800/80" />
                ) : <div className="text-center py-20 text-zinc-600 text-xs">Risk–coverage evidence not generated.</div>}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
