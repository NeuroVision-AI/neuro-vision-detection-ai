/**
 * Centralized API client for the AI NeuroOnco backend.
 * All fetch calls go through this module so the base URL is configured once.
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface PredictionResult {
  predicted_class: string;
  confidence: number;
  is_uncertain: boolean;
  heatmap_base64: string | null;
  predictions: Record<string, number>;
  model_used?: string;
  entropy?: number;
  fhir_metadata?: any;
  calibrated?: boolean;
  temperature?: number;
  checkpoint?: string;
  intended_use?: string;
  input_scope?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  params: string;
  recommended: boolean;
  trained: boolean;
}

export interface ModelHealthResponse {
  status: string;
  device: string;
  torch_version: string;
  cuda_available: boolean;
  mps_available: boolean;
  num_classes: number;
  class_names: string[];
  cached_models: string[];
}

export interface KBStats {
  document_chunks: number;
  status: string;
}

export interface RAGSource {
  source: string;
  page: number | null;
  snippet: string;
  row?: number | null;
  doi?: string | null;
  link?: string | null;
  record_id?: string | null;
  title?: string | null;
  corpus_version?: string | null;
  chunk_id?: string | null;
}

export interface ChatResponse {
  answer: string;
  provider: string;
  sources: RAGSource[];
}

export interface ReportParams {
  predicted_class: string;
  confidence: number;
  predictions: Record<string, number>;
  model_used: string;
  heatmap_base64: string | null;
  patient_name?: string;
  patient_id?: string;
  comments?: string;
  calibrated?: boolean;
  confidence_threshold?: number;
}

export interface ModelMetricsResponse {
  trained: boolean;
  metrics_directory: string;
  confusion_matrix_base64: string | null;
  roc_curves_base64: string | null;
  training_history_base64: string | null;
  calibration_curve_base64: string | null;
  risk_coverage_base64: string | null;
  classification_report_text: string | null;
  classification_report_json: any | null;
  research_metrics_json: any | null;
  metadata: {
    device: string;
    batch_size: number;
    learning_rate: number;
    epochs_max: number;
    early_stopping_patience: number;
    classes: string[];
    default_model: string;
  };
}

// ── API Methods ────────────────────────────────────────────────────────────────

/** Check if the backend API is reachable. */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

/** Run MRI classification. */
export async function predictMRI(
  file: File,
  modelName: string,
  threshold: number
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_name", modelName);
  formData.append("threshold", threshold.toString());

  const res = await fetch(`${API_BASE}/predict/`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

/** Download generated PDF report. */
export async function downloadPDFReport(params: ReportParams): Promise<Blob> {
  const res = await fetch(`${API_BASE}/predict/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Failed to build report" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.blob();
}

/** List available model architectures. */
export async function listModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/predict/models`);
  if (!res.ok) throw new Error("Failed to fetch models");
  const data = await res.json();
  return data.models;
}

/** Get model service health info. */
export async function getModelHealth(): Promise<ModelHealthResponse> {
  const res = await fetch(`${API_BASE}/predict/health`);
  if (!res.ok) throw new Error("Model health check failed");
  return res.json();
}

/** Upload a document to the knowledge base. */
export async function uploadDocument(file: File): Promise<{ status: string; chunks_ingested: number; filename: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/rag/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

/** Send a chat message to the RAG pipeline. */
export async function sendChatMessage(
  message: string,
  provider: string,
  imageBase64?: string | null
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/rag/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, provider, image_base64: imageBase64 }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

/** Get knowledge base statistics. */
export async function getKBStats(): Promise<KBStats> {
  const res = await fetch(`${API_BASE}/rag/stats`);
  if (!res.ok) throw new Error("Failed to fetch KB stats");
  return res.json();
}

/** Get model training metrics. */
export async function getModelMetrics(): Promise<ModelMetricsResponse> {
  const res = await fetch(`${API_BASE}/predict/metrics`);
  if (!res.ok) throw new Error("Failed to fetch model metrics");
  return res.json();
}
