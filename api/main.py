"""
FastAPI Application — AI NeuroOnco Backend

Routes:
  GET  /              → Health check
  GET  /docs          → Interactive Swagger UI
  POST /predict/      → MRI classification + Grad-CAM
  GET  /predict/models→ List model architectures
  GET  /predict/health→ Model service health
  POST /rag/upload    → Ingest document into ChromaDB
  POST /rag/chat      → Chat with RAG pipeline
  GET  /rag/stats     → Knowledge base statistics
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'src' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import predict, rag

app = FastAPI(
    title="AI NeuroOnco API",
    description=(
        "Brain tumor MRI classification (EfficientNet / Custom CNN + Grad-CAM) "
        "and Medical Literature RAG pipeline (MedGemma / Gemini / OpenAI via LangChain + ChromaDB)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(predict.router)
app.include_router(rag.router)


# ── Root Health Check ─────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    """API root — returns service status."""
    return {
        "status": "ok",
        "message": "AI NeuroOnco API is running",
        "version": "1.0.0",
        "docs": "/docs",
    }
