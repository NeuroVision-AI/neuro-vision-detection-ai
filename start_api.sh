#!/bin/bash
# Start the FastAPI backend
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR/api"
source venv/bin/activate
echo "▶ Starting FastAPI backend on http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"
echo ""
uvicorn main:app --reload --port 8000 --host 0.0.0.0
