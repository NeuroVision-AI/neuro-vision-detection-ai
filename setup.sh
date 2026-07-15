#!/bin/bash
# =============================================================================
# AI NeuroOnco — Full Project Setup Script
# =============================================================================
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# What this does:
#   1. Creates Python venv for the backend
#   2. Installs all Python dependencies
#   3. Installs Node.js dependencies for the frontend
#   4. Creates .env file from template if it doesn't exist
#   5. Checks for Ollama and prints setup instructions
# =============================================================================

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_DIR="$ROOT_DIR/api"
WEB_DIR="$ROOT_DIR/web"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         AI NeuroOnco — Setup Script                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Python venv ──────────────────────────────────────────────────────────────
echo "▶ Setting up Python virtual environment..."
cd "$API_DIR"

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
else
    PYTHON_BIN="python3"
fi

if [ ! -d "venv" ]; then
    "$PYTHON_BIN" -m venv venv
    echo "  ✓ Created venv with $PYTHON_BIN"
else
    echo "  ✓ venv already exists"
fi

source venv/bin/activate
echo "  ✓ Activated venv"

# ── Install Python packages ────────────────────────────────────────────────────
echo ""
echo "▶ Installing Python dependencies..."
pip install --quiet --no-cache-dir --upgrade pip
pip install --quiet --no-cache-dir -r "$ROOT_DIR/requirements.txt"
pip install --quiet --no-cache-dir -r requirements.txt
echo "  ✓ Python packages installed"

# ── .env file ──────────────────────────────────────────────────────────────────
echo ""
echo "▶ Setting up environment file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ✓ Created api/.env from template"
    echo "  ⚠️  Edit api/.env to add your API keys (GOOGLE_API_KEY, OPENAI_API_KEY) for cloud LLMs"
else
    echo "  ✓ api/.env already exists"
fi

# ── Node.js / npm ─────────────────────────────────────────────────────────────
echo ""
echo "▶ Installing Node.js dependencies..."
cd "$WEB_DIR"

if [ ! -d "node_modules" ]; then
    npm install --silent
    echo "  ✓ npm packages installed"
else
    echo "  ✓ node_modules already exists"
fi

# ── Ollama check ──────────────────────────────────────────────────────────────
echo ""
echo "▶ Checking Ollama..."
if command -v ollama &>/dev/null; then
    echo "  ✓ Ollama is installed: $(ollama --version)"
    echo ""
    echo "  Checking if medgemma:latest is available..."
    if ollama list 2>/dev/null | grep -q "medgemma"; then
        echo "  ✓ medgemma:latest is already pulled"
    else
        echo "  ⚠️  medgemma:latest not found locally."
        echo "  Run: ollama pull medgemma:latest"
    echo "  (This is a large research model download; it is not a clinical system.)"
    fi
else
    echo "  ⚠️  Ollama not found. Install it first:"
    echo "  macOS:  curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Then:   ollama pull medgemma:latest"
fi

# ── Output directories ─────────────────────────────────────────────────────────
echo ""
echo "▶ Creating output directories..."
cd "$ROOT_DIR"
mkdir -p data/raw data/manifests data/processed/train data/processed/val data/processed/test
mkdir -p data/external_manifests data/external_processed
mkdir -p outputs/models outputs/logs outputs/metrics outputs/heatmaps
echo "  ✓ Directory structure created"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup complete! Next steps:                              ║"
echo "║                                                              ║"
echo "║  1. Download data:                                           ║"
echo "║     cd $ROOT_DIR                                 "
echo "║     python scripts/download_data.py --exclude-conflicting-groups ║"
echo "║                                                              ║"
echo "║  2. Train the model:                                         ║"
echo "║     python -m src.train --model efficientnet                 ║"
echo "║                                                              ║"
echo "║  3. Start the backend (in one terminal):                     ║"
echo "║     ./start_api.sh                                           ║"
echo "║                                                              ║"
echo "║  4. Start the frontend (in another terminal):                ║"
echo "║     ./start_web.sh                                           ║"
echo "║                                                              ║"
echo "║  5. Open http://localhost:3000                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
