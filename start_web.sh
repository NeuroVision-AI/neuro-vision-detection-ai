#!/bin/bash
# Start the Next.js frontend
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR/web"
echo "▶ Starting Next.js frontend on http://localhost:3000"
echo ""
npm run dev
