#!/bin/bash
# master.sh — Project Xceed cold-start launcher
# Usage: bash master.sh
# Starts: FastAPI backend → Streamlit dashboard → detect.py (foreground)
# Ctrl+C stops all three cleanly.

set -e

PROJECT_DIR="$HOME/project-xceed"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
source "$HOME/xceed-env/bin/activate"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

BACKEND_PORT=8000
FRONTEND_PORT=8501

# ── Cleanup on exit ──────────────────────────────────
cleanup() {
    echo ""
    echo "[master.sh] Shutting down..."
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID"
        echo "[master.sh] Backend stopped."
    fi
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID"
        echo "[master.sh] Frontend stopped."
    fi
    echo "[master.sh] All services stopped."
}
trap cleanup EXIT INT TERM

# ── Step 1: verify model exists ──────────────────────
MODEL="$PROJECT_DIR/ai/best320.onnx"
if [ ! -f "$MODEL" ]; then
    echo "[ERROR] Model not found: $MODEL"
    echo "        Copy best.onnx to $PROJECT_DIR/ai/ before running."
    exit 1
fi
echo "[master.sh] Model found: $MODEL"

# ── Step 2: start FastAPI backend ────────────────────
echo "[master.sh] Starting FastAPI backend on port $BACKEND_PORT..."
cd "$PROJECT_DIR"
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port $BACKEND_PORT \
    --log-level warning \
    > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
for i in {1..10}; do
    if curl -s "http://localhost:$BACKEND_PORT/status" > /dev/null 2>&1; then
        echo "[master.sh] Backend ready (PID $BACKEND_PID)"
        break
    fi
    sleep 1
    if [ $i -eq 10 ]; then
        echo "[ERROR] Backend did not start. Check $BACKEND_LOG"
        cat "$BACKEND_LOG"
        exit 1
    fi
done

# ── Step 3: start Streamlit frontend ─────────────────
echo "[master.sh] Starting Streamlit dashboard on port $FRONTEND_PORT..."
streamlit run "$PROJECT_DIR/frontend/dashboard.py" \
    --server.port $FRONTEND_PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

sleep 3
if kill -0 "$FRONTEND_PID" 2>/dev/null; then
    # Get Pi IP for dashboard URL
    PI_IP=$(hostname -I | awk '{print $1}')
    echo "[master.sh] Dashboard ready → http://${PI_IP}:${FRONTEND_PORT}"
else
    echo "[WARNING] Streamlit may have failed. Check $FRONTEND_LOG"
fi

# ── Step 4: run inference (foreground) ───────────────
echo ""
echo "============================================================"
echo "  Project Xceed — running"
echo "  Backend  : http://localhost:${BACKEND_PORT}"
PI_IP=$(hostname -I | awk '{print $1}')
echo "  Dashboard: http://${PI_IP}:${FRONTEND_PORT}"
echo "  Press Ctrl+C to stop everything."
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
source "$HOME/xceed-env/bin/activate"
env | sort > /tmp/env_master.txt
python3 ai/detect.py
