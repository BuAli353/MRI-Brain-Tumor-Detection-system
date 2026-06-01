#!/bin/bash

# Kill any existing processes hanging on the ports
echo "🧹 Cleaning up old background processes on ports 8000 and 7861..."
fuser -k 8000/tcp 2>/dev/null
fuser -k 7861/tcp 2>/dev/null
sleep 2

# Start the FastAPI backend in the background
echo "🚀 Starting backend server on port 8000..."
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for the backend to be ready
echo "⏳ Waiting for backend to initialize..."
sleep 8

# Start the Gradio frontend
echo "🚀 Starting frontend..."
python3 frontend/app.py

# When the frontend exits, kill the backend
kill $BACKEND_PID
