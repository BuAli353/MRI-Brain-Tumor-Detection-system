@echo off
title MRI Brain Tumor Detection Runner

echo 🧹 Cleaning up old background processes on ports 8000 and 7860/7861...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7860" ^| findstr "LISTENING"') do taskkill /f /pid %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7861" ^| findstr "LISTENING"') do taskkill /f /pid %%a 2>nul
timeout /t 2 >nul

echo 🚀 Starting backend server on port 8000...
start "MRI Backend" cmd /c "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo ⏳ Waiting for backend to initialize (8 seconds)...
timeout /t 8 >nul

echo 🚀 Starting frontend...
python frontend/app.py

echo 🧹 Stopping backend server...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a 2>nul

echo Done!
pause
