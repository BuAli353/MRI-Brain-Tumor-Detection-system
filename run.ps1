# Windows PowerShell Runner for MRI Brain Tumor Detection

# Set Title
$host.UI.RawUI.WindowTitle = "MRI Brain Tumor Detection Runner"

# Helper to kill process by port
function Kill-PortProcess ($port) {
    $proc = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($proc) {
        Write-Host "🧹 Stopping old process ($($proc.OwningProcess)) on port $port..." -ForegroundColor Yellow
        Stop-Process -Id $proc.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "🧹 Cleaning up old background processes on ports 8000 and 7860/7861..." -ForegroundColor Cyan
Kill-PortProcess 8000
Kill-PortProcess 7860
Kill-PortProcess 7861
Start-Sleep -Seconds 2

# Start Backend
Write-Host "🚀 Starting backend server on port 8000..." -ForegroundColor Green
$backendProcess = Start-Process -FilePath "python" -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000" -PassThru

# Wait for backend
Write-Host "⏳ Waiting for backend to initialize (8 seconds)..." -ForegroundColor Cyan
Start-Sleep -Seconds 8

# Start Frontend
Write-Host "🚀 Starting frontend..." -ForegroundColor Green
python frontend/app.py

# Cleanup Backend on Exit
Write-Host "🧹 Cleaning up backend..." -ForegroundColor Yellow
if ($backendProcess) {
    Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "Done!" -ForegroundColor Green
