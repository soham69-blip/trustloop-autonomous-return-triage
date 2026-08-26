# ============================================================
# TrustLoop Demo Startup Script (PowerShell)
# Launches the FastAPI engine and opens the Case Room web application
# ============================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  TrustLoop — AI Return Investigation & Responsibility Engine" -ForegroundColor Cyan
Write-Host "  Starting Demo Environment on http://127.0.0.1:8080..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Set Python Path to workspace root
$env:PYTHONPATH = "."

# Check Python virtual environment
$pythonExe = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "Error: Virtual environment python.exe not found at $pythonExe" -ForegroundColor Red
    exit 1
}

# Verify model hash integrity before launching
Write-Host "`n[1/3] Verifying Production Model Hash Integrity..." -ForegroundColor Yellow
& $pythonExe -c "
import hashlib
from pathlib import Path
p = Path('models/lightgbm_model.pkl')
expected = 'db3a6c03149fa096c7df0d2df214043236f9cb8a8b0fe8ea198d1c24ea94d485'
actual = hashlib.sha256(p.read_bytes()).hexdigest()
if actual.lower() == expected.lower():
    print('  [PASS] Production LightGBM Model SHA256 Verified (db3a6c03149fa096...)')
else:
    print('  [FAIL] Model hash mismatch!')
    exit(1)
"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Model verification failed. Aborting startup." -ForegroundColor Red
    exit 1
}

# Launch FastAPI server with Uvicorn
Write-Host "`n[2/3] Starting FastAPI Server on http://127.0.0.1:8080..." -ForegroundColor Yellow
Write-Host "`n[3/3] Opening Case Room Web Application in browser..." -ForegroundColor Green
Start-Process "http://127.0.0.1:8080/"

& $pythonExe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8080 --reload
