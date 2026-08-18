Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting FastAPI Backend Server on port 8000..." -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

Set-Location -Path "$PSScriptRoot\backend"

$venvActivate = "$PSScriptRoot\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

uvicorn app.main:app --reload --port 8000
