Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting FastAPI Backend Server on port 8000..." -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

Set-Location -Path "$PSScriptRoot\backend"

# Auto-create .env from .env.example if missing
if (-not (Test-Path "$PSScriptRoot\backend\.env")) {
    if (Test-Path "$PSScriptRoot\backend\.env.example") {
        Write-Host "Creating backend/.env from .env.example..." -ForegroundColor Yellow
        Copy-Item -Path "$PSScriptRoot\backend\.env.example" -Destination "$PSScriptRoot\backend\.env"
    }
}

$venvActivate = "$PSScriptRoot\.venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

uvicorn app.main:app --reload --port 8000
