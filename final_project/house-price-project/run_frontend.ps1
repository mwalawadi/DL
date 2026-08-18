Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting React Frontend on port 5173..." -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

Set-Location -Path "$PSScriptRoot\frontend"

# Auto-create .env from .env.example if missing
if (-not (Test-Path "$PSScriptRoot\frontend\.env")) {
    if (Test-Path "$PSScriptRoot\frontend\.env.example") {
        Write-Host "Creating frontend/.env from .env.example..." -ForegroundColor Yellow
        Copy-Item -Path "$PSScriptRoot\frontend\.env.example" -Destination "$PSScriptRoot\frontend\.env"
    }
}

npm run dev
